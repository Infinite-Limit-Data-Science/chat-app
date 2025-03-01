import os
import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Generator
from redis.client import Redis
from redis.connection import ConnectionPool
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from ..chat_bot_config import (
    LLMConfig,
    EmbeddingsConfig,
    RedisVectorStoreConfig,
    MongoMessageHistoryConfig,
    ChatBotConfig
) 
from ..chat_bot import ChatBot, StreamingResponse

import io
from fastapi import UploadFile
from ...langchain_doc import ingest

from bson import ObjectId
from uuid import uuid4
from pymongo import MongoClient
from pymongo.database import Database

@pytest.fixture
def redis_client() -> Redis:
    redis_client = Redis.from_pool(
        ConnectionPool.from_url(
            os.environ['REDIS_URL'], 
            max_connections=50,
            socket_timeout=30.0
        )
    )

    return redis_client

@pytest.fixture
def chat_bot_config(redis_client: Redis) -> ChatBotConfig:
    models = json.loads(os.environ['MODELS'])
    chat_model = models[0]
    llm_config = LLMConfig(**{
        'name': chat_model['name'],
        'endpoint': chat_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'parameters': { 'temperature': 0.8 },
        'server': 'tgi',
    })

    guardrails_model = models[-1]
    guardrails_config = LLMConfig(**{
        'name': guardrails_model['name'],
        'endpoint': guardrails_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'parameters': {},
        'server': 'tgi',
    })

    embeddings = json.loads(os.environ['EMBEDDING_MODELS'])
    embeddings_model = embeddings[0]
    embeddings_config = EmbeddingsConfig(**{
        'name': embeddings_model['name'],
        'endpoint': embeddings_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'dimensions': embeddings_model['dimensions'],
        'server': 'tei',
    })

    metadata_schema = json.loads(os.environ['VECTOR_STORE_SCHEMA'])
    vectorstore_config = RedisVectorStoreConfig(**{
        'client': redis_client,
        'metadata_schema': metadata_schema,
    })

    message_history_config = MongoMessageHistoryConfig(
        url=os.environ['MONGODB_URL'],
        name=os.environ['DATABASE_NAME'],
        collection_name='messages',
        session_id_key='conversation_id',
    )

    return ChatBotConfig(
        llm=llm_config,
        embeddings=embeddings_config,
        guardrails=guardrails_config,
        vectorstore=vectorstore_config,
        message_history=message_history_config,
    )

@pytest.fixture
def pdf_documents() -> List[UploadFile]:
    from starlette.datastructures import Headers

    current_dir = Path(__file__).parent
    pdf_path = current_dir / 'assets' / 'NVIDIAAn.pdf'

    with pdf_path.open('rb') as f:
        file_data = f.read()
    
    upload = UploadFile(
        file=io.BytesIO(file_data),
        filename='NVIDIAAn.pdf',
        headers=Headers({'content-type': 'application/pdf'}),
    )
    return [upload]

@pytest.fixture
def large_pdf_documents() -> List[UploadFile]:
    from starlette.datastructures import Headers

    current_dir = Path(__file__).parent
    pdf_path = current_dir / 'assets' / 'Calculus.pdf'

    with pdf_path.open('rb') as f:
        file_data = f.read()
    
    upload = UploadFile(
        file=io.BytesIO(file_data),
        filename='Calculus.pdf',
        headers=Headers({'content-type': 'application/pdf'}),
    )
    return [upload]

@pytest.fixture
def messages_db(chat_bot_config: ChatBotConfig) -> Database:
    url = chat_bot_config.message_history.url
    database = chat_bot_config.message_history.name

    client = MongoClient(url)
    db = client[database]
    return db

@pytest.fixture
def conversation_doc(messages_db: Database) -> Generator[Dict[str, Any], None, None]:
    conversations = messages_db['conversations']

    attributes = {
        '_id': ObjectId(),
        'sessionId': str(uuid4()),
    }

    doc = conversations.insert_one(attributes)
    attributes['_id'] = doc.inserted_id

    yield attributes

    conversations.delete_one({'_id': attributes['_id']})

@pytest.fixture
def vector_metadata(
    conversation_doc: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        'uuid': conversation_doc['sessionId'],
        'conversation_id': conversation_doc['_id'],
        # 'filename': pdf_documents[0].filename,
    }

@pytest.mark.asyncio
async def test_single_doc_prompt(
    chat_bot_config: ChatBotConfig,
    vector_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
    pdf_documents: List[UploadFile],
):
    chat_bot_config.message_history.session_id = conversation_doc['_id']
    vector_store = os.environ['VECTOR_STORE']
    metadatas = await ingest(
        vector_store, 
        pdf_documents, 
        chat_bot_config.embeddings,
        chat_bot_config.vectorstore,
        vector_metadata,
    )    

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            ('human', '{input}')
        ]
    )
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot

    config = RunnableConfig(
        tags=['chat_bot_run_test', f'uuid_${vector_metadata['uuid']}', f'conversation_id_${vector_metadata['uuid']}'],
        metadata={ 'vector_metadata': metadatas },
        configurable={ 'retrieval_mode': 'mmr' }
    )

    ai_content = ''
    streaming_resp = []
    async for chunk in chain.astream(
        {'input': 'Summarize the document'},
        config=config
    ):
        print(f'Custom event ${chunk.content}')
        ai_content += chunk.content
        streaming_resp.append(chunk)
    
    assert len(ai_content) > 0
    assert streaming_resp[-1].token_usage
    
@pytest.mark.asyncio
async def test_multi_doc_prompt(
    chat_bot_config: ChatBotConfig,
    vector_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
    pdf_documents: List[UploadFile],
):
    ...

        # I NEED TO DO A RUNNABLE_PARALLEL AND PASS retrieval_mode mmr for one and retrieval_mode similarity_search_with_threshold for the other !!!!
        # THIS SHOULD ONLY BE DONE WITH DOCUMENT UPLOADS. IF NO DOCUMENT UPLOAD, THEN ONLY SEND BACK A SINGLE RESPONSE.