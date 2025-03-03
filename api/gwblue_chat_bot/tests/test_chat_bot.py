import os
import json
import pytest
import base64
from pathlib import Path
from typing import Dict, Any, List, Generator
from redis.client import Redis
from redis.connection import ConnectionPool
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableParallel
from ..chat_bot_config import (
    LLMConfig,
    EmbeddingsConfig,
    RedisVectorStoreConfig,
    MongoMessageHistoryConfig,
    ChatBotConfig
) 
from ..chat_bot import ChatBot

import io
from fastapi import UploadFile
from starlette.datastructures import Headers
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
def compare_documents() -> list[UploadFile]:
    current_dir = Path(__file__).parent
    files = [
        current_dir / "assets" / "ruby-rails-developer-v2.docx",
        current_dir / "assets" / "senior-cloud-engineer-v1.docx",
        current_dir / "assets" / "senior-devops-engineer-v4.docx",
    ]

    uploads = []
    for path in files:
        with path.open("rb") as f:
            content = f.read()
        upload = UploadFile(
            file=io.BytesIO(content),
            filename=path.name,
            headers=Headers({"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
        )
        uploads.append(upload)

    return uploads

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

    # conversations.delete_one({'_id': attributes['_id']})

@pytest.fixture
def message_metadata(
    conversation_doc: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        'uuid': conversation_doc['sessionId'],
        'conversation_id': conversation_doc['_id'],
    }

@pytest.mark.asyncio
async def test_single_doc_prompt(
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
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
        message_metadata,
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
        tags=[
            'chat_bot_run_test', 
            f'uuid_${message_metadata['uuid']}', 
            f'conversation_id_${message_metadata['uuid']}'
        ],
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
    
@pytest.mark.asyncio
async def test_multi_doc_prompt(
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
    compare_documents: List[UploadFile],
):
    chat_bot_config.message_history.session_id = conversation_doc['_id']
    vector_store = os.environ['VECTOR_STORE']
    metadatas = await ingest(
        vector_store, 
        compare_documents, 
        chat_bot_config.embeddings,
        chat_bot_config.vectorstore,
        message_metadata,
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
        tags=[
            'chat_bot_run_test', 
            f'uuid_${message_metadata['uuid']}', 
            f'conversation_id_${message_metadata['uuid']}'
        ],
        metadata={ 'vector_metadata': metadatas },
        configurable={ 'retrieval_mode': 'mmr' }
    )

    ai_content = ''
    streaming_resp = []
    async for chunk in chain.astream(
        {'input': 'Compare the three documents'},
        config=config
    ):
        print(f'Custom event ${chunk.content}')
        ai_content += chunk.content
        streaming_resp.append(chunk)
    
    assert len(ai_content) > 0

@pytest.mark.asyncio
async def test_pretrained_corpus_prompt(
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
):
    chat_bot_config.message_history.session_id = conversation_doc['_id']
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            ('human', '{input}')
        ]
    )
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot

    config = RunnableConfig(
        tags=[
            'chat_bot_run_test', 
            f'uuid_${message_metadata['uuid']}', 
            f'conversation_id_${message_metadata['uuid']}'
        ],
        metadata={ 'vector_metadata': [message_metadata] },
        configurable={ 'retrieval_mode': 'mmr' }
    )

    ai_content = ''
    streaming_resp = []
    async for chunk in chain.astream(
        {'input': 'Tell me about the movie Memento.'},
        config=config
    ):
        print(f'Custom event ${chunk.content}')
        ai_content += chunk.content
        streaming_resp.append(chunk)
    
    assert len(ai_content) > 0    

@pytest.mark.asyncio
async def test_multimodal_image(
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
):
    chat_bot_config.message_history.session_id = conversation_doc['_id']

    image_path = Path(__file__).parent / 'assets' / 'baby.jpg'
    with image_path.open('rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    image_url = f'data:image/jpeg;base64,{base64_image}'

    chat_prompt = ChatPromptTemplate.from_messages([
        ('system', "You're a helpful assistant who can create text from images"),
        ('human', 
            [
                {'image_url': {'url': "{image_url}"}},
                '{input}'
            ]
        )
    ])
    
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot

    config = RunnableConfig(
        tags=[
            'chat_bot_run_test', 
            f'uuid_${message_metadata['uuid']}', 
            f'conversation_id_${message_metadata['uuid']}'
        ],
        metadata={ 'vector_metadata': [message_metadata] },
        configurable={ 'retrieval_mode': 'mmr' }
    )


    ai_content = ''
    streaming_resp = []
    async for chunk in chain.astream(
        {
            'input': 'Describe the image.',
            'image_url': image_url,
        },
        config=config
    ):
        print(f'Custom event ${chunk.content}')
        ai_content += chunk.content
        streaming_resp.append(chunk)

    assert len(ai_content) > 0

# async def test_message_history

# async def test_vector_history

# async def test_unsafe_content

# async def test_usage_tokens_with_callback

# actually system openai standard is human, ai, human, ai
# when i create multiple candidate completions, it will have
# to use the same human prompt twice!

# the only way to do this is to preserve the template,
# but change out the context based on new retriever strategy 
# such as similarity instead of mmr and then pass the newly
# generated template to new node in langgraph and then invoke the
# model again and store the aimessage in the return of the graph
# so it is also streamed, and you must make sure history preserved
# and self.chat_model.bind(new temperature)
# @pytest.mark.asyncio
# async def test_multiple_candidate_completions(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     compare_documents: List[UploadFile],    
# ):
#     ...


