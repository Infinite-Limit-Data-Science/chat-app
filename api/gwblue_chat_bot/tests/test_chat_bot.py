import os
import json
import pytest
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
from ..chat_bot import ChatBot

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
        'parameters': {},
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
        'server': 'tei',        
    })

    vectorstore_config = RedisVectorStoreConfig(**{
        'client': redis_client,
        'metadata_schema': os.environ['VECTOR_STORE_SCHEMA']
    })

    MongoMessageHistoryConfig(
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
    )

def test_single_doc_prompt(chat_bot_config: ChatBotConfig):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            ('human', '{input}')
        ]
    )
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot

    config = RunnableConfig(
        tags=['chat_bot_run_test', 'uuid_1', 'conversation_id_1'],
        metadata={ 'vector_metadata': {'uuid': '1', 'conversation_id': '1', 'source': 'test.pdf'} },
    )
    conversation = chain.invoke(
        input='Summarize the document', 
        config=config,
        metadata=[
            {'uuid': '1', 'conversation_id': '1', 'source': 'test.pdf'}
        ], 
        retrieval_mode='mmr',
    )
    # I NEED TO DO A RUNNABLE_PARALLEL AND PASS retrieval_mode mmr for one and retrieval_mode similarity_search_with_threshold for the other !!!!
    # THIS SHOULD ONLY BE DONE WITH DOCUMENT UPLOADS. IF NO DOCUMENT UPLOAD, THEN ONLY SEND BACK A SINGLE RESPONSE.
    assert conversation['messages'][-1].content.strip('\n') == 'safe'

    # for event in graph.stream(initial_state):
    #     for value in event.values():
    #         assert value['messages'].strip('\n') == 'safe'