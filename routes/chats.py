from typing import List, Callable, AsyncGenerator
from clients.mongo_strategy import mongo_instance as database_instance
from orchestrators.chat.chat_bot import ChatBot
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.model_proxy import ModelProxy as LLMProxy
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingProxy
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from models.message import MessageSchema

_DEFAULT_VECTOR_STORE='redis'

async def chat(user_prompt_template: str,
    models: List[LLM],
    embedding_models: List[BaseEmbedding],
    data: dict, 
    message_schema: MessageSchema) -> Callable[[], AsyncGenerator[str, None]]:
    """Chat"""
    model_proxy = LLMProxy(models)
    embedding_model_proxy = EmbeddingProxy(embedding_models)
    chat_bot = ChatBot(
        _DEFAULT_VECTOR_STORE,
        user_prompt_template, 
        model_proxy, 
        embedding_model_proxy, 
        {
            'connection_string': database_instance.connection_string,
            'database_name': database_instance.name,
            'collection_name': database_instance.message_history_collection,
            'session_id_key': database_instance.message_history_key,
        }, 
        {
            'metadata': {
                'uuid': data['uuid'], 
                'conversation_id': data['conversation_id'],
                'schema': [
                    {
                        'name': 'uuid', 
                        'type': 'tag'
                    },
                    {
                        'name': 'conversation_id', 
                        'type': 'tag'
                    },
                ]
            },
            'configurable': {
                'session_id': data['conversation_id'],
            }
        })
    return await chat_bot.chat(message_schema.content)