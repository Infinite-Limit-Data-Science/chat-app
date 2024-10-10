import logging
from typing import List, Optional, Callable, AsyncGenerator
from orchestrators.chat.chat_bot import ChatBot, ChatBotBuilder
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from clients.mongo_strategy import mongo_instance as database_instance
from models.message import MessageSchema

_DEFAULT_VECTOR_STORE='redis'

async def chat(
    user_prompt_template: str,
    models: List[LLM],
    embedding_models: List[BaseEmbedding],
    data: dict,
    ingestors: Optional[List[DocumentIngestor]],
    message_schema: MessageSchema) -> Callable[[], AsyncGenerator[str, None]]:
    """Invoke chat bot"""
    chat_bot = ChatBot()
    builder = ChatBotBuilder(chat_bot)
    builder.build_vector_part(
        _DEFAULT_VECTOR_STORE,
        ingestors, 
        embedding_models, 
        [
            {
                'name': 'uuid',
                'type': 'tag',
                'value': data['uuid'],
            },
            {
                'name': 'conversation_id', 
                'type': 'tag',
                'value': str(data['conversation_id']),
            },
            {
                'name': 'source',
                'type': 'tag'
            },
    ])
    builder.build_llm_part(models)
    builder.build_prompt_part(user_prompt_template)
    builder.build_message_part({
        'connection_string': database_instance.connection_string,
        'database_name': database_instance.name,
        'collection_name': database_instance.message_history_collection,
        'session_id_key': database_instance.message_history_key,
    },
    {
        'session_id': data['conversation_id'],
    })

    return await chat_bot.chat(message_schema.content)