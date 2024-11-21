import os
from typing import List, Optional, Callable, AsyncGenerator
from ..langchain_chat import ChatBot, ChatBotBuilder, LLM
from ..langchain_doc import AbstractVectorRetriever, BaseEmbedding
from ..logger import logger
from ..clients.mongo_strategy import mongo_instance as database_instance
from ..models.message import MessageSchema

async def chat(
    user_prompt_template: str,
    models: List[LLM],
    guardrails: List[LLM],
    embedding_models: List[BaseEmbedding],
    data: dict,
    retrievers: Optional[List[AbstractVectorRetriever]],
    message_schema: MessageSchema
) -> Callable[[], AsyncGenerator[str, None]]:
    """Invoke chat bot"""
    logger.info(
        f'`chat-ai` invocation call to {list({llm.name for llm in models})} '
        f'with content `{message_schema.content}` and conversation '
        f'id {data['conversation_id']}'
    )

    if not (vector_store := os.getenv('VECTOR_STORE')):
        raise ValueError('Expected `VECTOR_STORE` to be defined')

    chat_bot = ChatBot()
    builder = ChatBotBuilder(chat_bot)
    builder.build_vector_part(
        vector_store,
        retrievers, 
        embedding_models,
        {
            **data,
            'conversation_id': str(data['conversation_id'])
        },
    )
    builder.build_llm_part(models)
    builder.build_guardrails_part(guardrails)
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