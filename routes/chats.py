from typing import List, Callable, AsyncGenerator
from fastapi import Request, Depends, HTTPException, logger
from clients.mongo_strategy import mongo_instance as database_instance
from orchestrators.chat.chat_bot import ChatBot
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.factories import FACTORIES as LLM_FACTORIES
from orchestrators.chat.llm_models.model_proxy import ModelProxy as LLMProxy
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingProxy
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.embedding_models.factories import FACTORIES as EMBEDDING_FACTORIES
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.model_config import (
    ModelConfigSchema,
    ModelConfig,
)
from models.embedding_config import EmbeddingConfigSchema
from models.setting import (
    Setting,
    SettingSchema
)
from models.message import MessageSchema

_DEFAULT_VECTOR_STORE='redis'

_DEFAULT_PREPROMPT='You are a helpful assistant. Answer all the questions to the best of your ability.'

SettingRepo = factory(Setting)

async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting = await SettingRepo.find_one(options={request.state.uuid_name: request.state.uuid})
    setting_schema = SettingSchema(**setting)
    return setting_schema

async def get_active_model_config(setting_schema: SettingSchema = Depends(get_user_settings)) -> ModelConfigSchema:
    """Get the active model config for current user"""
    active_model_config = next((config for config in setting_schema.model_configs if config.active), None)
    return active_model_config

async def get_current_models(
    request: Request,
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> List[LLM]:
    """Return the active model(s) of settings for current user"""
    models = [
        LLM_FACTORIES[endpoint['type']](**{
            'name': model_config_schema.name,
            'description': model_config_schema.description,
            'preprompt': model_config_schema.preprompt,
            'parameters': dict(model_config_schema.parameters),
            'server_kwargs': { 'headers': {'Authorization': f'Bearer {request.state.authorization}' }},
            'endpoint': endpoint,
        })
        for endpoint in model_config_schema.endpoints
    ]
    return models    

async def get_current_embedding_models(request: Request) -> List[BaseEmbedding]:
    import os
    import json
    model_dict = json.loads(os.environ['EMBEDDING_MODELS'])
    model_configs = [EmbeddingConfigSchema(**config) for config in model_dict]
    active_model_config = next((config for config in model_configs if config.active), None)
    if not active_model_config:
        HTTPException(status_code=404, detail='Expected Embedding Model, got None')
    models = [
        EMBEDDING_FACTORIES[endpoint['type']](**{
            'name': active_model_config.name,
            'description': active_model_config.description,
            'task': active_model_config.task,
            'endpoint': endpoint,
            'dimensions': active_model_config.dimensions,
            'token': request.state.authorization,
        })
        for endpoint in active_model_config.endpoints
    ]
    return models

async def get_prompt_template(
    setting_schema: SettingSchema = Depends(get_user_settings), 
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> str:
    """Derive system prompt from either custom prompts or default system prompt"""
    prompt = next((prompt for prompt in setting_schema.prompts for model_config_id in prompt.model_configs if model_config_id == model_config_schema.id), None)
    if prompt is not None:
        return prompt.content
    return model_config_schema.preprompt or _DEFAULT_PREPROMPT

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