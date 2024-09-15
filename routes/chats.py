from typing import List
from fastapi import Request, Depends, logger
from clients.mongo_strategy import mongo_instance
from orchestrators.chat.chat_bot import ChatBot
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.factories import FACTORIES
from orchestrators.chat.messages.message_history import (
    MongoMessageHistory, 
    MongoMessageHistorySchema, 
    AIMessage,
)
from orchestrators.chat.llm_models.model_proxy import ModelProxy
from repositories.base_mongo_repository import base_mongo_factory as factory
from models.llm_schema import PromptDict
from models.model_config import (
    ModelConfigSchema,
    ModelConfig,
)
from models.setting import (
    Setting,
    SettingSchema
)
from models.message import MessageSchema

_DEFAULT_PREPROMPT='Please respond to the following question as if you were a knowledgeable expert in the field.'

SettingRepo = factory(Setting)

async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting = await SettingRepo.find_one(options={request.state.uuid_name: request.state.uuid})
    setting_schema = SettingSchema(**setting)
    return setting_schema

async def get_active_model_config(request: Request, setting_schema: SettingSchema = Depends(get_user_settings)) -> ModelConfigSchema:
    """Get the active model config for current user"""
    active_model_config = next((config for config in setting_schema.model_configs if config.active), None)
    return active_model_config

async def get_current_models(
    request: Request,
    setting_schema: SettingSchema = Depends(get_user_settings),
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> List[LLM]:
    """Return the active model(s) of settings for current user"""
    models = [
        FACTORIES[endpoint['type']](**{
            'name': model_config_schema.name,
            'description': model_config_schema.description,
            'preprompt': model_config_schema.preprompt,
            'parameters': dict(model_config_schema.parameters),
            'endpoint': endpoint
        })
        for endpoint in model_config_schema.endpoints
    ]
    return models

async def get_prompt_template(
    request: Request, 
    setting_schema: SettingSchema = Depends(get_user_settings), 
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> str:
    """Derive system prompt from either custom prompts or default system prompt"""
    prompt = next((prompt for prompt in setting_schema.prompts for model_config_id in prompt.model_configs if model_config_id == model_config_schema.id), None)
    if prompt is not None:
        return prompt.content
    return model_config_schema.preprompt

async def get_message_history(session_id: str) -> MongoMessageHistory:
    """Return Message History delegator"""
    return MongoMessageHistory(MongoMessageHistorySchema(
        connection_string=mongo_instance.connection_string,
        database_name=mongo_instance.name,
        collection_name=mongo_instance.message_history_collection,
        session_id_key=mongo_instance.message_history_key,
        session_id=session_id
    ))

async def chat(system_prompt: str,
    models: List[LLM], 
    metadata: dict, 
    message_schema: MessageSchema) -> AIMessage:
    """Chat"""
    mongo_message_history = await get_message_history(metadata['conversation_id'])
    model_proxy = ModelProxy(models)
    chat_bot = ChatBot(system_prompt, model_proxy, mongo_message_history, metadata)
    if mongo_message_history.has_no_messages:
        preprompt = model_proxy.models[0].preprompt
        if not preprompt:
            preprompt = _DEFAULT_PREPROMPT
        await chat_bot.add_system_message(preprompt)
    history = message_schema.model_dump(by_alias=True, include={'History'})
    message = await chat_bot.add_human_message(history['History'])
    return await chat_bot.chat(session_id=metadata['conversation_id'], message=message)