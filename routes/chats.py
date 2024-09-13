from typing import List
from fastapi import Request, Depends, logger
from clients.mongo_strategy import mongo_instance as instance
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


SettingRepo = factory(Setting)
ModelConfigRepo = factory(ModelConfig)

async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting_schema = SettingSchema(await SettingRepo.find(options={request.state.uuid_name: request.state.uuid}))
    return setting_schema

async def get_active_model_config(request: Request, setting_schema: SettingSchema = Depends(get_user_settings)) -> ModelConfigSchema:
    """Get the active model config"""
    model_config_schema = ModelConfigSchema(await ModelConfigRepo.find(options={'name': setting_schema.activeModel}))
    return model_config_schema

async def get_current_models(
    request: Request,
    setting_schema: SettingSchema = Depends(get_user_settings),
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> List[LLM]:
    """Return the active model(s) of settings for current user"""
    models = [
        FACTORIES[endpoint.type](**{
            'name': model_config_schema.name,
            'description': model_config_schema.description,
            'default_prompt': model_config_schema.default_prompt,
            'parameters': model_config_schema.parameters,
            'endpoint': endpoint
        })
        for endpoint in model_config_schema.endpoints
    ]
    return models

async def get_system_prompt(
    request: Request, 
    setting_schema: SettingSchema = Depends(get_user_settings), 
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> PromptDict:
    """Derive system prompt from either custom prompts or default system prompt"""
    prompt = next((prompt for prompt in setting_schema.prompts for model_config_id in prompt.model_configs if model_config_id == model_config_schema.id), None)
    if prompt is not None:
        return { 'title': prompt.title, 'prompt': prompt.prompt}
    return model_config_schema.default_prompt

async def get_message_history(request: Request) -> MongoMessageHistory:
    """Return the Message History"""
    return MongoMessageHistory(MongoMessageHistorySchema(
        connection_string=instance.connection_string,
        session_id=instance.message_history_identifier,
        database_name=instance.name,
        collection_name=instance.message_history_collection
    ))

async def chat(system_prompt: PromptDict,
    models: LLM, 
    mongo_message_history: MongoMessageHistory, 
    metadata: dict, 
    message_schema: MessageSchema) -> AIMessage:
    chat_bot = ChatBot(system_prompt, ModelProxy(models), mongo_message_history, metadata)
    if mongo_message_history.has_no_messages:
        await chat_bot.add_system_message(dict(system_prompt))
    message = await chat_bot.add_human_message(dict(message_schema))
    return chat_bot.chat(session_id=metadata['uuid'], message=message)