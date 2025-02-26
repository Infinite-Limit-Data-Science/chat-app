import os
import json
from typing import Dict, Optional, Any
from fastapi import Request, Depends, HTTPException
from ..logger import logger
from ..models.system_model_config import SystemModelConfigSchema
from ..models.setting import Setting, SettingSchema
from ..models.system_model_config import SystemModelConfigSchema
from ..models.user_model_config import UserModelConfigSchema
from ..models.system_embedding_config import SystemEmbeddingConfigSchema
from ..models.model_observer import ModelSubject, ModelObserver
from ..models.classification import get_strategy_for_classification
from ..repositories.base_mongo_repository import base_mongo_factory as factory
from ..clients.mongo_strategy import mongo_instance as database_instance
from ..huggingblue_chat_bot.chat_bot_config import (
    ChatBotConfig,
    LLMConfig,
    EmbeddingsConfig,
    RedisVectorStoreConfig,
    MongoMessageHistoryConfig,
)

DEFAULT_PREPROMPT='You are an assistant for question-answering tasks. Answer the questions to the best of your ability.'

SettingRepo = factory(Setting)

async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting = await SettingRepo.find_one(options={request.state.uuid_name: request.state.uuid})
    setting_schema = SettingSchema(**setting)
    return setting_schema

def load_system_model_config() -> Dict[str, SystemModelConfigSchema]:
    """Load system model configs from env"""
    model_dicts = json.loads(os.environ['MODELS'])
    system_configs = {config['name']: SystemModelConfigSchema(**config) for config in model_dicts}
    return system_configs

async def get_active_model_config(
    setting_id: str, 
    configs: Dict[str, SystemModelConfigSchema], 
    active_model_name: Optional[str]
) -> SystemModelConfigSchema:
    """Get active model config or default to the first model config if none active"""
    if active_model_name:
        model_config = configs[active_model_name]
        active_model = active_model_name
    else:
        model_config = next(iter(configs.values()))
        active_model = model_config.name
    await SettingRepo.update_one(setting_id, _set={'activeModel': active_model})
    logger.info(f'Active model config {active_model}')
    return model_config

async def refresh_model_configs(
    setting_schema: SettingSchema = Depends(get_user_settings),
    system_model_configs: Dict[str, SystemModelConfigSchema] = Depends(load_system_model_config)
) -> SystemModelConfigSchema:
    """Refresh model configs for the current user."""
    classification_subjects = {}
    user_model_configs = {model_config.name: model_config for model_config in setting_schema.user_model_configs}
    active_model_name = None

    removed_models = [name for name in user_model_configs if name not in system_model_configs]

    if removed_models:
        await SettingRepo.remove_from_field(
            setting_schema.id,
            options={'user_model_configs': {'name': {'$in': removed_models}}}
        )
        for name in removed_models:
            del user_model_configs[name]

    for system_model_name, system_config in system_model_configs.items():
        classification = system_config.classification
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_text_generation(): # and not strategy.is_image_to_text()
            continue
        
        if classification not in classification_subjects:
            classification_subjects[classification] = ModelSubject(classification)

        observer = ModelObserver(name=system_model_name)
        classification_subjects[classification].add_observer(observer)

        if system_model_name in user_model_configs:
            user_config = user_model_configs[system_model_name]
            system_config.parameters = user_config.parameters 
        else:
            new_user_model_config = UserModelConfigSchema(
                name=system_model_name,
                description=system_config.description,
                preprompt=system_config.preprompt,
                parameters=system_config.parameters,
                classification=system_config.classification,
                active=system_config.active
            )

            await SettingRepo.update_one(
                setting_schema.id,
                push={
                    'user_model_configs': new_user_model_config.model_dump(by_alias=True),
                }
            )

            user_model_configs[system_model_name] = new_user_model_config

    for classification, subject in classification_subjects.items():
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_text_generation():
            continue
        
        for user_model_name, user_model_config in user_model_configs.items():
            if user_model_config.classification == classification:
                if user_model_config.active:
                    subject.set_active(user_model_name)
                    active_model_name = user_model_name
                    break
        
        if not active_model_name:
            for system_model_name in subject.observers:
                strategy.set_active(system_model_configs, system_model_name)
                if system_model_configs[system_model_name].active:
                    subject.set_active(system_model_name)
                    active_model_name = system_model_name
                    break

    for classification, subject in classification_subjects.items():
        for model_name, observer in subject.observers.items():
            is_active = observer.active
            await SettingRepo.update_one(
                setting_schema.id,
                _set={'user_model_configs.$[elem].active': is_active},
                array_filters=[{'elem.name': model_name}],
            )

    return await get_active_model_config(setting_schema.id, system_model_configs, active_model_name)

async def llm_model_config(
    request: Request,
    system_model_config: SystemModelConfigSchema = Depends(refresh_model_configs)
) -> Dict[str, any]:
    return {
        'name': system_model_config.name,
        'endpoint': system_model_config.endpoints[0],
        'token': request.state.authorization,
        'parameters': dict(system_model_config.parameters),
    }

async def guardrails_model_config(
    request: Request, 
    system_model_configs: Dict[str, SystemModelConfigSchema] = Depends(load_system_model_config)
) -> Optional[Dict[str, any]]:
    """Return optional LLM (if no guardrails LLM found, then guardrails is disabled)"""
    for _, system_model_config in system_model_configs.items():
        classification = system_model_config.classification
        strategy = get_strategy_for_classification(classification)
        
        if not strategy.is_content_safety():
            continue
        
        if system_model_config.active:
            return await llm_model_config(request, system_model_config)
    
    return []

async def embeddings_model_config(request: Request) -> Dict[str, any]:
    model_dict = json.loads(os.environ['EMBEDDING_MODELS'])
    model_configs = [SystemEmbeddingConfigSchema(**config) for config in model_dict]
    active_model_config = next((config for config in model_configs if config.active), None)
    logger.info(
        f'using `Bearer {request.state.authorization}` for '
        f'embedding model config {active_model_config.name}'
    )
    if not active_model_config:
        HTTPException(status_code=404, detail='Expected Embedding Model, got None')
    return {
        'name': active_model_config.name,
        'endpoint': active_model_config.endpoints[0],
        'token': request.state.authorization,
        'parameters': {}
    }

async def get_prompt_template(
    setting_schema: SettingSchema = Depends(get_user_settings),
    system_config_schema: SystemModelConfigSchema = Depends(refresh_model_configs)
) -> str:
    """Derive prompt from either custom user prompts or system prompts if any"""
    if(
        active_user_config_id := next((config.id for config in setting_schema.user_model_configs if config.active), None)
    ) is not None:
        if(
            prompt := next((prompt for prompt in setting_schema.prompts for id in prompt.user_model_configs if id == active_user_config_id), None)
        ) is not None:
            logger.info(f'using prompt template {prompt.content}')
            return prompt.content
    
    logger.info(f'using prompt template {system_config_schema.preprompt or 'None'}')
    return system_config_schema.preprompt

def vector_store_config(request: Request) -> Dict[str, Any]:
    return {
        'url': os.environ['REDIS_URL'],
        'uuid': request.state.uuid,
        'session_id_key': database_instance.message_history_key
    }

def message_history_config() -> Dict[str, Any]:
    return {
        'url': database_instance.connection_string,
        'name': database_instance.name,
        'collection_name': database_instance.message_history_collection,
        'session_id_key': database_instance.message_history_key,
    }    

async def active_chat_bot_config(
    request: Request,
    llm_model_config: Dict[str, Any] = Depends(llm_model_config),
    embeddings_config: Dict[str, Any] = Depends(embeddings_model_config),
    guardrails_config: Optional[Dict[str, Any]] = Depends(guardrails_model_config),
    vector_store_config: Dict[str, Any] = Depends(vector_store_config),
    message_history_config: Dict[str, Any] = Depends(message_history_config),
) -> ChatBotConfig:
    llm = LLMConfig(**llm_model_config)
    embeddings = EmbeddingsConfig(**embeddings_config)
    guardrails = LLMConfig(**guardrails_config) if guardrails_config else None
    vectorstore = RedisVectorStoreConfig(**vector_store_config)
    message_history = MongoMessageHistoryConfig(**message_history_config)

    return ChatBotConfig(
        llm=llm,
        embeddings=embeddings,
        guardrails=guardrails,
        vectorstore=vectorstore,
        message_history=message_history
    )