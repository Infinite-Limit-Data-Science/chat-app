async def get_user_settings(request: Request) -> SettingSchema:
    """Retrieve settings for current user"""
    setting_schema = SettingSchema(await SettingRepo.find(options={request.state.uuid_name: request.state.uuid}))
    return setting_schema

async def get_active_model_config(request: Request, setting_schema: SettingSchema) -> ModelConfigSchema:
    """Get the active model config"""
    model_config_schema = ModelConfigSchema(await ModelConfigRepo.find(options={'name': setting_schema.activeModel}))
    return model_config_schema
    
async def get_system_prompt(
    request: Request, 
    setting_schema: SettingSchema = Depends(get_user_settings), 
    model_config_schema: ModelConfigSchema = Depends(get_active_model_config)) -> PromptDict:
    """Derive system prompt from either custom prompts or default system prompt"""
    prompt = next((prompt for prompt in setting_schema.prompts for model_config_id in prompt.model_configs if model_config_id == model_config_schema.id), None)
    if prompt is not None:
        return { 'title': prompt.title, 'prompt': prompt.prompt}
    return model_config_schema.default_prompt

async def get_current_models(
    request: Request, 
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

# won't need get_message_history since when creating conversation, there will be no message history. It's only relevant for creating messages which are part of an existing conversation