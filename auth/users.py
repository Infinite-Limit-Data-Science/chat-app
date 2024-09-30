import logging
from fastapi import Request
from models.user import User, UserSchema
from models.setting import Setting, SettingSchema
from models.jwt_token import JWTToken as Token
from repositories.base_mongo_repository import base_mongo_factory as factory
from routes.configs import refresh_model_configs, load_system_model_config

# for compatibility with HuggingFace chat-ui
CURRENT_UUID_NAME = 'sessionId'

UserRepo = factory(User)

SettingRepo = factory(Setting)

def chat_ui_data(attributes: dict) -> bool:
    special_key = '__chat_ui__'
    if attributes.get(special_key):
        del attributes[special_key]
        return True
    else:
        return False

async def load_user_settings(request: Request) -> None:
    if (
        setting_attributes := await SettingRepo.find_one(options={CURRENT_UUID_NAME: request.state.uuid}) 
    ) is not None:
        if chat_ui_data(setting_attributes):
            await SettingRepo.update_one(
                options={'sessionId': setting_attributes['sessionId'] }, 
                _set=SettingSchema(setting_attributes).model_dump(by_alias=True, exclude='sessionId'))
    if not setting_attributes:
        setting_attributes = await SettingRepo.create(schema=SettingSchema(uuid=request.state.uuid))
    system_model_configs = load_system_model_config()
    await refresh_model_configs(SettingSchema(**setting_attributes), system_model_configs)
    
async def fetch_user(user_attributes: dict) -> UserSchema:
    if chat_ui_data(user_attributes):
        user = UserSchema(**user_attributes)
        user_attributes = user.model_dump(by_alias=True, exclude={'uuid','id'})
        await UserRepo.update_one(
            user.id, 
            _set=user_attributes)
    else:
        user = UserSchema(**user_attributes)
    return user