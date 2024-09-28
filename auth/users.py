import os
import json
from models.user import User, UserSchema
from models.model_config import ModelConfigSchema
from models.setting import Setting, SettingSchema
from models.jwt_token import JWTToken as Token
from repositories.base_mongo_repository import base_mongo_factory as factory

# for backward compatibility with chat-ui
CURRENT_UUID_NAME = 'sessionId'

UserRepo = factory(User)

SettingRepo = factory(Setting)

async def get_model_config(uuid: str) -> dict:
    model_dict = json.loads(os.environ['MODELS'])
    model_configs = [ModelConfigSchema(**config) for config in model_dict]
    synced_model_dicts = await SettingRepo.sync(options={CURRENT_UUID_NAME: uuid}, source=model_configs, attribute='model_configs', identifier='name')
    active_model_dict = next((model_dict for model_dict in synced_model_dicts if model_dict['active']), None)
    await SettingRepo.update_one(options={CURRENT_UUID_NAME: uuid}, assigns={'activeModel': active_model_dict['name']})
    return active_model_dict

async def validate_settings(token: Token) -> None:
    setting_attributes = await SettingRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
    if setting_attributes['__incompatible__']:
        setting_attributes.pop('__incompatible__')
        setting = SettingSchema(setting_attributes)
        await SettingRepo.update_one(
            options={'sessionId': setting.uuid }, 
            assigns=setting.model_dump(by_alias=True, exclude={'sessionId'}))

async def fetch_user(user_attributes: dict, token: Token) -> UserSchema:
    if user_attributes['__incompatible__']:
        user_attributes.pop('__incompatible__')
        user = UserSchema(user_attributes)
        await UserRepo.update_one(
            options={'sessionId': user.uuid }, 
            assigns=user.model_dump(by_alias=True, exclude={'sessionId'}))
    else:
        user = UserSchema(user_attributes)
    await validate_settings(token)
    return user