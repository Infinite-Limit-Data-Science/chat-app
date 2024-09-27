import os
import json
import datetime
from pymongo.results import UpdateResult
from models.user import UserSchema, User
from models.model_config import ModelConfigSchema
from models.setting import Setting, SettingSchema
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

async def grandfathered_user(user_attributes: dict) -> UserSchema:
    options = {
        CURRENT_UUID_NAME: user_attributes[CURRENT_UUID_NAME] 
    }
    assigns = { 
        'roles': [],
        'createdAt': timestamp,
        'updatedAt': timestamp
    }
    timestamp = datetime.now()
    result: UpdateResult = await UserRepo.update_one(
            options=options, 
            assigns=assigns)
    return UserSchema(**{ '_id': result.upserted_id, **options, **assigns })