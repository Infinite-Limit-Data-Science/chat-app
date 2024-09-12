import os
import json
from typing import List
from fastapi import Request, Depends, logger
from fastapi.security import HTTPBearer
from jwt import decode, DecodeError
from models.user import UserSchema
from repositories.base_mongo_repository import base_mongo_factory as factory
from repositories.user_mongo_repository import UserMongoRepository as UserRepo
from models.ldap_token import LdapToken as Token
from models.setting import Setting, SettingSchema
from models.model_config import ModelConfigSchema, ModelConfigCollectionSchema

# for backward compatibility with chat-ui
CURRENT_UUID_NAME = 'sessionId'

SettingRepo = factory(Setting)

security = HTTPBearer()
    
def validate_jwt(authorization: str = Depends(security)) -> Token:
    credentials = authorization.credentials
    if not credentials:
        return {'error': 'Missing Token'}, 401
    try:
        token = decode(credentials + '.', options={'verify_signature': False})
        token = Token(**token)
        if token.sub is None:
            return {'error': 'Missing LDAP UUID attribute of User Entry'}, 401
        if token.is_expired():
            return {'error': 'Token has expired'}, 401
    except DecodeError:
        return {'error': 'Invalid Token, decoding failed'}, 401
    return token

async def get_model_config(uuid: str) -> ModelConfigCollectionSchema:
    model_dict = json.loads(os.environ['MODELS'])
    model_configs = [ModelConfigSchema(config) for config in model_dict]
    schema = ModelConfigCollectionSchema(await SettingRepo.sync(options={CURRENT_UUID_NAME: uuid}, source=model_configs, attribute='model_configs', identifier='name'))
    # TODO: need to add boolean attribute to model_config for activeModel
    activeModel = schema[0]['name']
    await SettingRepo.update_one({CURRENT_UUID_NAME: uuid}, {'activeModel': activeModel})
    return schema

async def get_current_user(request: Request, token: Token = Depends(validate_jwt)) -> UserSchema:
    if (
        user_attributes := await UserRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
    ) is None:
        user_attributes = await UserRepo.create(schema=UserSchema(uuid=token.sub, roles=token.roles))
        if (
            _ := await SettingRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
        ) is None:
            await SettingRepo.create(schema=SettingSchema(uuid=token.sub))
    await get_model_config(user.uuid)
    user = UserSchema(**UserRepo.grandfather(user_attributes))
    request.state.uuid = user.uuid
    request.state.uuid_name = CURRENT_UUID_NAME
    return user