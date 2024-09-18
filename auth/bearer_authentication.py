import os
import json
from fastapi import Request, Depends, logger
from fastapi.security import HTTPBearer
from jwt import decode, DecodeError
from models.user import UserSchema
from repositories.base_mongo_repository import base_mongo_factory as factory
from repositories.user_mongo_repository import UserMongoRepository as UserRepo
from models.jwt_token import JWTToken as Token
from models.setting import Setting, SettingSchema
from models.model_config import ModelConfigSchema

# for backward compatibility with chat-ui
CURRENT_UUID_NAME = 'sessionId'

SettingRepo = factory(Setting)

security = HTTPBearer()

def load_token_schema(token) -> Token:
    keys_to_extract = ['app', 'sub', 'roles', 'aud', 'exp', 'iat']
    required_attributes = {key: token[key] for key in keys_to_extract}
    token = Token(**required_attributes)
    return token

def validate_jwt(authorization: str = Depends(security)) -> Token:
    credentials = authorization.credentials
    if not credentials:
        return {'error': 'Missing Token'}, 401
    try:
        decoded_jwt = decode(credentials, options={'verify_signature': False})
        token = load_token_schema(decoded_jwt)
        if token.is_expired():
            return {'error': 'Token has expired'}, 401
    except DecodeError:
        return {'error': 'Invalid Token, decoding failed'}, 401
    return token

async def get_model_config(uuid: str) -> dict:
    model_dict = json.loads(os.environ['MODELS'])
    model_configs = [ModelConfigSchema(**config) for config in model_dict]
    synced_model_dicts = await SettingRepo.sync(options={CURRENT_UUID_NAME: uuid}, source=model_configs, attribute='model_configs', identifier='name')
    active_model_dict = next((model_dict for model_dict in synced_model_dicts if model_dict['active']), None)
    await SettingRepo.update_one(options={CURRENT_UUID_NAME: uuid}, assigns={'activeModel': active_model_dict['name']})
    return active_model_dict

async def get_current_user(request: Request, authorization: str = Depends(security), token: Token = Depends(validate_jwt)) -> UserSchema:
    if (
        user_attributes := await UserRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
    ) is None:
        user_attributes = await UserRepo.create(schema=UserSchema(uuid=token.sub, roles=token.roles))
        if (
            _ := await SettingRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
        ) is None:
            await SettingRepo.create(schema=SettingSchema(uuid=token.sub))
    user = UserSchema(**UserRepo.grandfather(user_attributes))
    await get_model_config(user.uuid)
    request.state.uuid = user.uuid
    request.state.uuid_name = CURRENT_UUID_NAME
    request.state.authorization = authorization.credentials
    return user