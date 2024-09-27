
from fastapi import Request, Depends, HTTPException, logger
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt import decode, DecodeError
from models.jwt_token import JWTToken as Token
from auth.users import (
    CURRENT_UUID_NAME,
    grandfathered_user, 
    get_model_config,
    UserRepo,
    SettingRepo,
    UserSchema,
    SettingSchema,
)

security = HTTPBearer()

def load_token_schema(token) -> Token:
    keys_to_extract = ['app', 'sub', 'roles', 'aud', 'exp', 'iat']
    required_attributes = {key: token[key] for key in keys_to_extract}
    token = Token(**required_attributes)
    return token

def validate_jwt(authorization: HTTPAuthorizationCredentials = Depends(security)) -> Token:
    credentials = authorization.credentials
    if not credentials:
        raise HTTPException(status_code=401, detail='Missing Token')
    try:
        decoded_jwt = decode(credentials, options={'verify_signature': False})
        token = load_token_schema(decoded_jwt)
        if token.is_expired():
            raise HTTPException(status_code=401, detail='Token has expired')
    except DecodeError:
        raise HTTPException(status_code=401, detail='Invalid Token, decoding failed')
    return token

async def get_current_user(request: Request, authorization: str = Depends(security), token: Token = Depends(validate_jwt)) -> UserSchema:
    if (
        user_attributes := await UserRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
    ) is None:
        user_attributes = await UserRepo.create(schema=UserSchema(uuid=token.sub, roles=token.roles))
        if (
            _ := await SettingRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
        ) is None:
            await SettingRepo.create(schema=SettingSchema(uuid=token.sub))

    if user_attributes['roles']:
        user = UserSchema(**user_attributes)
    else:
        user = await grandfathered_user(user_attributes)  
    
    await get_model_config(user.uuid)
    request.state.uuid = user.uuid
    request.state.uuid_name = CURRENT_UUID_NAME
    request.state.authorization = authorization.credentials
    return user