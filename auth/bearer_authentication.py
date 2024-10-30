import base64
from typing import Tuple
from fastapi import Request, Depends, HTTPException, logger
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt import decode, DecodeError
from models.jwt_token import JWTToken as Token
from auth.users import (
    CURRENT_UUID_NAME,
    fetch_user,
    load_user_settings,
    UserRepo,
    UserSchema,
)

security = HTTPBearer()

def load_token_schema(token) -> Token:
    keys_to_extract = ['app', 'sub', 'roles', 'aud', 'exp', 'iat']
    required_attributes = {key: token[key] for key in keys_to_extract}
    token = Token(**required_attributes)
    return token

def validate_jwt(authorization: HTTPAuthorizationCredentials = Depends(security)) -> Tuple[Token, str]:
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
    return token, credentials

def get_session_id(session: str) -> str:
    return base64.b64encode(session.encode('utf-8'))

async def get_current_user(
    request: Request, 
    authorization: HTTPAuthorizationCredentials = Depends(security), 
    token_credentials: Tuple[Token, str] = Depends(validate_jwt)
) -> UserSchema:
    token, session = token_credentials
    if (
        user_attributes := await UserRepo.find_one(options={CURRENT_UUID_NAME: token.sub}) 
    ) is None:
        user_attributes = await UserRepo.create(schema=UserSchema(uuid=token.sub, roles=token.roles))
    user = await fetch_user(user_attributes)
    request.state.session_id = get_session_id(session)
    request.state.uuid = user.uuid
    request.state.uuid_name = CURRENT_UUID_NAME
    request.state.authorization = authorization.credentials
    await load_user_settings(request)
    
    return user