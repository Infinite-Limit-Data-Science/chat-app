import base64
from typing import Tuple
from fastapi import Request, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt import decode, DecodeError
from ..logger import logger
from ..models.jwt_token import JWTToken as Token
from ..middleware import fetch_public_key
from .users import (
    CURRENT_UUID_NAME,
    fetch_user,
    load_user_settings,
    UserRepo,
    UserSchema,
)

security = HTTPBearer()

def load_token_schema(token: str, host: str) -> Token:
    keys_to_extract = ['app', 'sub', 'roles', 'aud', 'exp', 'iat', 'mail', 'displayname']
    required_attributes = {key: token[key] for key in keys_to_extract}
    token = Token(**{**required_attributes, 'host': host})
    return token

def validate_jwt(
    request: Request, 
    authorization: HTTPAuthorizationCredentials = Depends(security)
) -> Tuple[Token, str]:
    credentials = authorization.credentials
    if not credentials:
        raise HTTPException(status_code=401, detail='Missing Token')
    try:
        decoded_jwt = decode(
            credentials, 
            fetch_public_key(), 
            algorithms=["RS256"], 
            options={'verify_signature': True, "verify_aud": False}
        )
        token = load_token_schema(decoded_jwt, request.url.hostname)
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
    user_attributes = await UserRepo.find_one(options={CURRENT_UUID_NAME: token.sub})

    if user_attributes is None:
        user_attributes = await UserRepo.create(
            schema=UserSchema(
                uuid=token.sub, 
                roles=token.roles, 
                mail=token.mail, 
                displayname=token.displayname
            )
        )
    else:
        fields_to_update = {}
    
        if not user_attributes.get('mail'):
            fields_to_update["mail"] = token.mail
    
        if not user_attributes.get('displayname'):
            fields_to_update['displayname'] = token.displayname
    
        if fields_to_update:
            await UserRepo.update_one(
                user_attributes.get('_id'),
                update=fields_to_update
            )

    user = await fetch_user(user_attributes)
    logger.info(f'User found {user_attributes}')
    request.state.session_id = get_session_id(session)
    request.state.uuid = user.uuid
    request.state.uuid_name = CURRENT_UUID_NAME
    request.state.authorization = authorization.credentials
    await load_user_settings(request)
    
    return user