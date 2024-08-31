from fastapi import Request, Depends, logger
from fastapi.security import HTTPBearer
from jwt import decode, DecodeError
from models.user import UserSchema
from repositories.user_mongo_repository import UserMongoRepository as UserRepo
from models.ldap_token import LdapToken as Token

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

async def get_current_user(request: Request, token: Token = Depends(validate_jwt)) -> UserSchema:
    user = await UserRepo.find_or_create_by_uuid(token)
    logger.logging.warning(f'USER MODEL: {UserRepo.model} AND {UserRepo.get_collection()}')
    request.state.uuid = user.uuid
    return user