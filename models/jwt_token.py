import logging
import re
from datetime import datetime
from typing import List, Any, Optional
from pydantic import field_validator, model_validator
from models.mongo_schema import ChatSchema, Field

class JWTToken(ChatSchema):
    app: str = Field(description='JWT app attribute')
    sub: str = Field(description='LDAP UUID attribute of User Entry')
    aud: str = Field(description='JWT audience attribute')
    roles: List[str] = Field(description='LDAP group entries')
    exp: datetime = Field(description='JWT token expiration time')
    iat: datetime = Field(description='JWT token iat time')

    class Config:
        allow_mutation = False

    @field_validator('sub', mode='before')
    @classmethod
    def validate_sub(cls, value: str) -> str:
        return value.lower()
    
    @field_validator('roles', mode='before')
    @classmethod
    def validate_roles(cls, value: List[str]) -> List[str]:
        pattern = r"^CN=Data\s+Services?\s+Entitlements.*$"
        valid_roles =[role for role in value if re.match(pattern, role)]
        if len(valid_roles) < 1:
             # raise ValueError('Received Invalid Group Entries, expecting at least one valid LDAP Group Entry')
             logging.warning(f'No data entitlements for sub')
        return value
    
    @field_validator('exp', mode='before')
    @classmethod
    def validate_exp(cls, value: str) -> datetime:
        return datetime.fromtimestamp(value)
    
    @field_validator('iat', mode='before')
    @classmethod
    def validate_iat(cls, value: str) -> datetime:
        return datetime.fromtimestamp(value)

    # @model_validator(mode='before')
    # @classmethod
    # def validate_token(cls, v: dict[str, Any]) -> dict[str, Any]:
    #     if 'sub' not in v and 'roles' not in v and 'exp' not in v and 'iat' not in v:
    #         raise ValueError('Missing attributes are required')
    #     return v
    
    def is_expired(self) -> bool:
        return datetime.now() > self.exp