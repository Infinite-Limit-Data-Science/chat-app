from typing import List, Optional
from .abstract_model import AbstractModel
from .mongo_schema import (
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
)

class User(AbstractModel):
    __modelname__ = 'users'

    @classmethod
    def get_model_name(cls):
        return cls.__modelname__
    
    @classmethod
    def chat_ui_compatible(cls) -> List[str]:
        return ['roles', 'createdAt', 'updatedAt']

class UserSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: str = Field(alias="sessionId", description='Unique identifer working across LDAP, ForgeRock, Microsoft Azure Entra ID, and AWS IAM Identity Center', frozen=True)
    roles: Optional[List[str]] = Field(description='Represents LDAP Roles, Entra ID Roles, or IAM Roles', frozen=True, default_factory=list)
    mail: str = Field(description='email of user', frozen=True)
    displayname: str = Field(description='display name of the user', frozen=True)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True