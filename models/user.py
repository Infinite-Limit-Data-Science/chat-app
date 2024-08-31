from typing import List
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
)

class User(AbstractModel):
    __modelname__ = 'users'

    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class UserSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: str = Field(alias="sessionId", description='Unique identifer working across LDAP, ForgeRock, Microsoft Azure Entra ID, and AWS IAM Identity Center', frozen=True)
    roles: List[str] = Field(description='Represents LDAP Roles, Entra ID Roles, or IAM Roles', frozen=True)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True