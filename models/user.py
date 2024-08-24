from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from chat_client import ChatClient as client
from models.bsonid import PyObjectId
from models.ldap_token import LdapToken as Token

class UserModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)
    uuid: str = Field(alias="sessionId", description='Unique identifer working across LDAP, ForgeRock, Microsoft Azure Entra ID, and AWS IAM Identity Center', frozen=True)
    roles: List[str] = Field(description='Represents LDAP Roles, Entra ID Roles, or IAM Roles', frozen=True)
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class UserFacade:
    @staticmethod
    def get_collection() -> AsyncIOMotorCollection:
        """Get the collection associated with Pydantic model"""
        return client.instance().db().get_collection('users')

    @classmethod
    async def find_or_create_by_uuid(cls, token: Token) -> UserModel:
        """Find or create a User by uuid attribute of LDAP User Entry and return it"""
        collection = cls.get_collection()
        user_attributes = await collection.find_one({'sessionId': token.sub})
        if user_attributes is None:
            await collection.insert_one(UserModel(uuid=token.sub, roles=token.roles).model_dump(by_alias=True) )
            user_attributes = await collection.find_one({ 'sessionId': token.sub })
        return UserModel(**cls.grandfather(user_attributes))
    
    def grandfather(attributes):
        """Ensure historic data complies with new validations by adding missing attributes."""
        required_attributes = ['roles', 'createdAt', 'updatedAt']
        for attr in required_attributes:
            attributes.setdefault(attr, [] if attr == 'roles' else datetime.now())
        return attributes