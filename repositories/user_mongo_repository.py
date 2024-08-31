from datetime import datetime
from models.ldap_token import LdapToken as Token
from models.user import User, UserSchema
from repositories.base_mongo_repository import base_mongo_factory

class UserMongoRepository(base_mongo_factory(User)):
    @classmethod
    async def find_or_create_by_uuid(cls, token: Token) -> UserSchema:
        """Find or create a User by uuid attribute of LDAP User Entry and return it"""
        collection = cls.get_collection()
        user_attributes = await collection.find_one({'sessionId': token.sub})
        if not user_attributes:
            await collection.insert_one(UserSchema(uuid=token.sub, roles=token.roles).model_dump(by_alias=True) )
            user_attributes = await collection.find_one({ 'sessionId': token.sub })
        return UserSchema(**cls.grandfather(user_attributes))
    
    @staticmethod
    def grandfather(attributes):
        """Ensure historic data complies with new validations by adding missing attributes."""
        required_attributes = ['roles', 'createdAt', 'updatedAt']
        for attr in required_attributes:
            attributes.setdefault(attr, [] if attr == 'roles' else datetime.now())
        return attributes