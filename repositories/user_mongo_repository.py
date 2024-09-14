from typing import Optional
from datetime import datetime
from models.ldap_token import LdapToken as Token
from models.user import User, UserSchema
from repositories.base_mongo_repository import base_mongo_factory

class UserMongoRepository(base_mongo_factory(User)):
    @staticmethod
    def grandfather(attributes):
        """Ensure historic data complies with new validations by adding missing attributes."""
        required_attributes = ['roles', 'createdAt', 'updatedAt']
        for attr in required_attributes:
            if attr not in attributes:
                attributes.setdefault(attr, [] if attr == 'roles' else datetime.now())
        return attributes