from typing import List, Dict, Any
from pymongo import ReturnDocument
from motor.motor_asyncio import AsyncIOMotorCollection
from clients.mongo_strategy import mongo_instance as instance
from models.mongo_schema import ObjectId
from models.abstract_model import AbstractModel
from models.mongo_schema import ChatSchema

def base_mongo_factory(model: AbstractModel):
    """Abstract the data storage and retrieval logic from the business logic of the application using first-class object BaseMongoRepository"""
    class BaseMongoRepository:
        @staticmethod
        def get_collection() -> AsyncIOMotorCollection:
            """Get the collection associated with Pydantic model"""
            
            return instance.get_database().get_collection(model.get_model_name())
        
        @classmethod
        async def all(cls, uuid: str, offset: int, limit: int) -> List[Dict[str, Any]]:
            """Fetch all documents in database filtered by user, limit, and offset"""
            return await cls.get_collection().find({'sessionId': uuid}).skip(offset).limit(limit).to_list(limit)

        @classmethod
        async def create(cls, uuid: str, schema: ChatSchema) -> Dict[str, Any]:
            insert_data = {**schema.model_dump(by_alias=True), 'sessionId': uuid}
            new_document = await cls.get_collection().insert_one(insert_data)
            return await cls.get_collection().find_one({'_id': new_document.inserted_id})
        
        @classmethod
        async def find(cls, uuid: str, id: str) -> Dict[str, Any]:
            """"Find a conversation by id"""
            return await cls.get_collection().find_one({"_id": ObjectId(id), 'sessionId': uuid })
        
        @classmethod
        async def update(cls, uuid: str, id: str, schema: ChatSchema):
            """"Update a conversation"""
            # keep only fields with values
            document = {
                k: v for k, v in schema.model_dump(by_alias=True).items() if v is not None
            }
            if len(document) >= 1:
                update_result = await cls.get_collection().find_one_and_update(
                    {"_id": ObjectId(id), 'sessionId': uuid},
                    {"$set": document},
                    return_document=ReturnDocument.AFTER,
                )
            return update_result
        
        @classmethod
        async def delete(cls, uuid: str, id: str) -> bool:
            """"Delete a conversation"""
            delete_result = await cls.get_collection().delete_one({"_id": ObjectId(id), 'sessionId': uuid})
            if delete_result.deleted_count == 1:
                return True
            return False

    BaseMongoRepository.model = model
    return BaseMongoRepository