import logging
from typing import List, Dict, Any, Optional, Coroutine
from pymongo import ReturnDocument
from pymongo.results import UpdateResult
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel
from clients.mongo_strategy import mongo_instance as instance
from models.mongo_schema import ObjectId
from models.abstract_model import AbstractModel
from models.mongo_schema import ChatSchema
from hf_chat_ui_decorators.decorators import chat_ui 

def base_mongo_factory(model: AbstractModel):
    """Abstract the data storage and retrieval logic from the business logic of the application using first-class object BaseMongoRepository"""
    class BaseMongoRepository:
        @staticmethod
        def get_collection() -> AsyncIOMotorCollection:
            """Get the collection associated with Pydantic model"""
            return instance.get_database().get_collection(model.get_model_name())
        
        @classmethod
        async def all(
            cls, 
            *, 
            options: Optional[dict] = {}, 
            offset: int = 0, 
            limit: int = 20
        ) -> List[Dict[str, Any]]:
            """Fetch all documents in database filtered by user, limit, and offset"""
            return await cls.get_collection().find(options).skip(offset).limit(limit).to_list(limit)

        @classmethod
        async def create(
            cls, 
            *, 
            schema: ChatSchema = BaseModel, 
            options: Optional[dict] = {}
        ) -> Dict[str, Any]:
            """Create document"""
            insert_data = {**schema.model_dump(by_alias=True), **options}
            new_document = await cls.get_collection().insert_one(insert_data)
            return await cls.find_one(new_document.inserted_id)
        
        @classmethod
        async def find(cls, id: str = None, *, options: dict = {}) -> List[Dict[str, Any]]:
            """"Find documents by filter"""
            query = {"_id": ObjectId(id)} if id else {} 
            return await cls.get_collection().find({**query, **options}).to_list()
        
        @classmethod
        @chat_ui(model)  
        async def find_one(cls, id: str = None, *, options: dict = {}) -> Dict[str, Any]:
            """"Find a document by filter"""
            query = {"_id": ObjectId(id)} if id else {} 
            return await cls.get_collection().find_one({**query, **options})
        
        @classmethod
        async def count(cls, *, options: dict = {}) -> int:
            """"Count number of documents in filter"""
            return await cls.get_collection().count_documents(options)

        @classmethod
        async def update_one_and_return(cls, id: str, *, schema: ChatSchema = ChatSchema, options: dict = {}):
            """"Update a document"""
            # keep only fields with values
            document = {
                k: v for k, v in schema.model_dump(by_alias=True).items() if v is not None
            }
            if len(document) >= 1:
                update_result = await cls.get_collection().find_one_and_update(
                    {"_id": ObjectId(id), **options},
                    {"$set": document},
                    return_document=ReturnDocument.AFTER,
                )
            return update_result
        
        @classmethod
        async def update_one(
            cls, 
            id: str, 
            *, 
            options: Dict[str, Any] = {}, 
            _set: Dict[str, Any] = {}, 
            push: Dict[str, Any] = {}, 
            array_filters: Optional[list] = None
        ) -> Coroutine[Any, Any, UpdateResult]:
            """Update single document"""
            query = {"_id": ObjectId(id)} if id else {}
            operation = {}

            if _set:
                operation['$set'] = _set
            if push:
                operation['$push'] = push

            update_options = {**options}
            if array_filters:
                update_options['array_filters'] = array_filters

            return await cls.get_collection().update_one(query, operation, **update_options)
        
        @classmethod
        async def remove_from_field(cls, id: str = None, *, options: dict = {}) -> Dict[str, Any]:
            """Remove element from mongo field array"""
            query = {"_id": ObjectId(id)} if id else {}
            return await cls.get_collection().find_one_and_update(
                query, 
                { "$pull": options }, 
                return_document=ReturnDocument.AFTER, )
            
        @classmethod
        async def delete(cls, id: str, *, options: Optional[dict] = {}) -> int:
            """"Delete a document"""
            delete_result = await cls.get_collection().delete_one({"_id": ObjectId(id), **options})
            return delete_result.deleted_count

        @classmethod
        async def delete_many(cls, *, options: dict) -> int:
            """"Delete all documents by filter"""
            delete_result = await cls.get_collection().delete_many(options)
            return delete_result.deleted_count

    BaseMongoRepository.model = model
    return BaseMongoRepository