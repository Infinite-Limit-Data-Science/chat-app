from typing import Dict, Any, Optional
from pymongo import ReturnDocument
from models.message import Message, UpdateMessageSchema
from models.mongo_schema import ObjectId
from repositories.base_mongo_repository import base_mongo_factory as factory

class MessageMongoRepository(factory(Message)):
    @classmethod
    async def find(cls, uuid: str, conversation_id: str, id: str) -> Dict[str, Any]:
        """"Find a message by id"""
        conversation = await cls.get_collection().find_one(
            {"_id": ObjectId(conversation_id), 'sessionId': uuid},
            { "messages": { "$elemMatch": { "_id": ObjectId(id) } } }
        )
        if conversation:
            return conversation['messages'][0]
        return None

    @classmethod
    async def update(cls, uuid: str, conversation_id: str, id: str, message: UpdateMessageSchema) -> Optional[Dict[str,Any]]:
        """"Update a message"""
        message = {
            k: v for k, v in message.model_dump(by_alias=True).items() if v is not None
        }
        if len(message) >= 1:
            conversation = await cls.get_collection().find_one_and_update(
                {"_id": ObjectId(conversation_id), 'sessionId': uuid},
                {
                    "$set": {
                        f"messages.$[message].{k}": v
                        for k, v in message.items()
                    }
                },
                array_filters=[{"message._id": ObjectId(id)}],
                return_document=ReturnDocument.AFTER,
            )
            updated_message = next((msg for msg in conversation["messages"] if msg["_id"] == ObjectId(id)), None)
            return updated_message
        
    @classmethod
    async def delete(cls, uuid: str, conversation_id: str, id: str) -> bool:
        """"Delete a message"""
        delete_result = await cls.get_collection().update_one(
            {"_id": ObjectId(conversation_id), 'sessionId': uuid},
            {"$pull": {"messages": {"_id": ObjectId(id)}}}
        )
        if delete_result.modified_count == 1:
            return True
        return False