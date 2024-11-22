from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from ..models.mongo_schema import ObjectId
from ..models.conversation import (
    Conversation, 
    ConversationSchema, 
)
from ..models.message import Message
from ..repositories.base_mongo_repository import base_mongo_factory as factory

MessageRepo = factory(Message)

_JOIN = {
            '$lookup': {
                'from': 'messages',
                'localField': '_id',
                'foreignField': 'conversation_id',
                'as': 'messages',
            }
        }

_PROJECT = {
                '$project': {
                    '_id': 1,
                    'createdAt': 1,
                    'updatedAt': 1,
                    'title': 1,
                    'prompt_used': 1,
                    'model_name': 1,
                    'filenames': 1,
                    'messages': {
                        '$map': {
                            'input': '$messages',
                            'as': 'msg',
                            'in': {
                                '_id': '$$msg._id',
                                'type': '$$msg.type',
                                'content': '$$msg.content',
                                'conversation_id': '$$msg.conversation_id',
                                'createdAt': '$$msg.createdAt',
                                'updatedAt': '$$msg.updatedAt'
                            }
                        }
                    }
                }
            }

class ConversationMongoRepository(factory(Conversation)):
    @classmethod
    async def all(
        cls, 
        *, 
        options: Optional[dict] = {}, 
        offset: int = 0, 
        limit: int = 20, 
        sort_field: str = 'updatedAt',
        sort_direction: str = 'desc'
    ) -> List[Dict[str, Any]]:
        """Fetch all documents in database filtered by user, limit, and offset"""
        stages = []
        stages.append({ '$match': options })
        stages.append(_JOIN)
        stages.append(_PROJECT)
        stages.append({ '$sort': { sort_field: 1 if sort_direction == 'asc' else -1 }})
        stages.append({ '$skip': offset })
        stages.append({ '$limit': limit })
        return await cls.get_collection().aggregate(stages).to_list(length=None)
    
    @classmethod
    async def create(cls, *, conversation_schema: ConversationSchema = BaseModel) -> str:
        """Create conversation"""
        created_conversation = await super().create(schema=conversation_schema)
        conversation_id = created_conversation['_id']
        return conversation_id
    
    @classmethod
    async def find_by_aggregate(cls, id: str = None, *, options: dict = {}) -> Dict[str, Any]:
        stages = []
        stages.append({ '$match': { '_id': ObjectId(id) }})
        stages.append(_JOIN)
        stages.append(_PROJECT)
        result = await cls.get_collection().aggregate(stages).to_list(length=None)
        return result and result[0]
    
    @classmethod
    async def delete_many(cls, *, options: dict) -> int:
        conversations = await cls.find(options=options)
        conversation_ids = [conversation['_id'] for conversation in conversations]
        deleted_count = await super().delete_many(options=options)
        if deleted_count > 0:
            await MessageRepo.delete_many(options={'conversation_id': {'$in': conversation_ids}})
        return deleted_count