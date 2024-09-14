from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from models.mongo_schema import ObjectId
from models.message import Message, MessageSchema
from models.conversation import (
    Conversation, 
    ConversationSchema, 
    CreateConversationSchema, 
    UpdateConversationSchema,
)
from repositories.base_mongo_repository import base_mongo_factory as factory

MessageRepo = factory(Message)

class ConversationMongoRepository(factory(Conversation)):
    @classmethod
    def pipeline(cls, options: Optional[dict] = {}) -> List[dict]:
        stages = []
        if options:
            stages.append({
                '$match': options
            })
        stages.append({
            '$lookup': {
                'from': 'messages',
                'localField': 'message_ids',
                'foreignField': '_id',
                'as': 'messages'
            }
        })
        return stages

    @classmethod
    async def all(cls, *, options: Optional[dict] = {}, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch all documents in database filtered by user, limit, and offset"""
        stages = cls.pipeline(options)
        stages.append({ '$skip': offset })
        stages.append({ '$limit': limit })
        return await cls.get_collection().aggregate(stages).to_list(length=None)
    
    @classmethod
    async def create(cls, *, schema: ConversationSchema = BaseModel,  options: Optional[dict] = {}) -> Dict[str, Any]:
        """Create conversation"""
        conversation_dict = {
            **options,
            'title': schema.title,
            'message_ids': []
        }
        create_schema = CreateConversationSchema(conversation_dict)
        created_conversation = await super().create(schema=create_schema)
        return created_conversation['_id']
        # for message in schema.messages:
        #     message_doc = MessageSchema({
        #         'conversation_id': conversation_id,
        #         'History': message.History
        #     })
        #     created_message = await MessageRepo.create(schema=message_doc)
        #     conversation_dict['message_ids'].append(created_message['_id'])
        # await super().update_one(options={"_id": conversation_id}, assigns=conversation_dict)
        # return conversation_dict
    
    @classmethod
    async def find_one(cls, id: str = None, *, _: dict = {}) -> Dict[str, Any]:
        stages = cls.pipeline({"_id": ObjectId(id)})
        return await cls.get_collection().aggregate(stages)