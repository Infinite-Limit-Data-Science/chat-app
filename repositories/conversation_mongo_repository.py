from typing import Dict, Any, Optional, List
from models.conversation import Conversation, ConversationCollectionSchema
from repositories.base_mongo_repository import base_mongo_factory as factory

class ConversationMongoRepository(factory(Conversation)):
    @classmethod
    async def all(cls, *, options: Optional[dict] = {}, offset: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
        conversations = ConversationCollectionSchema(super().all(options=options, offset=offset, limit=limit))
        # TODO: start here
        db.conversations.aggregate([
        {
            $match: {
                _id: ObjectId("...") // replace with the conversation ID
            }
        },
        {
            $lookup: {
                from: "messages",
                localField: "messageIds",
                foreignField: "_id",
                as: "messages"
        }
    }
])

