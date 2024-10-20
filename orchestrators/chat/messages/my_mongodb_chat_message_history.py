import logging
import json
import datetime as dt
from pymongo import errors
from langchain_mongodb import MongoDBChatMessageHistory
from orchestrators.chat.messages.message_history import BaseMessage
from langchain_core.messages import BaseMessage, message_to_dict

logger = logging.getLogger(__name__)

_ROOT_COLLECTION='conversations'

class MyMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        try:
            current_time = dt.datetime.now(dt.timezone.utc)
            if( new_document := self.collection.insert_one(
                {
                    self.session_id_key: self.session_id,
                    self.history_key: json.dumps(message_to_dict(message)),
                    'createdAt': current_time,
                    'updatedAt': current_time,
                    'type': message.type,
                    'content': message.content,
                }
            )) is not None:
                logging.warning(f'session id type({type(self.session_id)})')
                self.db[_ROOT_COLLECTION].update_one(
                    { '_id': self.session_id },
                    { '$push': { 'message_ids': new_document.inserted_id } }
                )
        
        except errors.WriteError as err:
            logger.error(err)

    def add_summary(self, summary: str) -> None:
        self.db[_ROOT_COLLECTION].update_one(
            { '_id': self.session_id },
            { '$set': { 'title': summary } }
        )