import json
import datetime as dt
from typing import List
from pymongo import errors
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from ..logger import logger

_ROOT_COLLECTION='conversations'

class MyMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from MongoDB"""
        try:
            if self.history_size is None:
                cursor = self.collection.find({self.session_id_key: self.session_id})
            else:
                total_docs_for_session = self.collection.count_documents({self.session_id_key: self.session_id})
                skip_count = max(0, total_docs_for_session - self.history_size)
                cursor = self.collection.find(
                    {self.session_id_key: self.session_id}
                ).skip(skip_count)
        except errors.OperationFailure as error:
            logger.error(error)
            return []

        items = [json.loads(document[self.history_key]) for document in cursor] if cursor else []
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in MongoDB"""
        # try:
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
            self.db[_ROOT_COLLECTION].update_one(
                { '_id': self.session_id },
                { '$push': { 'message_ids': new_document.inserted_id } }
            )
        
        # except errors.WriteError as err:
        #     logger.error(err)

    def add_summary(self, summary: str) -> None:
        self.db[_ROOT_COLLECTION].update_one(
            { '_id': self.session_id },
            { '$set': { 'title': summary } }
        )