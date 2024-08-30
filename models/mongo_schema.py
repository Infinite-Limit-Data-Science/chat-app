from datetime import datetime
from typing import TypeAlias, Optional, Annotated
from pydantic import BaseModel, Field, BeforeValidator
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(str)]

ChatSchema: TypeAlias = BaseModel

class PrimaryKeyMixinSchema(ChatSchema):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)

class TimestampMixinSchema(ChatSchema):
    createdAt: datetime = Field(description='Created At timestamp', default_factory=datetime.now)
    updatedAt: datetime = Field(description='Updated At timestamp', default_factory=datetime.now)