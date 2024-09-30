from datetime import datetime
from typing import TypeAlias, Optional, Annotated, Any
from pydantic import BaseModel, Field, BeforeValidator
from bson import ObjectId

PyObjectId = Annotated[str, BeforeValidator(str)]

# class PyObjectId(ObjectId):
#     @classmethod
#     def __get_pydantic_json_schema__(cls, schema):
#         schema.update(type="string")
#         return schema

#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     @classmethod
#     def validate(cls, v, field: Optional[Any] = None, config: Optional[Any] = None):
#         if isinstance(v, ObjectId):
#             return v
#         if isinstance(v, str) and ObjectId.is_valid(v):
#             return ObjectId(v)
#         raise ValueError("Invalid ObjectId")

ChatSchema: TypeAlias = BaseModel

class PrimaryKeyMixinSchema(ChatSchema):
    id: Optional[PyObjectId] = Field(alias="_id", description='bson object id', default_factory=ObjectId)

class TimestampMixinSchema(ChatSchema):
    createdAt: datetime = Field(description='Created At timestamp', default_factory=datetime.now)
    updatedAt: datetime = Field(description='Updated At timestamp', default_factory=datetime.now)