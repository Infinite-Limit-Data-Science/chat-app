from datetime import datetime
from typing import List, Dict, Optional
from models.abstract_model import AbstractModel
from models.mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
    PyObjectId
)
from models.prompt import PromptSchema

class Setting(AbstractModel):
    __modelname__ = 'settings'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class SettingSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    sessionId: str = Field(description='downcased alphanumeric session id')
    activeModel: str = Field(description='active model of user')
    # customPrompts: Optional[Dict[str,str]] = Field(description='custom templates per model', default_factory=dict)
    hideEmojiOnSidebar: Optional[bool] = Field(description='hide emoji on sidebar', default=False)
    ethicsModalAcceptedAt: datetime = Field(default_factory=datetime.now)
    prompts: List[PromptSchema] = Field(description='List of prompts associated with user setting', default_factory=[])

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateSettingSchema(ChatSchema):
    activeModel: Optional[str] = None
    # customPrompts: Optional[Dict[str,str]] = None
    hideEmojiOnSidebar: Optional[bool] = None
    prompts: Optional[List[PromptSchema]] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingCollectionSchema(ChatSchema):
    settings: List[SettingSchema]