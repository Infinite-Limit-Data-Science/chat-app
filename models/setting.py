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
from models.model_config import ModelConfigSchema

class Setting(AbstractModel):
    __modelname__ = 'settings'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

class SettingSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: str = Field(alias="sessionId", description='downcased alphanumeric session id')
    activeModel: Optional[str] = Field(description='active model of user', default=None)
    hideEmojiOnSidebar: Optional[bool] = Field(description='hide emoji on sidebar', default=False)
    ethicsModalAcceptedAt: datetime = Field(default_factory=datetime.now)
    prompts: Optional[List[PromptSchema]] = Field(description='List of prompts associated with user setting', default_factory=list)
    my_model_configs: Optional[List[ModelConfigSchema]] = Field(alias='model_configs', description='List of model configs associated with user settings', default_factory=list)
    # Legacy attributes
    customPrompts: Optional[Dict[str,str]] = Field(description='Legacy attribute', default = None)
    shareConversationsWithModelAuthors: Optional[bool] = Field(description='Legacy attribute', default=None)
    modelParameters: Optional[dict] = Field(description='Legacy attribute', default=None)

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
    model_configs: Optional[List[ModelConfigSchema]] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingCollectionSchema(ChatSchema):
    settings: List[SettingSchema]