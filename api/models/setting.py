from datetime import datetime
from typing import List, Optional
from .abstract_model import AbstractModel
from .mongo_schema import (
    ChatSchema,
    PrimaryKeyMixinSchema,
    TimestampMixinSchema,
    Field,
    PyObjectId
)
from .prompt import PromptSchema
from .user_model_config import UserModelConfigSchema, UpdateUserModelConfigSchema

class Setting(AbstractModel):
    __modelname__ = 'settings'
    
    @classmethod
    def get_model_name(cls):
        return cls.__modelname__

    @classmethod
    def chat_ui_compatible(cls) -> List[str]:
        return ['custom_prompt','model_name','prompts', 'user_model_configs', 'createdAt', 'updatedAt']

class SettingSchema(PrimaryKeyMixinSchema, TimestampMixinSchema):
    uuid: str = Field(alias="sessionId", description='downcased alphanumeric session id')
    activeModel: Optional[str] = Field(description='active model of user', default=None)
    hideEmojiOnSidebar: Optional[bool] = Field(description='hide emoji on sidebar', default=False)
    ethicsModalAcceptedAt: datetime = Field(default_factory=datetime.now)
    prompts: Optional[List[PromptSchema]] = Field(description='List of prompts associated with user setting', default_factory=list)
    user_model_configs: Optional[List[UserModelConfigSchema]] = Field(description='List of user model configs associated with user settings', default_factory=list)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingIdSchema(ChatSchema):
    id: PyObjectId = Field(alias="_id", description='bson object id')

class UpdateSettingSchema(ChatSchema):
    activeModel: Optional[str] = None
    hideEmojiOnSidebar: Optional[bool] = None
    prompts: Optional[List[PromptSchema]] = None
    user_model_configs: Optional[List[UpdateUserModelConfigSchema]] = None
    updatedAt: datetime = Field(default_factory=datetime.now)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class SettingCollectionSchema(ChatSchema):
    settings: List[SettingSchema]