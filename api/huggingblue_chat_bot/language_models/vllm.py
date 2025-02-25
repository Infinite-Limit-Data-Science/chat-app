from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from ..chat_bot_config import ChatBotConfig

class VLLMInference(BaseModel):
    config: ChatBotConfig
    chat_model: BaseChatModel = None
    safety_model: BaseChatModel = None