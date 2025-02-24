from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel

class VLLMInference(BaseModel):
    base_model: BaseChatModel = None
    safety_model: BaseChatModel = None