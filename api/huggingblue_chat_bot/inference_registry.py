from typing import Protocol
from ..chat_bot_config import ChatBotConfig

class BaseEngineLoader(Protocol):
    def __call__(self, config: ChatBotConfig) -> "EngineArtifacts":
        ...

class EngineArtifacts:
    def __init__(self, chat_model=None, safety_model=None, embedding_model=None):
        self.chat_model = chat_model
        self.safety_model = safety_model
        self.embedding_model = embedding_model

def huggingface_loader(config: ChatBotConfig) -> EngineArtifacts:
    """
    Build the relevant huggingface-based models
    from config.llm/guardrails/embeddings as needed.
    """
    from .language_models.huggingface import HuggingFaceInference
    inference = HuggingFaceInference(config=config)
    return EngineArtifacts(
        chat_model=inference.chat_model,
        safety_model=inference.safety_model,
        embedding_model=None  # or some embeddings object if appropriate
    )
