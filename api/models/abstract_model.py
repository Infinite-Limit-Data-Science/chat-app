from typing import List
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Get model by name"""
        pass

    @classmethod
    def chat_ui_compatible(cls) -> List[str]:
        """For backward compatibility with HuggingFace chat-ui"""
        return []
