from typing import List
from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @classmethod
    @abstractmethod
    def get_model_name(cls) -> str:
        """Get model by name"""
        pass

    @staticmethod
    def backward_compatible() -> List[str]:
        return []