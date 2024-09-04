from abc import ABC, abstractmethod

class LLMWrapper(ABC):
    @abstractmethod
    def generate(self, template: str, values: dict, **kwargs) -> str:
        """Generate Text"""
        pass