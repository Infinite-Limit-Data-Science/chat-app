from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @classmethod
    @abstractmethod
    def get_model_name():
        """Get model by name"""
        pass

    def template_method():
        """Not Yet Implemented"""
        pass