from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    @classmethod
    def get_model_name():
        """Get model by name"""
        pass

    def template_method():
        """Not Yet Implemented"""
        pass