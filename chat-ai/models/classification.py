import os
import json
import types
from enum import Enum
from typing import Callable, Any
from abc import ABC, abstractmethod

class Generations(Enum):
    TEXT_GENERATION = 'text-generation'
    CONTENT_SAFETY = 'content-safety'

class ActiveStrategy(ABC):
    def __init__(self):
        self.generate_classification_methods()
    
    @abstractmethod
    def set_active(self, model_configs: dict, active_model_name: str) -> None:
        """Apply active/inactive state changes to models in this classification."""
        pass
    
    @abstractmethod
    def get_classification(self) -> str:
        pass        

    @staticmethod
    def load_classifications_from_env() -> set:
        models_json = os.getenv('MODELS', '[]')
        models = json.loads(models_json)
        
        classifications = {model['classification'] for model in models if 'classification' in model}
        
        return classifications

    def classify(self, classification: str) -> Callable[[Any], bool]:
        def method(self) -> bool:
            return self.get_classification() == classification
        return method
    
    def generate_classification_methods(self) -> None:
        """Metaprogram classification methods based on the MODELS environment variable."""
        for generation in Generations:
            method_name = f'is_{generation.value.replace("-", "_")}'

            if not hasattr(self, method_name):
                setattr(
                    self, 
                    method_name, 
                    types.MethodType(self.classify(generation.value), self))

class TextGenerationActiveStrategy(ActiveStrategy):
    def set_active(self, model_configs: dict, active_model_name: str) -> None:
        for model_name, config in model_configs.items():
            config.active = model_name == active_model_name

    def get_classification(self) -> str:
        return Generations.TEXT_GENERATION.value

# class ImageToTextActiveStrategy(ActiveStrategy):
#     def set_active(self, model_configs: dict, active_model_name: str) -> None:
#         for model_name, config in model_configs.items():
#             config.active = model_name == active_model_name

#     def get_classification(self) -> str:
#         return 'image_to_text'

class ContentSafetyStrategy(ActiveStrategy):
    def set_active(self, model_configs: dict, active_model_name: str) -> None:
        pass

    def get_classification(self) -> str:
        return Generations.CONTENT_SAFETY.value
    
STRATEGY_MAP = {
    Generations.TEXT_GENERATION.value: TextGenerationActiveStrategy(),
    # 'image-to-text': ImageToTextActiveStrategy(),
    Generations.CONTENT_SAFETY.value: ContentSafetyStrategy(),
}

def get_strategy_for_classification(classification: str) -> ActiveStrategy:
    return STRATEGY_MAP.get(classification, None)