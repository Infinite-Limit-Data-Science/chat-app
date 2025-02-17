from typing import TypeVar, Optional, Protocol, Annotated
from typing_extensions import Doc
from transformers import PreTrainedTokenizerBase
import importlib
import os
import json

T = TypeVar('T')
U = TypeVar('U')

TRANSFORMER_TOKENIZER_CACHE = 'transformers/tokenizers/cache'

def get_tokenizer_class_by_prefix(prefix: str):
    class_name = f'{prefix}PretrainedTokenizer'
    try:
        return globals()[class_name]
    except KeyError:
        raise ImportError(f'Could not find class {class_name} in {__name__}')
    
class PretrainedTokenizerLike(Protocol):
    tokenizer: Annotated[PreTrainedTokenizerBase, Doc('A reference to a Pretrained Tokenizer used for underlying model')]
    sequence_length: Annotated[int, Doc('Maximum tokens per pass')]
    dimensions: Annotated[int, Doc('Number of training features per vector')]

    def __init__(self, name: Optional[str] = None) -> None:
        """Initializes the tokenizer with a default model name."""
        ...

class TokenizerDescriptor:
    def __set_name__(self, owner: T, name: str):
        self.name = name

    def __get__(self, instance: U, owner: T) -> PreTrainedTokenizerBase:
        if instance:
            return getattr(instance, 'tokenizer', None)
        return getattr(owner, 'tokenizer', None)
        
    def __set__(self, instance: U, value: str):
        if not isinstance(value, str):
            raise ValueError(f'{self.name} must be a string')
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),TRANSFORMER_TOKENIZER_CACHE, value, 'tokenizer_config.json'), 'r') as f:
            tokenizer_config = json.load(f)
            tokenizer_class_name: PreTrainedTokenizerBase = tokenizer_config['tokenizer_class']
            transformers_module = importlib.import_module('transformers')
            cls = getattr(transformers_module, tokenizer_class_name)
            tokenizer = cls.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)),TRANSFORMER_TOKENIZER_CACHE, value))

        setattr(instance, 'tokenizer', tokenizer)
        setattr(instance, 'sequence_length', tokenizer.model_max_length)
        setattr(instance, 'dimensions', self.load_dimensions(instance, value))

    def load_dimensions(self, instance: U, value: str) -> int:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),TRANSFORMER_TOKENIZER_CACHE, value, 'config.json'), 'r') as f:
            config = json.load(f)
            dimensions = int(instance.extract_dimensions(config))
        return dimensions

class BgeLargePretrainedTokenizer:
    _tokenizer_name = TokenizerDescriptor()

    def __init__(self, name: Optional[str] = None):
        self._tokenizer_name = name or 'baai_bge_large_en_v1.5'

    @staticmethod
    def extract_dimensions(config: dict) -> int:
        return config.get('hidden_size')

class NomicPretrainedTokenizer:
    _tokenizer_name = TokenizerDescriptor()

    def __init__(self, name: Optional[str] = None):
        self._tokenizer_name = name or 'nomic_ai_nomic_embed_text_v1.5'

    @staticmethod
    def extract_dimensions(config: dict) -> int:
        return config.get('n_embd')
    
class Llama70BInstructPretrainedTokenizer:
    _tokenizer_name = TokenizerDescriptor()

    def __init__(self, name: Optional[str] = None):
        self._tokenizer_name = name or 'meta_llama_3_1_70B_Instruct'

    @staticmethod
    def extract_dimensions(config: dict) -> int:
        return config.get('hidden_size')
    
class Llama90BVisionInstructPretrainedTokenizer:
    _tokenizer_name = TokenizerDescriptor()

    def __init__(self, name: Optional[str] = None):
        self._tokenizer_name = name or 'meta_llama_3_2_90B_Vision_Instruct'
    
    @staticmethod
    def extract_dimensions(config: dict) -> int:
        return config.get('text_config').get('hidden_size')