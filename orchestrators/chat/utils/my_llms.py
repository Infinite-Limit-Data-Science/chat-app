from typing import Callable, Any, TypeVar
from functools import wraps
from langchain_huggingface.chat_models.huggingface import _is_huggingface_endpoint

F = TypeVar('F', bound=Callable[..., Any])

def assert_valid_hf_llm(func: F) -> F:
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert _is_huggingface_endpoint(self.llm), 'The LLM must be a HuggingFaceEndpoint.'
        assert len(list(self)) > 1, 'Expected special tokens, got none'
        return func(self, *args, **kwargs)
    return wrapper