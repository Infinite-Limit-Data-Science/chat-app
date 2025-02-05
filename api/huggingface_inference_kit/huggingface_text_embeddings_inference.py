from typing import TypeVar, Generic
from pydantic import Field
from .inference_schema import InferenceKit
from .huggingface_inference_client import BaseInferenceClient

T = TypeVar('T', bound=BaseInferenceClient)

class HuggingFaceTextEmbeddingsInference(Generic[T], InferenceKit):
    pass