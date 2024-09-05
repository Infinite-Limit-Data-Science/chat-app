import os
import json
from typing import List
from langchain_huggingface import HuggingFaceEndpoint
from llm_schema import TGISchema

model_config: List[dict] = json.dumps(os.environ['MODELS'])

# Usage: HFInferenceClientLLM(name="meta-llama/Meta-Llama-3.1-8B-Instruct"), HFInferenceClientLLM(name="meta-llama/Meta-Llama-3.1-70B-Instruct")
class HuggingFaceInferenceClient:
    """This class is a wrapper to the HuggingFace Inference Client"""
    def __init__(self):
        self.endpoints = []

    def load_models():
        for config in model_config:
            TGISchema(**config)