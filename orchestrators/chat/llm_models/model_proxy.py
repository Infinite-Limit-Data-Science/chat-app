from typing import List
import requests
from orchestrators.chat.llm_models.llm import LLM

class ModelProxy:
    def __init__(self, models: List[LLM]) -> None:
        self.models = models

    def latency(self) -> LLM:
        for model in self.models:
            try:
                response = requests.get(model.endpoint_url, timeout=3)
                response.raise_for_status()
                return model
            except requests.exceptions.RequestException:
                continue
        raise Exception('No models responded within 3 seconds')