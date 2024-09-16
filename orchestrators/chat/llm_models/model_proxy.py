import logging
import socket
from urllib.parse import urlparse
from typing import List
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.chat.llm_models.llm import LLM

class ModelProxy:
    def __init__(self, models: List[LLM]) -> None:
        self.models = models

    def reachable(self) -> LLM:
        for model in self.models:
            try:
                url = urlparse(model.endpoint['url'])
                socket.create_connection((url.hostname, url.port), timeout=3)
                return model
            except (ConnectionRefusedError, TimeoutError) as e:
                logging.warning(f'Failed to reach endpoint {e} with {model.endpoint['url']}')
                continue        
        raise Exception('No models responded within 3 seconds')
    
    def runnable(self) -> HuggingFaceEndpoint:
        model = self.reachable()
        return model.endpoint_object