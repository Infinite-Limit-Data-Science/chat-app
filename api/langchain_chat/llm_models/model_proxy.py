import socket
from urllib.parse import urlparse
from typing import List
from ..logger import logger
from .llm import LLM

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
                logger.warning(f'Failed to reach endpoint {e} with {model.endpoint['url']}')
                continue        
        
        raise TimeoutError('No models responded within 3 seconds')
    
    def get(self) -> LLM:
        """Return a runnable"""
        # model = self.reachable()
        return self.models[0]