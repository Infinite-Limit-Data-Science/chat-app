import socket
from urllib.parse import urlparse
from typing import List
from .embedding import BaseEmbedding
from ..logger import logger

class ModelProxy:
    def __init__(self, models: List[BaseEmbedding]) -> None:
        self.models = models

    def reachable(self) -> BaseEmbedding:
        for model in self.models:
            try:
                url = urlparse(model.endpoint['url'])
                socket.create_connection((url.hostname, url.port), timeout=3)
                return model
            except (ConnectionRefusedError, TimeoutError) as e:
                logger.warning(f'Failed to reach endpoint {e} with {model.endpoint['url']}')
                continue     
           
        raise SystemError('No models responded within 3 seconds')
    
    def get(self) -> BaseEmbedding:
        """Return a runnable"""
        #model = self.reachable()
        return self.models[0]