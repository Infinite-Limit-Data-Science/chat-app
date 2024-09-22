from abc import ABC, abstractmethod
from typing import Callable, AsyncGenerator

class AbstractBot(ABC):
    @abstractmethod
    async def astream(self, **kwargs) -> Callable[[], AsyncGenerator[str, None]]:
        pass