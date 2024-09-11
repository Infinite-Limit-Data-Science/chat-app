from abc import ABC, abstractmethod

class AbstractBot(ABC):
    @abstractmethod
    def runnable(self):
        pass