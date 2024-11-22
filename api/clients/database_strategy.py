from abc import ABC, abstractmethod

class DatabaseStrategy(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_database(self):
        pass