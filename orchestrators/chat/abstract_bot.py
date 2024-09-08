from abc import ABC, abstractmethod

class AbstractBot(ABC):
    @abstractmethod
    def prompt(self):
        pass

    @abstractmethod
    def llm(self):
        pass

    @abstractmethod
    def embedding(self):
        pass

    def chat(self):
        """Template Method"""
        self.template()
        self.llm()
        self.embedding()
        self.output_parser()