from abc import ABC, abstractmethod

class AbstractChat(ABC):
    @abstractmethod
    def prompt(self):
        pass

    @abstractmethod
    def llm(self):
        pass

    @abstractmethod
    def embedding(self):
        pass

    def template_method(self):
        self.template()
        self.llm()
        self.embedding()
        self.output_parser()