from abc import ABC, abstractmethod
from typing import Iterable, Iterator
from langchain_core.documents import Document

class StreamingTextSplitter(ABC):
    """
    Minimal interface for something that splits documents into sub-documents
    in a streaming fashion
    """

    @abstractmethod
    def split_documents(self, docs: Iterable[Document]) -> Iterator[Document]:
        pass