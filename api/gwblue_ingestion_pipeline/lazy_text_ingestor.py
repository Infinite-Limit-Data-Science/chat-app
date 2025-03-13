from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from .document_ingestor import DocumentIngestor


class LazyTextIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = TextLoader(self._file)
        for doc in loader.lazy_load():
            yield doc

    load = lazy_load
