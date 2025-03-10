from typing import Iterator
from langchain_core.documents import Document
from .document_ingestor import DocumentIngestor
from ..gwblue_document_loaders.loaders import (
    PowerPointLoader
)

class LazyPowerPointIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = PowerPointLoader(self._file)
        for doc in loader.lazy_load():
            yield doc
    load = lazy_load