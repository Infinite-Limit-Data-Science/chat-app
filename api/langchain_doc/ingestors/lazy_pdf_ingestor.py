from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .document_ingestor import DocumentIngestor

class LazyPdfIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = PyPDFLoader(self._file)
        for doc in loader.lazy_load():
            yield doc
    load = lazy_load