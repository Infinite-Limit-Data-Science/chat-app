import logging
from typing import Iterator
from langchain_core.documents import Document
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from orchestrators.doc.document_loaders.power_point_loader import PowerPointLoader

class LazyPowerPointIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = PowerPointLoader(self._file)
        for doc in loader.lazy_load():
            yield doc
    load = lazy_load