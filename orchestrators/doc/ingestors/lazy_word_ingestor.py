import logging
from typing import Iterator
from langchain_core.documents import Document
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from orchestrators.doc.document_loaders.word_loader import WordLoader

class LazyWordIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = WordLoader(self._file)
        for doc in loader.lazy_load():
            yield doc
    load = lazy_load