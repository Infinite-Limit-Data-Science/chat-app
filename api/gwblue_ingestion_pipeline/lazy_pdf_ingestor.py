from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from .document_ingestor import DocumentIngestor
from ..gwblue_document_loaders.parsers import Base64BlobParser


class LazyPdfIngestor(DocumentIngestor):
    def lazy_load(self) -> Iterator[Document]:
        loader = PyPDFLoader(
            self._file,
            extract_images=True,
            images_parser=Base64BlobParser(),
            images_inner_format="raw",
            mode="page",
        )
        for doc in loader.lazy_load():
            yield doc

    load = lazy_load
