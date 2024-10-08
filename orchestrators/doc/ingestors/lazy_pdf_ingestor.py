import logging
from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor

class LazyPdfIngestor(DocumentIngestor):
    def load(self) -> Iterator[Document]:
        loader = PyPDFLoader(self._file)
        for doc in loader.lazy_load():
            yield doc
    
    def chunk(
        self, 
        docs: Iterator[Document], 
        metadata: dict, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150) -> Iterator[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            yield Document(
                page_content=chunk.page_content,
                metadata=metadata)