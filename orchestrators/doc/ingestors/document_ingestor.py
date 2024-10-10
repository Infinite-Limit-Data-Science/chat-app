import logging
from typing import List, Iterator
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractVectorStore
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractFlexiSchemaFields

def meta_format(metadata: List[AbstractFlexiSchemaFields]) -> dict:
    return {
        data['name']: data['value']
        for data in metadata
    }

class DocumentIngestor(ABC):
    def __init__(
        self, 
        file: str, 
        vector_store: AbstractVectorStore, 
        metadata: List[AbstractFlexiSchemaFields]):
        """Abstract Ingestor takes file, metadata, and abstract VectorStore Bridge"""
        self._file = file
        self._metadata = meta_format(metadata)
        self._vector_store_bridge = vector_store
    
    @abstractmethod
    def load(self) -> Iterator[Document]:
        pass

    def chunk(
        self, 
        docs: Iterator[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150) -> Iterator[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            yield Document(
                page_content=chunk.page_content,
                metadata=self._metadata) # chunk.metadata also contains page number

    async def embed(self, chunks: Iterator[Document]) -> List[str]:
        return await self._vector_store_bridge.aadd(chunks)

    async def ingest(self) -> List[str]:
        """Template Method"""
        docs = self.load()
        chunks = self.chunk(docs)
        return await self.embed(chunks)