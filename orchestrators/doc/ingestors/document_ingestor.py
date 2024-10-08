import logging
from typing import List, Iterator
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractVectorStore
from orchestrators.doc.runnable_extensions.wrapper_runnable_config import WrapperRunnableConfig

class DocumentIngestor(ABC):
    def __init__(
        self, 
        file: str, 
        vector_store: AbstractVectorStore, 
        wrapper_runnable_config: WrapperRunnableConfig):
        """Abstract Ingestor takes file, metadata, and abstract VectorStore Bridge"""
        self._file = file
        self._wrapper_runnable_config = wrapper_runnable_config
        self._vector_store_bridge = vector_store

    @abstractmethod
    def load(self) -> Iterator[Document]:
        pass

    @abstractmethod
    def chunk(
        self, 
        docs: Iterator[Document], 
        metadata: dict, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150) -> Iterator[List[Document]]:
        pass

    async def embed(self, chunks) -> List[str]:
        ids = await self._vector_store_bridge.aadd(chunks)
        return ids

    @property
    def fetch_metadata(self):
        return {
            k: str(v) 
            for k, v in self._wrapper_runnable_config['metadata'].items() if k != 'schema'
        }

    async def ingest(self) -> List[str]:
        """Factory Method"""
        docs = self.load()
        chunks = self.chunk(docs, self.fetch_metadata)
        return await self.embed(chunks)