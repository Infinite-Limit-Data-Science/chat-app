import logging
from typing import List, Iterator
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
                metadata={**chunk.metadata, **metadata})

    async def embed(self, chunks: Iterator[Document]) -> List[str]:
        return await self._vector_store_bridge.aadd(chunks)

    @property
    def fetch_metadata(self):
        return {
            k: str(v) 
            for k, v in self._wrapper_runnable_config['metadata'].items() if k != 'schema'
        }

    async def ingest(self) -> List[str]:
        """Template Method"""
        docs = self.load()
        chunks = self.chunk(docs, self.fetch_metadata)
        return await self.embed(chunks)