from typing import List, Iterator
from abc import ABC, abstractmethod
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..vector_stores import AbstractVectorStore
from ...langchain_chunkinator import Chunkinator
from ..task_execution_context import filename_var

class DocumentIngestor(ABC):
    def __init__(
        self, 
        file: str, 
        vector_store: AbstractVectorStore, 
        metadata: dict
    ):
        """Abstract Ingestor takes file, metadata, and abstract VectorStore Bridge"""
        self._file = file
        self._vector_store_bridge = vector_store
        self._metadata = metadata
        self.smart_chunking = True

        filename_var.set(self._file)
    
    @abstractmethod
    def load(self) -> Iterator[Document]:
        pass

    def chunk(
        self, 
        docs: Iterator[Document], 
        chunk_size: int = 1000, 
        chunk_overlap: int = 150) -> Iterator[Document]:
        if self.smart_chunking:
            chunkinator = Chunkinator.Base(docs, self._vector_store_bridge)
            chunks = chunkinator.chunk()                
        else:
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