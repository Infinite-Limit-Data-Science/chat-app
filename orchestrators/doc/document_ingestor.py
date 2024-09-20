from typing import List
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.embedding_models.model_proxy import ModelProxy
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.redistore import RediStore as VectorStore

class DocumentIngestor:
    def __init__(self, file: str, embeddings_models: List[BaseEmbedding], uuid: str, conversation_id: str):
        self._file = file
        self._uuid = uuid
        self._conversation_id = conversation_id
        self._vector_store = VectorStore(ModelProxy(embeddings_models), uuid, conversation_id)

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self._file)
        doc = loader.load()
        return doc

    def split(self, doc: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(doc)
        metadata = { 'uuid': self._uuid, 'conversation_id': self._conversation_id }
        chunks_with_metadata = [Document(page_content=chunk.page_content, metadata=metadata) for chunk in chunks]
        return chunks_with_metadata

    async def embed(self, chunks) -> List[str]:
        ids = await self._vector_store.add(chunks)
        logging.warning(f'first embedded id {ids[:1]}')
        return ids

    async def ingest(self) -> List[str]:
        doc = self.load()
        chunks = self.split(doc)
        return await self.embed(chunks)