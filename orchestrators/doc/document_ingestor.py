from typing import List
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.redistore import RediStore as VectorStore

class DocumentIngestor:
    def __init__(self, file: str, uuid: str, conversation_id: str):
        self.file = file
        self.uuid = uuid
        self.conversation_id = conversation_id
        self.vector_store = VectorStore

    def load(self) -> List[Document]:
        loader = PyPDFLoader(self.file)
        doc = loader.load()
        return doc

    def split(self, doc: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(doc)
        metadata = { 'uuid': self.uuid, 'conversation_id': self.conversation_id }
        chunks_with_metadata = [Document(page_content=chunk.page_content, metadata=metadata) for chunk in chunks]
        return chunks_with_metadata

    async def embed(self, chunks) -> List[str]:
        ids = await self.vector_store.add(chunks)
        logging.warning(f'first embedded id {ids[:1]}')
        return ids

    async def ingest(self) -> List[str]:
        doc = self.load()
        chunks = self.split(doc)
        return await self.embed(chunks)