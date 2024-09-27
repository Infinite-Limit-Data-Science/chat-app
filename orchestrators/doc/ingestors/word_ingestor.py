import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor

class WordIngestor(DocumentIngestor):
    def load(self) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(self._file)
        doc = loader.load()
        return doc
    
    def chunk(
            self,
            doc: List[Document], 
            metadata: dict, 
            chunk_size: int = 500, 
            chunk_overlap: int = 100) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(doc)
        chunks_with_metadata = [
            Document(
                page_content=chunk.page_content, 
                metadata=metadata
            ) for chunk in chunks
        ]
        return chunks_with_metadata
    
    async def embed(self, chunks) -> List[str]:
        ids = await self._vector_store_bridge.aadd(chunks)
        logging.warning(f'first embedded id {ids[:1]}')
        return ids