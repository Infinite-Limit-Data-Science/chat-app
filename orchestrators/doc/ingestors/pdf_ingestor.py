import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor

class PdfIngestor(DocumentIngestor):
    def load(self) -> List[Document]:
        loader = PyPDFLoader(self._file)
        docs = loader.load()
        return docs
    
    def chunk(
            self, 
            docs: List[Document], 
            metadata: dict, 
            chunk_size: int = 500, 
            chunk_overlap: int = 100) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        chunks_with_metadata = [
            Document(
                page_content=chunk.page_content, 
                metadata=metadata
            ) for chunk in chunks
        ]
        return chunks_with_metadata