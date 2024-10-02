import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from orchestrators.doc.document_loaders.word_loader import WordLoader

class WordIngestor(DocumentIngestor):
    def load(self) -> List[Document]:
        loader = WordLoader(self._file)
        docs = loader.load()
        return docs
    
    def chunk(
            self,
            docs: List[Document], 
            metadata: dict, 
            chunk_size: int = 1000, 
            chunk_overlap: int = 150) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        chunks_with_metadata = [
            Document(
                page_content=chunk.page_content, 
                metadata=metadata
            ) for chunk in chunks
        ]
        return chunks_with_metadata