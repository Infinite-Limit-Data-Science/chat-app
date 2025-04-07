from typing import (
    Iterator,
    List,
    Optional,
    Any,
)
import uuid
from langchain_core.documents import Document
from langchain.retrievers import MultiVectorRetriever#, ParentDocumentRetriever
from ..gwblue_text_splitters.streaming_text_splitter import StreamingTextSplitter

class StreamingParentDocumentRetriever(MultiVectorRetriever):
    """
    Validate:
    FT.SEARCH user_conversations "*" LIMIT 0 10
    TTL user_conversations:01JQX3EE253R2SFT6C94ZGSFBH 
    GET docstore:50c018b8-5c57-4a1d-9c97-92d5d6a89097:content 
    """
    child_splitter: StreamingTextSplitter

    def _split_docs_for_adding(
        self,
        documents: Iterator[Document],
        add_to_docstore: bool,
        **kwargs: Any,
    ) -> Iterator[Document]:
        if not add_to_docstore and self.id_key is None:
            raise ValueError(
                "If add_to_docstore=False, you must supply doc IDs via metadata "
                "or another mechanism so child docs can reference a known parent."
            )
        
        for parent_doc in documents:
            # If it's an "image-only" parent doc, skip storing in docstore since
            # there is no reason to chunk an image into smaller chunks
            if parent_doc.metadata.get("chunk_type") == "image":
                yield parent_doc
                continue

            if self.id_key in parent_doc.metadata:
                doc_id = parent_doc.metadata[self.id_key]
            else:
                doc_id = str(uuid.uuid4())
                parent_doc.metadata[self.id_key] = doc_id

            if add_to_docstore:
                self.docstore.set(doc_id, parent_doc)
        
            child_doc_stream: Iterator[Document] = self.child_splitter.split_documents([parent_doc])

            # send a batch of child doc chunk streams for given parent
            for child_doc in child_doc_stream:
                child_doc.metadata[self.id_key] = doc_id         
                yield child_doc

        
    def add_documents(
        self,
        documents: Iterator[Document],
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        for child_doc in self._split_docs_for_adding(documents, add_to_docstore):
            self.vectorstore.aadd_documents([child_doc])

    async def aadd_documents(
        self,
        documents: Iterator[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> List[str]:
        for child_doc in self._split_docs_for_adding(documents, add_to_docstore):
            await self.vectorstore.aadd_documents([child_doc])
    
    async def aadd_document_batch(
        self,
        documents: Iterator[Document],
        max_requests: int,
        add_to_docstore: bool = True,
        **kwargs: Any,            
    ) -> List[str]:
        """
        `MultiModalVectorStore.aadd_batch` accumulates text chunks up 
        to a certain batch size, flushes them, handles images individually, 
        and then continues.

        When `_split_docs_for_adding` yields child docs for a single
        parent doc, `aadd_batch` just sees them as a sequence 
        of documents. Even if you have multiple text child chunks for 
        a single parent doc, `aadd_batch` will accumulate them 
        in its internal batch until the limit is reached (or until it 
        sees an image).

        By the time `aadd_batch` starts reading the stream of 
        child docs, each parent doc has already been stored in the docstore 
        (because _split_docs_for_adding sets it there before yielding the 
        child docs).
        """
        doc_stream = self._split_docs_for_adding(documents, add_to_docstore)
        return await self.vectorstore.aadd_batch(doc_stream, max_requests=max_requests)       