from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    Any,
)
import uuid
from langchain_core.documents import Document
from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from ..gwblue_text_splitters.streaming_text_splitter import StreamingTextSplitter

class StreamingParentDocumentRetriever(MultiVectorRetriever):
    """
    Validate:
    FT.SEARCH user_conversations "*" LIMIT 0 10
    TTL user_conversations:01JQX3EE253R2SFT6C94ZGSFBH 
    GET docstore:50c018b8-5c57-4a1d-9c97-92d5d6a89097:content 
    """
    child_splitter: StreamingTextSplitter = None

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
    
    def _aggregate_parents(
        self, child_docs_with_scores: List[Tuple[Document, float]]
    ) -> List[str]:
        from collections import defaultdict
        parent_scores = defaultdict(float)
        for (child_doc, score) in child_docs_with_scores:
            parent_id = child_doc.metadata[self.id_key]
            parent_scores[parent_id] += score
        
        sorted_parent_ids = sorted(parent_scores.keys(),
                                   key=lambda pid: parent_scores[pid],
                                   reverse=True)
        return sorted_parent_ids

    def _filter_valid_docs(self, docs: List[Document]) -> List[Document]:
        return [d for d in docs if d is not None]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        child_docs_with_distances = self.vectorstore.similarity_search_with_score(
            query,
            **self.search_kwargs
        )
        
        child_docs_with_scores = []
        for doc, distance in child_docs_with_distances:
            score = 1.0 / (1.0 + distance)
            child_docs_with_scores.append((doc, score))
        
        sorted_parent_ids = self._aggregate_parents(child_docs_with_scores)
        parent_docs = self.docstore.mget(sorted_parent_ids)
        return self._filter_valid_docs(parent_docs)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Each child doc has a numeric similarity score
        _aggregate_parents adds up those child-level scores for all child docs belonging to a given parent.
        """
        child_docs_with_distances = await self.vectorstore.asimilarity_search_with_score(
            query,
            **self.search_kwargs
        )
        child_docs_with_scores = []
        for doc, distance in child_docs_with_distances:
            score = 1.0 / (1.0 + distance)
            child_docs_with_scores.append((doc, score))

        # TODO: For the top k documents, use chunk_index to get one full document forward
        # even if that document was not included in the returned chunks. This ensures if
        # important content spills over to next page, it will be covered. 
        # chat bot should take the results and if context length is reached, then trim
        # out documents by worst scores ones first. But it must also order documents by
        # chunk index, or should I do the ordering by chunk index in this method? ANd 
        # pass the scores as document metadata to the chat bot? 
        sorted_parent_ids = self._aggregate_parents(child_docs_with_scores)
        parent_docs = await self.docstore.amget(sorted_parent_ids)

        # for parent_id in sorted_parent_ids:
        #     doc = parent_docs_map[parent_id]
        #     doc.metadata["aggregated_score"] = parent_scores[parent_id]

        return self._filter_valid_docs(parent_docs)