from typing import (
    List,
    Iterator,
    Dict,
    Any,
    TypeAlias,
    Union,
    Optional,
    TypedDict,
)
from abc import ABC, abstractmethod
from redis.client import Redis
from langchain_redis import RedisConfig
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from pydantic import BaseModel, ConfigDict
from ..gwblue_text_splitters.mixed_content_text_splitter import MixedContentTextSplitter
from ..gwblue_vectorstores.redis.config import VectorStoreSchema
from ..gwblue_vectorstores.redis.multimodal_vectorstore import MultiModalVectorStore
from ..gwblue_vectorstores.redis.docstore import RedisDocStore
from ..gwblue_retrievers.streaming_parent_document_retriever import StreamingParentDocumentRetriever

VectorStoreClient: TypeAlias = Union[Redis]

class VectorStoreConfig(BaseModel):
    client: Optional[VectorStoreClient] = None
    url: Optional[str] = None
    metadata_schema: List[Dict[str, Any]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class EmbeddingsConfig(TypedDict):
    model: str
    endpoint: str
    token: str
    provider: str
    max_batch_tokens: int


class DocumentIngestor(ABC):
    def __init__(
        self,
        file: str,
        *,
        embeddings: Embeddings,
        metadata: Dict[str, Any],
        vector_config: VectorStoreConfig,
        add_to_docstore: bool = True,
        docstore: BaseStore = None,
    ):
        self._file = file
        self.embeddings = embeddings
        self.metadata = metadata

        config = RedisConfig(
            **{
                "redis_client": vector_config.client,
                "metadata_schema": vector_config.metadata_schema,
                "embedding_dimensions": self.embeddings.tokenizer.vector_dimension_length,
                **VectorStoreSchema().model_dump(),
            }
        )
        self.vectorstore = MultiModalVectorStore(self.embeddings, config=config)

        self.add_to_docstore = add_to_docstore
        self.docstore = docstore
        if self.add_to_docstore and not self.docstore:
            self.docstore = RedisDocStore(vector_config.client) 

    def inspect(documents: Iterator[Document]):
        import itertools

        c1, c2 = itertools.tee(documents)
        print(next(c1))
        print(next(c2))

    @abstractmethod
    def load(self) -> Iterator[Document]: ...


    def chunk_strategy(self, documents: Iterator[Document]) -> Iterator[Document]:
        mixed_content_splitter = MixedContentTextSplitter(metadata=self.metadata)

        for chunk in mixed_content_splitter.split_documents(documents):
            yield chunk

    def inheritable_chunk_strategy(self, documents: Iterator[Document]) -> Iterator[Document]:
        parent_text_splitter = MixedContentTextSplitter(
            self.embeddings.tokenizer.tokenizer,
            chunk_size=2000,
            metadata=self.metadata,
        )

        return parent_text_splitter.split_documents(documents)

    def chunk(
        self,
        documents: Iterator[Document],
    ) -> Iterator[Document]:
        if self.add_to_docstore:
            return self.inheritable_chunk_strategy(documents)
        else:
            return self.chunk_strategy(documents)

    async def embed_strategy(
        self, 
        documents: Iterator[Document], 
        requests: int
    ) -> List[str]:
        return await self.vectorstore.aadd_batch(
            documents, max_requests=requests
        )
    
    async def inheritable_embed_strategy(
        self, 
        documents: Iterator[Document], 
        requests: int
    ) -> List[str]:
        child_text_splitter = MixedContentTextSplitter(
            self.embeddings.tokenizer.tokenizer,
            chunk_size=150,
            metadata=self.metadata,
        )

        retriever = StreamingParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_text_splitter,   
        )

        return await retriever.aadd_document_batch(
            documents=documents,
            max_requests=requests
        )

    async def embed(self, documents: Iterator[Document]) -> List[str]:
        requests = (
            self.embeddings.tokenizer.max_batch_tokens_forward_pass
            // self.embeddings.tokenizer.sequence_length_forward_pass
        )
        
        if self.add_to_docstore:
            return await self.inheritable_embed_strategy(documents, requests)
        else:
            return await self.embed_strategy(documents, requests)

    async def ingest(self) -> List[str]:
        """Template Method"""
        docs = self.load()
        chunks = self.chunk(docs)
        return await self.embed(chunks)
