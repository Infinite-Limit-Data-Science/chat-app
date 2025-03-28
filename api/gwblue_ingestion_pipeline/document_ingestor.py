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
import itertools
from redis.client import Redis
from langchain_redis import RedisConfig
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict
from ..gwblue_text_splitters.mixed_content_text_splitter import MixedContentTextSplitter
from ..gwblue_vectorstores.redis.config import VectorStoreSchema
from ..gwblue_vectorstores.redis.multimodal_vectorstore import MultiModalVectorStore
from ..gwblue_huggingface.huggingface_transformer_tokenizers import (
    get_tokenizer_class_by_prefix,
)

_VECTOR_TTL_30_DAYS = 3600 * 24 * 30

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
        embeddings_config: EmbeddingsConfig,
    ):
        self._file = file
        self.embeddings = embeddings
        self.metadata = metadata
        self.local_tokenizer = get_tokenizer_class_by_prefix(embeddings_config.model)(embeddings_config.model)

        config = RedisConfig(
            **{
                "redis_client": vector_config.client,
                "metadata_schema": vector_config.metadata_schema,
                "embedding_dimensions": self.local_tokenizer.vector_dimension_length,
                **VectorStoreSchema().model_dump(),
            }
        )

        self.vector_store = MultiModalVectorStore(self.embeddings, config=config)

    def inspect(documents: Iterator[Document]):
        import itertools

        c1, c2 = itertools.tee(documents)
        print(next(c1))
        print(next(c2))

    @abstractmethod
    def load(self) -> Iterator[Document]: ...

    def chunk(
        self,
        docs: Iterator[Document],
    ) -> Iterator[Document]:
        overlap_sampler = list(
            map(pow, itertools.repeat(0.05, times=4), itertools.count())
        )

        sequence_length = round(self.local_tokenizer.sequence_length_forward_pass, -2)
        overlap = int(sequence_length * overlap_sampler[1])

        encode_fn = lambda text: self.local_tokenizer.tokenizer.encode(
            text, 
            add_special_tokens=False
        )

        decode_fn = lambda token_ids: self.local_tokenizer.tokenizer.decode(
            token_ids, 
            skip_special_tokens=True
        )

        mixed_content_splitter = MixedContentTextSplitter(
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            chunk_size=sequence_length,
            chunk_overlap=overlap,
            metadata=self.metadata,
        )

        for chunk in mixed_content_splitter.split_documents_stream(docs):
            yield chunk

    async def embed(self, chunks: Iterator[Document]) -> List[str]:
        requests = self.local_tokenizer.max_batch_tokens_forward_pass // self.local_tokenizer.sequence_length_forward_pass
        return await self.vector_store.aadd_batch_with_ttl(
            chunks, ttl_seconds=_VECTOR_TTL_30_DAYS, max_requests=requests
        )

    async def ingest(self) -> List[str]:
        """Template Method"""
        docs = self.load()
        chunks = self.chunk(docs)
        return await self.embed(chunks)
