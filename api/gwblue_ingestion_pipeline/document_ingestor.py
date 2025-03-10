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
from ..gwblue_vectoretrievers.redis.config import VectorStoreSchema
from ..gwblue_vectoretrievers.redis.vectorstore import RedisVectorStoreTTL
from ..gwblue_huggingface.huggingface_transformer_tokenizers import (
    get_tokenizer_class_by_prefix
)

_VECTOR_TTL_30_DAYS = 3600 * 24 * 30

"""
Even if embeddings model has large sequence length where
single sequence can be up to 131,072 tokens (as in the case
of vlm2vec), we want to ensure each context chunk we later 
feed to model in chat prompt template is not too large.
"""
_MAX_TOKENS_PER_INPUT = 2000

VectorStoreClient: TypeAlias = Union[Redis]

class VectorStoreConfig(BaseModel):
    client: Optional[VectorStoreClient] = None
    url: Optional[str] = None
    metadata_schema: List[Dict[str, Any]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

class EmbeddingsConfig(TypedDict):
    name: str
    endpoint: str
    token: str
    server: str
    dimensions: int

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
        self.local_tokenizer = get_tokenizer_class_by_prefix(embeddings_config.name)()
        
        config = RedisConfig(**{
            'redis_client': vector_config.client,
            'metadata_schema': vector_config.metadata_schema,
            'embedding_dimensions': self.local_tokenizer.dimensions,
            **VectorStoreSchema().model_dump()
        })
            
        self.vector_store = RedisVectorStoreTTL(
            self.embeddings, 
            config=config
        )

    def inspect(documents: Iterator[Document]):
        import itertools

        c1, c2 = itertools.tee(documents)
        print(next(c1))
        print(next(c2))

    @abstractmethod
    def load(self) -> Iterator[Document]:
        ...

    def chunk(
        self, 
        docs: Iterator[Document],
    ) -> Iterator[Document]:
        overlap_sampler = list(map(pow, itertools.repeat(0.05,times=4), itertools.count()))

        sequence_length = _MAX_TOKENS_PER_INPUT if self.local_tokenizer.empirical_sequence_length > _MAX_TOKENS_PER_INPUT else self.local_tokenizer.empirical_sequence_length
        overlap = int(sequence_length * overlap_sampler[1])
        len_function = lambda text: len(self.local_tokenizer.tokenizer.encode(text))

        mixed_content_splitter = MixedContentTextSplitter(
            chunk_size=sequence_length,
            chunk_overlap=overlap,
            length_function=len_function,
            metadata=self.metadata,
        )
        chunks = mixed_content_splitter.split_documents(docs)

        for chunk in chunks:
            yield chunk

    async def embed(self, chunks: Iterator[Document]) -> List[str]:
        requests = self.local_tokenizer.recommended_token_batch // _MAX_TOKENS_PER_INPUT   
        return await self.vector_store.aadd_documents_with_ttl(
            chunks,
            ttl_seconds=_VECTOR_TTL_30_DAYS,
            max_requests=requests
        )

    async def ingest(self) -> List[str]:
        """Template Method"""
        docs = self.load()
        chunks = self.chunk(docs)
        return await self.embed(chunks)