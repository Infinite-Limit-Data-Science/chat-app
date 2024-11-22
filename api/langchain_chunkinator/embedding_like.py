from typing import Protocol, Annotated
from typing_extensions import Doc

class EmbeddingLike(Protocol):
    @property
    def name(self) -> Annotated[str, Doc('Name of embedding model')]:
        ...
    
    @property
    def max_batch_tokens(self) -> Annotated[int, Doc('Max tokens per batch')]:
        ...

    @property
    def max_batch_requests(self) -> Annotated[int, Doc('Max requests per batch')]:
        ...