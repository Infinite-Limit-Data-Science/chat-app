from typing import (
    Optional, 
    Iterator, 
    Sequence,
)
import json
import asyncio
from langchain_core.stores import BaseStore
from langchain_core.documents import Document

class RedisDocStore(BaseStore):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def set(self, doc_id: str, doc: Document) -> None:
        self.mset([(doc_id, doc)])
    
    def mset(self, docs: list[tuple[str, Document]]):
        for (doc_id, doc) in docs:
            obj = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            serialized = json.dumps(obj)
            self.redis.set(f"docstore:{doc_id}", serialized)

    async def amset(self, docs: list[tuple[str, Document]]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.mset, docs)

    def mget(self, ids: list[str]) -> list[Document]:
        docs = []
        for doc_id in ids:
            raw_val = self.redis.get(f"docstore:{doc_id}")
            if not raw_val:
                docs.append(None)
                continue
            data = json.loads(raw_val)
            doc = Document(
                page_content=data["page_content"],
                metadata=data["metadata"]
            )
            doc.metadata["doc_id"] = doc_id
            docs.append(doc)
        return docs

    async def amget(self, ids: list[str]) -> list[Document]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.mget, ids)
    
    def mdelete(self, ids: Sequence[str]) -> None:
        for doc_id in ids:
            self.redis.delete(f"docstore:{doc_id}:content")

    async def amdelete(self, ids: list[str]) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.mdelete, ids)

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        pattern = f"docstore:{prefix or '*'}:content"
        for key in self.redis.scan_iter(pattern):
            short_key = key.decode("utf-8")
            short_key = short_key.replace("docstore:", "", 1).replace(":content", "")
            yield short_key