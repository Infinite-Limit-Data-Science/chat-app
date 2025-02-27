from typing import Iterator, List, Any
import asyncio
from langchain_core.documents import Document
from langchain_redis import RedisVectorStore

class RedisVectorStoreTTL(RedisVectorStore):
    async def aadd_documents_with_ttl(
        self, 
        documents: Iterator[Document], 
        ttl_seconds: int,
        max_requests: int,
        **kwargs: Any
    ) -> List[str]:
        """
        Example:
        > EXISTS user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 1
        > TTL user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 2591813
        """
        redis_client = self.config.redis()

        semaphore = asyncio.Semaphore(max_requests)

        async def process_document(document: Document):
            async with semaphore:
                batch_ids = await self._process_batch([document], ttl_seconds, redis_client, **kwargs)
                return batch_ids
        
        tasks = [asyncio.create_task(process_document(document)) for document in documents]
        results = await asyncio.gather(*tasks)
        document_ids = [doc_id for batch_ids in results for doc_id in batch_ids]

        return document_ids
    
    async def _process_batch(
        self, 
        batch: List[Document], 
        ttl_seconds: int, 
        redis_client, 
        **kwargs: Any
    ) -> List[str]:
        batch_ids = await self.aadd_documents(batch, **kwargs)
        for doc_id in batch_ids:
            await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)

        return batch_ids