from typing import (
    List, 
    Any, 
    TypeVar, 
    Iterable,
    Iterator, 
    AsyncIterator,
)
import asyncio
from langchain_core.documents import Document
from langchain_redis import RedisVectorStore
from redisvl.redis.utils import array_to_buffer

I = TypeVar("I")

async def _async_iterator_wrapper(iterator: Iterable[I]) -> AsyncIterator[I]:
    for item in iterator:
        yield item

class MultiModalVectorStore(RedisVectorStore):
    """
    Supports storing multimodal vectors

    Supports TTL:
        > EXISTS user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 1
        > TTL user_conversations:a2c8a48073ee4a429b6910b1cfefb9f4
        (integer) 2591813

    FT._LIST
    FT.INFO
    FT.SEARCH user_conversations "*" LIMIT 0 10
    FT.SEARCH user_conversations "*" LIMIT 0 0
    FT.INFO user_conversations
    FT.DROPINDEX user_conversations DD
    """

    def _split_docs_for_multi_modal(
        self, documents: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        for doc in documents:
            if "chunk_type" not in doc.metadata:
                doc.metadata["chunk_type"] = "text"

        text_docs = [doc for doc in documents if doc.metadata["chunk_type"] == "text"]
        image_docs = [doc for doc in documents if doc.metadata["chunk_type"] == "image"]
        return text_docs, image_docs

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        text_docs, image_docs = self._split_docs_for_multi_modal(documents)

        text_ids = []
        if text_docs:
            text_ids = super().add_documents(text_docs, **kwargs)

        image_ids = []
        if image_docs:
            image_ids = self._add_image_docs(image_docs, **kwargs)

        return text_ids + image_ids

    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        text_docs, image_docs = self._split_docs_for_multi_modal(documents)

        text_ids = []
        if text_docs:
            text_ids = await super().aadd_documents(text_docs, **kwargs)

        image_ids = []
        if image_docs:
            image_ids = await self._aadd_image_docs(image_docs, **kwargs)

        return text_ids + image_ids

    async def aadd_documents_with_ttl(
        self,
        documents: Iterator[Document],
        *,
        ttl_seconds: int,
        max_requests: int,
        **kwargs: Any,
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
                batch_ids = await self._process_batch(
                    [document], ttl_seconds, redis_client, **kwargs
                )
                return batch_ids

        tasks = [
            asyncio.create_task(process_document(document)) for document in documents
        ]
        results = await asyncio.gather(*tasks)
        document_ids = [doc_id for batch_ids in results for doc_id in batch_ids]

        return document_ids
    
    async def aadd_batch_with_ttl(
        self,
        documents: Iterator[Document],
        *,
        ttl_seconds: int,
        max_requests: int,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> List[str]:    
        semaphore = asyncio.Semaphore(max_requests)
        redis_client = self.config.redis()

        tasks = []
        text_batch = []

        async def flush_text_batch(batch: List[Document]) -> None:
            if not batch:
                return

            local_batch = list(batch)
            batch.clear()

            async def do_embed_text():
                async with semaphore:
                    doc_ids = await super(MultiModalVectorStore, self).aadd_documents(local_batch, **kwargs)
                    for doc_id in doc_ids:
                        await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)
                    return doc_ids

            tasks.append(asyncio.create_task(do_embed_text()))

        async def process_image_doc(doc: Document) -> None:
            async def do_embed_image():
                async with semaphore:
                    doc_ids = await self._aadd_image_docs([doc], **kwargs)
                    for doc_id in doc_ids:
                        await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)
                    return doc_ids

            tasks.append(asyncio.create_task(do_embed_image()))

        async for doc in _async_iterator_wrapper(documents):
            if doc.metadata.get("chunk_type", "text") == "text":
                text_batch.append(doc)
                if len(text_batch) >= batch_size:
                    await flush_text_batch(text_batch)
            else:
                if text_batch:
                    await flush_text_batch(text_batch)
                await process_image_doc(doc)

        if text_batch:
            await flush_text_batch(text_batch)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_doc_ids: List[str] = []
        for res in results:
            if isinstance(res, Exception):
                raise res
            else:
                all_doc_ids.extend(res)
        return all_doc_ids

    async def _process_batch(
        self, batch: List[Document], ttl_seconds: int, redis_client, **kwargs: Any
    ) -> List[str]:
        batch_ids = await self.aadd_documents(batch, **kwargs)
        for doc_id in batch_ids:
            await asyncio.to_thread(redis_client.expire, doc_id, ttl_seconds)

        return batch_ids

    def _prepare_image_records(
        self, docs: List[Document], embeddings: List[List[float]]
    ) -> List[dict]:
        records = []
        for doc, emb in zip(docs, embeddings):
            record = {
                self.config.content_field: doc.page_content,
                self.config.embedding_field: (
                    array_to_buffer(emb, dtype=self.config.vector_datatype)
                ),
            }
            for field_name, field_value in doc.metadata.items():
                if isinstance(field_value, list):
                    record[field_name] = self.config.default_tag_separator.join(
                        field_value
                    )
                else:
                    record[field_name] = field_value
            records.append(record)
        return records

    def _sync_load_records(self, records: List[dict], **kwargs) -> List[str]:
        keys = kwargs.get("keys")
        if keys:
            record_keys = [f"{self.config.key_prefix}:{key}" for key in keys]
            result = self._index.load(records, keys=record_keys, ttl=self.ttl)
        else:
            result = self._index.load(records, ttl=self.ttl)
        return list(result) if result else []

    def _add_image_docs(self, docs: List[Document], **kwargs) -> List[str]:
        image_texts = [d.page_content for d in docs]
        embeddings = self._embeddings.embed_images(image_texts)

        records = self._prepare_image_records(docs, embeddings)

        return self._sync_load_records(records, **kwargs)

    async def _aadd_image_docs(self, docs: List[Document], **kwargs) -> List[str]:
        if not docs:
            return []

        image_texts = [d.page_content for d in docs]
        embeddings = await self._embeddings.aembed_images(image_texts)

        records = self._prepare_image_records(docs, embeddings)

        loop = asyncio.get_running_loop()

        def load_sync():
            return self._sync_load_records(records, **kwargs)

        result = await loop.run_in_executor(None, load_sync)
        return result
