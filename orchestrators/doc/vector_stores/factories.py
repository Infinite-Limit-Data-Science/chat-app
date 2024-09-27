from orchestrators.doc.vector_stores.redis_vector_store import RedisVectorStore

FACTORIES = {
    'redis': RedisVectorStore,
}

__all__ = ['RedisVectorStore', 'FACTORIES']