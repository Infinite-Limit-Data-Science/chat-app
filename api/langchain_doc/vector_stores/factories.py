from .redis_vector_proxy import create_redis_vector_proxy
from .redis_vector_retriever import RedisVectorRetriever

STORE_FACTORIES = {
    'redis': create_redis_vector_proxy,
}

RETRIEVER_FACTORIES = {
    'redis': RedisVectorRetriever,
}