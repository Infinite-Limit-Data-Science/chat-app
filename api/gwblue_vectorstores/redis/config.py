from pydantic import BaseModel

_INDEX_NAME = "user_conversations"

_DISTANCE_METRIC = "COSINE"

_INDEXING_ALGORITHM = "FLAT"

_VECTOR_DATATYPE = "FLOAT32"

_STORAGE_TYPE = "hash"

_CONTENT_FIELD_NAME = "text"

_EMBEDDING_VECTOR_FIELD_NAME = "embedding"


class VectorStoreSchema(BaseModel):
    """
    Example supported index:
    {
        "index": {
            "name": "user_conversations",
            "prefix": "doc:",  # all keys start with doc:
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "content",
                "type": "text",
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 1536,
                    "distance_metric": "COSINE",
                    "algorithm": "FLAT",
                    "datatype": "FLOAT32",
                },
            },
            {
                "name": "source",
                "type": "tag",
            },
            {
                "name": "author",
                "type": "tag",
            },
        ],
    }
    """

    index_name: str = _INDEX_NAME
    distance_metric: str = _DISTANCE_METRIC
    indexing_algorithm: str = _INDEXING_ALGORITHM
    vector_datatype: str = _VECTOR_DATATYPE
    storage_type: str = _STORAGE_TYPE
    content_field: str = _CONTENT_FIELD_NAME
    embedding_field: str = _EMBEDDING_VECTOR_FIELD_NAME
