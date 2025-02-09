import pytest
import os
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore
from redisvl.query.filter import Tag
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer 
from ...langchain_chunkinator import Chunkinator
from ..huggingface_inference_server_config import HuggingFaceTEIConfig
from .corpus import dummy_corpus1

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceTEIConfig:
    return HuggingFaceTEIConfig(
        name='BAAI/bge-large-en-v1.5',
        url='http://100.28.34.190:8070/',
        auth_token='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOiJzdmMtY2hhdC10ZXN0Iiwic3ViIjoibjFtNCIsIm1haWwiOiJqb2huLmRvZUBiY2JzZmwuY29tIiwic3JjIjoiam9obi5kb2VAYmNic2ZsLmNvbSIsInJvbGVzIjpbIiJdLCJpc3MiOiJQTUktVGVzdCIsImF0dHJpYnV0ZXMiOlt7ImdpdmVubmFtZSI6IkpvaG4ifSx7InNuIjoiSkRvZSJ9LHsibWFpbCI6ImpvaG4uZG9lQGJjYnNmbC5jb20ifSx7ImRpc3BsYXluYW1lIjoiSkRvZSwgSm9obiJ9LHsiYmNic2ZsLWlkbVBpY3R1cmVVUkwiOiIifV0sImF1ZCI6ImNoYXRhcHAtdHN0YS50aHJvdGwuY29tIiwiZ2l2ZW5uYW1lIjoiSm9obiIsImRpc3BsYXluYW1lIjoiRG9lLCBKb2huIiwic24iOiJKRG9lIiwiaWRtX3BpY3R1cmVfdXJsIjoiIiwiZXhwIjoxODkzNDU2MDAwLCJpYXQiOjE3MTQxNDQ4NDEsInNlc3Npb25faWQiOiIiLCJqdGkiOiIifQ.rxHyA_WeMprlMtDsTGPvqgjRbQ2qT7VkiT6Ak1aSQmTl3nOFR_v0ev2AmUogUHXJi9CmGZcw3i-Wsis86ggOJKl4e7TwuKSBqt-s81jzGePI2yIsyKInEXwieKHXpWl1JFMtSkDpkRBeaiSlM1qpJ33BJLekRRkW-mDhV-yG5VVxyOWxRZDSfXRgrQ3CoNzChvITqdC1VOCeMAMI5Vg5zvo9bNOjOqOCLEtncsHdDiD7gYmPsGWeR9eXcT0y2-KONa0LvsYBewBcXjvJE63xe3XViiQ3HQPayjA1UAxWekD83_Kq7y-LJEjrQNNphEq_XyocpzvlmK-tlf59UGJJcw',        
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

@pytest.fixture
def chunks(tei_self_hosted_config: HuggingFaceTEIConfig) -> List[str]:
    documents = [Document(page_content=dummy_corpus1, metadata={'source': 'book'})]
    chunkinator = Chunkinator.Base(documents, tei_self_hosted_config)
    return chunkinator.chunk() 

@pytest.fixture
def embeddings(tei_self_hosted_config: HuggingFaceTEIConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        base_url=tei_self_hosted_config.url,
        credentials=tei_self_hosted_config.auth_token
    )

@pytest.fixture
def tokenizer() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    tokenizer: BgeLargePretrainedTokenizer
) -> Iterator[RedisVectorStore]:
    # TODO: need to create a cleaner wrapper class for redis vectorstore
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ['REDIS_URL'],
        metadata_schema=[
            {"name": "source", "type": "tag"},
        ],
        # setting this avoids unnecessary request for embeddings
        embedding_dimensions=tokenizer.dimensions
    )

    store = RedisVectorStore(embeddings, config=config)

    yield store

    store.index.clear()
    store.index.delete(drop=True)

# @pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_embed_documents(embeddings: HuggingFaceEmbeddings):
    """
    Embeds text as high-dimensional vectors using the lower level 
    `HuggingFaceInferenceClient` class 
    
    To implement the langchain_core Embeddings abstract class,
    `embed_documents` take the returned numpy float32 arrays
    and converts them to list[list[float]]

    list[float] represents a vector and list[list[float]] 
    represents a list of vectors (in the case of multiple
    text sequences passed in as an array of texts). Each of
    those text sequences will be its own vector.

    Example:
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":["Today is a nice day", "I like you"]}' \
        -H 'Content-Type: application/json'    

    The first request will return a single vector. The second
    will return 2 vectors.    
    """
    embedded_vectors = embeddings.embed_documents([dummy_corpus1])
    assert len(embedded_vectors) > 0

def test_embed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = embeddings.embed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

def test_embed_query(embeddings: HuggingFaceEmbeddings):
    """
    Takes a single query text. Generates the embedding.
    Since it was a single query text, and since a single
    vector was generated, this returns a single vector in 
    a list
    """
    embedded_vectors = embeddings.embed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

def test_embed_documents_in_vector_db(vectorstore: RedisVectorStore):
    """
    To search over unstructured data embedded in vector database,
    the vectors must first be stored in the vector database. We
    use `add_texts` to store the vectors in vector store that were 
    generated by the embeddings model
    
    The returned IDs are just document references in Redis, 
    which combine the index name with a unique key string. The 
    ids are not high-dimensional float32 vectors or tokens part 
    of a model's vocabulary. They are simply internal Redis 
    identifiers
    """
    ids = vectorstore.add_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

def test_embed_documents_with_similarity_search(vectorstore: RedisVectorStore, chunks: List[str]):
    """
    Use vector store to store embeddings in a database. This internally 
    invokes the embeddings class to generate the embeddings and, thus,
    goes out to the TEI. The returned vectors are stored in a Redis
    field which defaults to the name "embedding". The embeddings, the
    original text, and metadata are all stored together in a Redis hash
    type, which holds the following structure:

    "index_name"
    "type", "HASH",
    "fields", [
        "text", "TEXT",
        "embedding", "VECTOR HNSW 1024 FLOAT32",
        "metadata_field", "TAG"
    ]

    The Redis storage type of the vector field is either JSON or hash.
    The data type of the vector is FLOAT32.

    The vector store interface provides methods so you don't have to 
    directly query the vector store like this:
    > FT._LIST
    > FT.SEARCH test1 "*" LIMIT 0 10
    > FT.SEARCH test1 "*" LIMIT 0 0
    > FT.DROPINDEX test1 DD

    The similarity search searches for vectors similar to the vectorized
    query. Therefore, returning relevant textual documents that correspond
    to the similar vectors (hence why redis vector store stores both text
    and embedding as part of the same document hash).

    Specifically, Redis supports the distance_metric algorthims of Cosine 
    Similarity and Euclidean Distance for actual mathematical vector
    searches. 

    However, there is also the indexing algorithm used to enhance how nearest 
    neighbors are determined in the entire vector space. Nearest Neighbor Search 
    is the process of finding the closest data points (e.g., vectors) in a high-
    dimensional space before the similarity (Cosine Similarity) or distance 
    metric (Euclidean Distance) are performed to help improve efficiency and 
    performance.

    Exact Nearest Neighbor (NN): Searches the entire dataset exhaustively to 
    find the exact closest matches. Approximate Nearest Neighbor (ANN): Introduces 
    optimizations or approximations to reduce the computational complexity while 
    still finding close (but not always exact) matches.

    There are three common indexing algorithms to perform Nearest Neighbor 
    Searches (both NN and ANN): Flat Index, Hierarchical Navigable Small World 
    Graphs (HNSW), and Product Quantization (PQ). The default being used here
    is Flat Index.
    """
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = vectorstore.similarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score(vectorstore: RedisVectorStore, chunks: List[str]):
    """
    A vector score (or similarity score) measures how close a query embedding 
    is to stored embeddings in the vector database. The score depends on the 
    similarity metric used, such as:
    - Cosine Similarity (higher = more similar)
    - Cosine Distance (lower = more similar)
    - Euclidean Distance (lower = more similar)
    - Dot Product Similarity (higher = more similar)

    In Redis, the score is normalized, meaning its scale depends on the chosen 
    metric.

    Cosine Similarity: Measures the cosine of the angle between vectors, indicating 
    their similarity. Higher values mean greater similarity. Cosine Distance: 
    Measures the dissimilarity between vectors as the complement of the cosine 
    similarity. Higher values mean greater dissimilarity.

    In RedisVectorStore, the distance_metric parameter supports "COSINE", but it 
    does not explicitly distinguish between cosine similarity and cosine distance 
    in its documentation. Specifying COSINE as a distance metric implicitly means 
    cosine distance, not cosine similarity. Redis uses cosine distance 
    (1 - cosine similarity), where lower values indicate greater similarity.

    The computed similarity score in Redis is captured in the "vector_distance" 
    field.

    If the with_vectors=True is set, the search also retrieves and returns the 
    full vector embeddings.
    """    
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.similarity_search_with_score(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    score1 = results[0][1]
    score2 = results[1][1]

    assert score1 < score2

def test_embed_documents_with_max_marginal_relevance_search(vectorstore: RedisVectorStore, chunks: List[str]):
    """
    Maximal Marginal Relevance (MMR) is an optimization method used in vector 
    search and document retrieval to balance relevance and diversity.

    When searching for relevant documents in a vector database, a standard similarity 
    search retrieves the most similar documents to the query. However, sometimes 
    these documents are too similar to each other, leading to redundancy.

    MMR solves this problem by selecting documents that are:
    - Similar to the query (high relevance)
    - Diverse from each other (avoid redundancy)

    At each step, MMR picks the next document that maximizes the following score:
    MMR Score=λ × Similarity to Query - (1 - λ) × max(Similarity to Selected Docs)
    """
    vectorstore.add_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"    
    results = vectorstore.max_marginal_relevance_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

def test_embed_documents_with_similarity_search_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)
    
    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.similarity_search_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2

def test_embed_documents_with_similarity_search_with_score_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)

    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.similarity_search_with_score_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    score1 = results[0][1]
    score2 = results[1][1]

    assert score1 < score2

def test_embed_documents_with_max_marginal_relevance_search_by_vector(embeddings: HuggingFaceEmbeddings, vectorstore: RedisVectorStore, chunks: List[str]):
    vectorstore.add_documents(chunks)

    query = "What did King Ulfric Stormborn do in 879"
    float_32_1024_dimensional_bge_vector = embeddings.embed_documents([query])[0]
    results = vectorstore.max_marginal_relevance_search_by_vector(
        float_32_1024_dimensional_bge_vector, 
        k=2, 
        filter=Tag('source') == 'book'
    )

    assert len(results) == 2

@pytest.mark.asyncio
async def test_aembed_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_documents([dummy_corpus1])
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_multiple_documents(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_documents([dummy_corpus1, dummy_corpus1.upper()])
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_query(embeddings: HuggingFaceEmbeddings):
    embedded_vectors = await embeddings.aembed_query(dummy_corpus1)
    assert len(embedded_vectors) > 0

@pytest.mark.asyncio
async def test_aembed_documents_in_vector_db(vectorstore: RedisVectorStore):
    ids = await vectorstore.aadd_texts([dummy_corpus1], [{'source': 'book'}])
    assert ids[0].startswith('test1')

@pytest.mark.asyncio
async def test_aembed_documents_with_similarity_search(vectorstore: RedisVectorStore, chunks: List[str]):
    await vectorstore.aadd_documents(chunks)
    query = "What did King Ulfric Stormborn do in 879"
    results = await vectorstore.asimilarity_search(
        query, 
        k=2, 
        filter=Tag('source') == 'book'
    )
    assert len(results) == 2