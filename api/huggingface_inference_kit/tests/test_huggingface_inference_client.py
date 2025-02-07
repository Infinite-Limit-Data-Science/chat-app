import pytest
from ..huggingface_inference_client import HuggingFaceInferenceClient
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer

# TODO: start with cleaning up this and then add the async methods and then the example selectors in the dataframe expression tool tests
attributes = {
    'name': 'BAAI/bge-large-en-v1.5',
    'endpoint': {'url':'http://100.28.34.190:8070/', 'type':'tei'},
    'max_batch_tokens': 32768,
    'max_client_batch_size': 128,
    'max_batch_requests': 64,
    'num_workers': 8,
    'auto_truncate': True,
    'token': 'eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOiJzdmMtY2hhdC10ZXN0Iiwic3ViIjoibjFtNCIsIm1haWwiOiJqb2huLmRvZUBiY2JzZmwuY29tIiwic3JjIjoiam9obi5kb2VAYmNic2ZsLmNvbSIsInJvbGVzIjpbIiJdLCJpc3MiOiJQTUktVGVzdCIsImF0dHJpYnV0ZXMiOlt7ImdpdmVubmFtZSI6IkpvaG4ifSx7InNuIjoiSkRvZSJ9LHsibWFpbCI6ImpvaG4uZG9lQGJjYnNmbC5jb20ifSx7ImRpc3BsYXluYW1lIjoiSkRvZSwgSm9obiJ9LHsiYmNic2ZsLWlkbVBpY3R1cmVVUkwiOiIifV0sImF1ZCI6ImNoYXRhcHAtdHN0YS50aHJvdGwuY29tIiwiZ2l2ZW5uYW1lIjoiSm9obiIsImRpc3BsYXluYW1lIjoiRG9lLCBKb2huIiwic24iOiJKRG9lIiwiaWRtX3BpY3R1cmVfdXJsIjoiIiwiZXhwIjoxODkzNDU2MDAwLCJpYXQiOjE3MTQxNDQ4NDEsInNlc3Npb25faWQiOiIiLCJqdGkiOiIifQ.rxHyA_WeMprlMtDsTGPvqgjRbQ2qT7VkiT6Ak1aSQmTl3nOFR_v0ev2AmUogUHXJi9CmGZcw3i-Wsis86ggOJKl4e7TwuKSBqt-s81jzGePI2yIsyKInEXwieKHXpWl1JFMtSkDpkRBeaiSlM1qpJ33BJLekRRkW-mDhV-yG5VVxyOWxRZDSfXRgrQ3CoNzChvITqdC1VOCeMAMI5Vg5zvo9bNOjOqOCLEtncsHdDiD7gYmPsGWeR9eXcT0y2-KONa0LvsYBewBcXjvJE63xe3XViiQ3HQPayjA1UAxWekD83_Kq7y-LJEjrQNNphEq_XyocpzvlmK-tlf59UGJJcw'
}

@pytest.fixture
def inference_client() -> HuggingFaceInferenceClient:
    return HuggingFaceInferenceClient(
        base_url=attributes['endpoint']['url'],
        credentials=attributes['token']
    )

@pytest.fixture
def bge() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def corpus() -> str:
    text = """
    The quick brown fox jumps over the lazy dog near the riverbank. 
    As the sun set, the sky turned shades of orange and pink, casting a golden glow on the water. 
    Birds chirped melodiously in the background while a gentle breeze rustled the leaves.
    """
    return text

def test_inference_client_feature_extraction(inference_client: HuggingFaceInferenceClient, corpus: str):
    """
    Invokes the feature-extraction task of the HuggingFace TEI. Note
    this expects the TEI is running in a self-hosted environment.
    A single text or a list of texts are accepted for embedding.
    Internally calls InferenceClient of langchain_huggingface,
    which uses the requests package to send a post request to TEI.

    Note requests package has a json= parameter which will be sent 
    as part of post request to TEI. The format of the json field changes  
    whether using a single text or list of texts:

    single text: '{"inputs":"What is Deep Learning?"}'
    list of texts: '{"inputs":["Today is a nice day", "I like you"]}'
    
    The json= parameter takes a dictionary and it internally gets converted 
    to an actual json string

    TEI returns a byte string of high-dimensional vectors from the requests 
    package post request.
    Example: b'[[0.006687683,-0.02827644,0.021200065,-0.006164584, ... ]'

    Curl examples:

    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":["Today is a nice day", "I like you"]}' \
        -H 'Content-Type: application/json'    

    Note the InferenceClient doesn't append /embed but rather specifies
    the task 'feature_extraction'
        
    feature_extract converts the bytes to a numpy array of dtype float32. 
    These floats are representations of dense numerical vectors of the 
    original input text. If there were six inputs of text, then there will 
    be 6 arrays of floats returned. Each array will be of size 1024, if 
    the dimensions of the model are 1024, such as in BGE-Large. In effect,
    each array is a vector.

    In the case of BGE-Large and most modern embedding models, the entire 
    query text is converted into a single vector representation rather than 
    breaking it into individual word vectors like word2vec or TF-IDF.

    Unlike traditional word embeddings (like word2vec), BGE-Large and 
    transformer-based models use contextual embeddings where the meaning 
    of a word depends on the surrounding words. The model processes the 
    entire query as a sequence and outputs a single vector representing 
    the meaning of the whole query.
    """
    embeddings = inference_client.feature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_feature_extraction_trunc(inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    corpus = " ".join([corpus] * 10)
    embeddings = inference_client.feature_extraction(corpus, truncate=True)

    assert embeddings.size == bge.dimensions

def test_inference_client_feature_extraction_not_tokens(inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    """
    The output of feature_extraction() is not tokens of text, 
    but rather dense numerical vector representations of the 
    input text.
    
    feature_extraction returns a high-dimensional vector 
    (1024 features for BGE-Large).
    
    The tokenizer converts text into token IDs. The token ids 
    are indices in the tokenizer's vocabulary. But that's not 
    what is returned here. 
    
    Hence, the vectors do not correspond to token IDs

    As such, you cannot directly do anything with the vectors. 
    You cannot convert them back into tokens or text. You can, 
    however, embed a query and then use the vectors of that 
    query to find semantically similar vectors from the original 
    text.
    """
    tokens  = bge.tokenizer.encode(corpus, add_special_tokens=True)
    embeddings = inference_client.feature_extraction(corpus, truncate=True)
    decoded = bge.tokenizer.decode(embeddings[0])

    assert tokens != decoded