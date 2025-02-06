import pytest
from ..huggingface_inference_client import HuggingFaceInferenceClient
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer

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
    documents = """
    The quick brown fox jumps over the lazy dog near the riverbank. 
    As the sun set, the sky turned shades of orange and pink, casting a golden glow on the water. 
    Birds chirped melodiously in the background while a gentle breeze rustled the leaves.
    """
    return documents

def test_inference_client_feature_extraction(inference_client: HuggingFaceInferenceClient, corpus: str):
    embeddings = inference_client.feature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_feature_extraction_trunc(inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    """
    feature_extract returns dense numerical vector representations of the input text
    Depending on the model used, each vector will have a specified number of dimensions
    For example, each vector of bge-large will have 1024 features

    The return value is a numpy array which contains the 1024 features 
    of the first vector of the embeddings
    """
    corpus = " ".join([corpus] * 10)
    embeddings = inference_client.feature_extraction(corpus, truncate=True)

    assert embeddings.size == bge.dimensions

def test_inference_client_feature_extraction_not_tokens(inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    """
    The output of feature_extraction() is not tokenized text, 
    but rather dense numerical vector representations of the input text.
    feature_extraction returns a high-dimensional vector (1024 features for BGE-Large).
    
    The tokenizer converts text into token IDs.
    These numbers are indices in the tokenizer's vocabulary. 
    
    Hence, the vectors do not correspond to token IDs
    """
    tokens  = bge.tokenizer.encode(corpus, add_special_tokens=True)
    embeddings = inference_client.feature_extraction(corpus, truncate=True)
    decoded = bge.tokenizer.decode(embeddings[0])

    assert tokens != decoded