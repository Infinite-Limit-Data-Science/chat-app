import pytest
import os
from dotenv import load_dotenv
from ..huggingface_inference_server_config import (
    HuggingFaceInferenceConfig,
    HuggingFaceEmbeddingsConfig,
)

load_dotenv()

@pytest.fixture
def tgi_self_hosted_config() -> HuggingFaceInferenceConfig:
    return HuggingFaceInferenceConfig(
        name='meta-llama/Meta-Llama-3.1-70B-Instruct',
        url=os.environ['TEST_TGI_URL'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_input_tokens=12582,
        max_total_tokens=16777,
        max_batch_prefill_tokens=12582+50,
        payload_limit=5_000_000
    )

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceEmbeddingsConfig:
    return HuggingFaceEmbeddingsConfig(
        name='BAAI/bge-large-en-v1.5',
        url=os.environ['TEST_TEI_URL'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

def test_tgi_self_hosted_config(tgi_self_hosted_config: HuggingFaceInferenceConfig):
    assert tgi_self_hosted_config.provider == 'hf-inference'

def test_tei_self_hosted_config(tei_self_hosted_config: HuggingFaceEmbeddingsConfig):
    assert tei_self_hosted_config.provider == 'hf-inference'