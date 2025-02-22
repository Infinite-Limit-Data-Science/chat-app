import pytest
import os
from dotenv import load_dotenv
from ..huggingface_inference_server_config import HuggingFaceTGIConfig, HuggingFaceTEIConfig

load_dotenv()

@pytest.fixture
def tgi_self_hosted_config() -> HuggingFaceTGIConfig:
    return HuggingFaceTGIConfig(
        name='meta-llama/Meta-Llama-3.1-70B-Instruct',
        url=os.environ['TEST_TGI_URL'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_input_tokens=12582,
        max_total_tokens=16777,
        max_batch_prefill_tokens=12582+50,
        payload_limit=5_000_000
    )

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceTEIConfig:
    return HuggingFaceTEIConfig(
        name='BAAI/bge-large-en-v1.5',
        url=os.environ['TEST_TEI_URL'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

def test_tgi_self_hosted_config(tgi_self_hosted_config: HuggingFaceTGIConfig):
    assert tgi_self_hosted_config.endpoint_type == 'tgi'

def test_tei_self_hosted_config(tei_self_hosted_config: HuggingFaceTEIConfig):
    assert tei_self_hosted_config.endpoint_type == 'tei'