import pytest
import os
import json
from dotenv import load_dotenv
from ..huggingface_inference_server_config import (
    HuggingFaceInferenceConfig,
    HuggingFaceEmbeddingsConfig,
)

load_dotenv()


def _model_config(model_type: str, model_name: str) -> str:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        "name": model["name"],
        "url": model["endpoints"][0]["url"],
        "provider": model["endpoints"][0]["provider"],
    }

@pytest.fixture
def inference_config() -> HuggingFaceInferenceConfig:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    return HuggingFaceInferenceConfig(
        model=config["name"],
        url=config["url"],
        provider=config["provider"],
        auth_token=os.environ["TEST_AUTH_TOKEN"],
        max_input_tokens=36286,
        max_total_tokens=38334,
        max_batch_prefill_tokens=36286 + 50,
    )

@pytest.fixture
def embeddings_config() -> HuggingFaceEmbeddingsConfig:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddingsConfig(
        model=config["name"],
        url=config["url"],
        provider=config["provider"],
        auth_token=os.environ["TEST_AUTH_TOKEN"],
        max_batch_tokens=8096,
    )


def test_tgi_self_hosted_config(inference_config: HuggingFaceInferenceConfig):
    assert inference_config.provider == "hf-inference"


def test_tei_self_hosted_config(embeddings_config: HuggingFaceEmbeddingsConfig):
    assert embeddings_config.provider == "vllm"
