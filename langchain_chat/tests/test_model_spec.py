import pytest
import json
from pathlib import Path
from orchestrators.chat.llm_models.model_spec import ModelSpec

@pytest.fixture(autouse=True, scope='session')
def setup_teardown():
    yield

@pytest.fixture(name='llama_config')
def llama_3_1_70b_inst_model_config() -> str:
    file_path = Path(__file__).parent / 'llm_models' / 'local_tokenizer' / 'Meta-Llama-3.1-70B-Instruct' / 'config.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

@pytest.fixture
def model_spec(llama_config: str) -> ModelSpec:
    model_spec = ModelSpec.model_validate_json(llama_config)
    return model_spec

@pytest.fixture
def available_gpu_memory() -> int:
    available_mem_per_gpu = 24 - 7
    total_available_mem = 4 * available_mem_per_gpu
    return total_available_mem

@pytest.mark.slow
@pytest.mark.parametrize('available_mem, expected_max', [(5.5, 1000)])
def test_max_batch_total_tokens(
    model_spec: ModelSpec, 
    available_mem: float, 
    expected_max: int
):
    model_spec.available_mem = available_mem
    assert model_spec.calculate_max_batch_total_tokens() == expected_max