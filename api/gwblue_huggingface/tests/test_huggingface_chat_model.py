import pytest
import os
import re
import json
import uuid
import base64
from pathlib import Path
import itertools
from typing import Iterator, List
from faker import Faker
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate
)
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.runnables import RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field, field_validator, ConfigDict
from ..huggingface_llm import HuggingFaceLLM
from ..huggingface_chat_model import HuggingFaceChatModel
from ..huggingface_embeddings import HuggingFaceEmbeddings
from .corpus import examples
from .tools import PandasExpressionTool, PandasExpressionInput
from ..huggingface_transformer_tokenizers import (
    VLM2VecFullPretrainedTokenizer
)
from ...gwblue_vectorstores.redis.multimodal_vectorstore import (
    MultiModalVectorStore
)

load_dotenv()

_MAX_INPUT_TOKENS = 12582

_MAX_TOTAL_TOKENS = 16777

def _model_config(model_type: str, model_name: str) -> str:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        'name': model['name'],
        'url': model['endpoints'][0]['url'],
        'provider': model['endpoints'][0]['provider'],
    }

@pytest.fixture
def llm() -> HuggingFaceLLM:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    return HuggingFaceLLM(
        base_url=config['url'],
        credentials=os.environ['TEST_AUTH_TOKEN'],
        max_tokens=_MAX_TOTAL_TOKENS-_MAX_INPUT_TOKENS,
        temperature=0.8,
        provider=config['provider'],
        model=config['name'],
    )

@pytest.fixture
def chat_model(llm: HuggingFaceLLM) -> HuggingFaceChatModel:
    return HuggingFaceChatModel(llm=llm)

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config['url'],
        credentials=os.environ['TEST_AUTH_TOKEN'],
        provider=config['provider'],
        model=config['name'],
    )

@pytest.fixture
def vlm_tokenizer() -> VLM2VecFullPretrainedTokenizer:
    return VLM2VecFullPretrainedTokenizer()

@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    vlm_tokenizer: VLM2VecFullPretrainedTokenizer
) -> Iterator[MultiModalVectorStore]:
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ['REDIS_URL'],
        metadata_schema=[
            {"name": "input", "type": "text"},
            {"name": "output", "type": "text"},
        ],
        embedding_dimensions=vlm_tokenizer.dimensions
    )

    store = MultiModalVectorStore(embeddings, config=config)

    yield store

    store.index.clear()
    store.index.delete(drop=True)


@pytest.fixture
def sample_population() -> List[str]:
    fake = Faker("en_GB")
    return [fake.name() for _ in range(100)]

class MovieSummary(BaseModel):
    title: str = Field(description='Title of the movie')
    release_year: int = Field(description='Year the movie was released')
    director: str = Field(description='Director of the movie')
    plot_summary: str = Field(description='Brief summary of the movie plot')

    @field_validator("release_year")
    @classmethod
    def validate_release_year(cls, year: int) -> int:
        if year < 1888 or year > 2100:
            raise ValueError("Invalid release year. Must be between 1888 and 2100.")
        return year

    @field_validator("director")
    @classmethod
    def validate_director_name(cls, name: str) -> str:
        if not re.match(r"^[a-zA-Z\s.]+$", name):
            raise ValueError("Invalid director name. Must contain only letters and spaces.")
        return name

class OutputMessage(BaseModel):
    summary: str = Field(description='A short summary of the content')
    content: str = Field(description='The full unmodified text from the model')

class MockCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.llm_end_called = False
        self.llm_end_data = None

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.llm_end_called = True
        self.llm_end_data = response

class SpyHuggingFaceLLM(HuggingFaceLLM):
    last_used_temperature: float | None = Field(None, exclude=True)
    
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        llm_result = super()._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)
        if not llm_result.llm_output:
            llm_result.llm_output = {}
        llm_result.llm_output['final_temp'] = self.temperature

        return llm_result

@pytest.fixture
def spy_llm() -> SpyHuggingFaceLLM:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    return SpyHuggingFaceLLM(
        base_url=config['url'],
        credentials=os.environ['TEST_AUTH_TOKEN'],
        max_tokens=_MAX_TOTAL_TOKENS-_MAX_INPUT_TOKENS,
        temperature=0.8,
        provider=config['provider'],
        model=config['name'],
    )

@pytest.fixture
def spy_chat_model(spy_llm: SpyHuggingFaceLLM) -> HuggingFaceChatModel:
    return HuggingFaceChatModel(llm=spy_llm)

class ConfigurableCaptureCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.captured_temp = None

    def on_llm_end(self, response: LLMResult, run_id=None, **kwargs):
        if response.llm_output is not None:
            self.captured_temp = response.llm_output.get('final_temp')

def test_chat_model_invoke_with_image_to_text(chat_model: HuggingFaceChatModel):
    image_path = Path(__file__).parent / 'assets' / 'baby.jpg'
    with image_path.open('rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    image_url = f'data:image/jpeg;base64,{base64_image}'

    chat_prompt = ChatPromptTemplate.from_messages([
        ('system', "You're a helpful assistant who can create text from images"),
        ('human', 
            [
                {'image_url': {'url': "{image_url}"}},
                'Describe this image.'
            ]
        )
    ])

    formatted_messages = chat_prompt.format_messages(image_url=image_url)
    print(formatted_messages[0].content)

    chain = chat_prompt | chat_model
    ai_message = chain.invoke({'image_url': image_url})
    assert len(ai_message.content) > 0