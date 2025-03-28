import pytest
import re
import os
import uuid
import json
from uuid import UUID
import itertools
from typing import Iterator, List, Optional, Dict
from dotenv import load_dotenv
from faker import Faker
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_redis import RedisConfig
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.runnables.utils import ConfigurableField
from ..huggingface_llm import HuggingFaceLLM
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_transformer_tokenizers import (
    get_tokenizer_class_by_prefix,
    BaseLocalTokenizer,
)
from ...gwblue_vectorstores.redis.multimodal_vectorstore import MultiModalVectorStore
from .corpus import examples

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
def llama_11B_vision_instruct() -> BaseLocalTokenizer:
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    return get_tokenizer_class_by_prefix(model_name)(model_name)

@pytest.fixture
def llm(llama_11B_vision_instruct: BaseLocalTokenizer) -> HuggingFaceLLM:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    return HuggingFaceLLM(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        max_tokens=llama_11B_vision_instruct.max_new_tokens,
        temperature=0.8,
        provider=config["provider"],
        model=config["name"],
    )


@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        provider=config["provider"],
        model=config["name"],
    )

@pytest.fixture
def vlm_tokenizer() -> BaseLocalTokenizer:
    model_name = "TIGER-Lab/VLM2Vec-Full"
    return get_tokenizer_class_by_prefix(model_name)(model_name)

@pytest.fixture
def vectorstore(
    embeddings: HuggingFaceEmbeddings, 
    vlm_tokenizer: BaseLocalTokenizer
) -> Iterator[MultiModalVectorStore]:
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ["REDIS_URL"],
        metadata_schema=[
            {"name": "input", "type": "text"},
            {"name": "output", "type": "text"},
        ],
        embedding_dimensions=vlm_tokenizer.vector_dimension_length,
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
    title: str = Field(description="Title of the movie")
    release_year: int = Field(description="Year the movie was released")
    director: str = Field(description="Director of the movie")
    plot_summary: str = Field(description="Brief summary of the movie plot")

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
            raise ValueError(
                "Invalid director name. Must contain only letters and spaces."
            )
        return name


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
        llm_result = super()._generate(
            prompts, stop=stop, run_manager=run_manager, **kwargs
        )
        if not llm_result.llm_output:
            llm_result.llm_output = {}
        llm_result.llm_output["final_temp"] = self.temperature

        return llm_result


class UsageCollectorWithChainID(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.chain_run_ids: List[UUID] = []
        self.usage_by_run_id: Dict[UUID, any] = {}

    def on_chain_start(
        self,
        serialized: dict[str, any],
        inputs: dict[str, any],
        run_id: Optional[UUID] = None,
        **kwargs: any,
    ):
        if run_id is not None:
            self.chain_run_ids.append(run_id)

    def on_llm_end(
        self,
        response: LLMResult,
        run_id: Optional[UUID] = None,
        **kwargs: any,
    ):
        usage = response.llm_output.get("token_usage")
        for run_id in self.chain_run_ids:
            self.usage_by_run_id[run_id] = usage


class ConfigurableCaptureCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.captured_temp = None

    def on_llm_end(self, response: LLMResult, run_id=None, **kwargs):
        if response.llm_output is not None:
            self.captured_temp = response.llm_output.get("final_temp")


@pytest.fixture
def spy_llm(llama_11B_vision_instruct: BaseLocalTokenizer) -> SpyHuggingFaceLLM:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")

    return SpyHuggingFaceLLM(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        max_tokens=llama_11B_vision_instruct.max_new_tokens,
        temperature=0.8,
        provider=config["provider"],
        model=config["name"],
    )


def test_llm_type(llm: HuggingFaceLLM):
    assert getattr(llm, "_llm_type") == "huggingface_llm"


def test_identifying_params(llm: HuggingFaceLLM):
    assert getattr(llm, "_identifying_params") == {
        "endpoint_url": llm.base_url,
        "model_kwargs": {},
    }


def test_llm_invoke(llm: HuggingFaceLLM):
    ai_message = llm.invoke("What is Generative AI?")
    assert len(ai_message) > 0


def test_llm_invoke_with_prompt_template(llm: HuggingFaceLLM):
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = chain.invoke({"input": "Memento"})
    assert len(ai_message) > 0


def test_llm_invoke_with_output_parser(llm: HuggingFaceLLM):
    output_parser = PydanticOutputParser(pydantic_object=MovieSummary)

    prompt = PromptTemplate(
        template="Tell me about the movie {input}.\nReturn ONLY a single valid JSON object.\n{format_instructions}\n",
        input_variables=["input"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )
    print(prompt.format(input="Memento"))

    chain = prompt | llm | output_parser
    ai_message = chain.invoke({"input": "Memento"})
    assert isinstance(ai_message, MovieSummary)
    assert ai_message.release_year == 2000
    assert ai_message.director == "Christopher Nolan"
    assert len(ai_message.plot_summary) > 1


def test_llm_invoke_with_few_shot_prompt(
    llm: HuggingFaceLLM,
    vectorstore: Iterator[MultiModalVectorStore],
    sample_population: List[str],
):
    def example_to_text(
        example: dict[str, str],
    ) -> str:
        sorted_keys = sorted(example.keys())
        return " ".join(example[k] for k in sorted_keys)

    string_examples = [example_to_text(eg) for eg in examples]

    index_ids = vectorstore.add_texts(string_examples, metadatas=examples)
    print(index_ids)

    example_selector = MaxMarginalRelevanceExampleSelector(vectorstore=vectorstore, k=3)

    selector_output = example_selector.select_examples(
        {"input": "Which names have the highest salaries?"}
    )
    print(selector_output)

    # Once we have the selected examples example selector returned, then we format them for the eventual prompt sent to the model
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    salaries = itertools.count(start=10_000, step=10_000)
    ages = itertools.count(start=25, step=5)
    data = zip(sample_population, ages, salaries)
    df = pd.DataFrame(data, columns=["Name", "Age", "Salary"])

    prefix = f"""
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df.head()}

    IMPORTANT: Return EXACTLY a Python expression referencing df, with NO extra text or explanation. 
    You must NOT provide any additional commentary or lines.

    In the following examples, any string wrapped in angle brackets (e.g., <col_name>, <col1>, <value>, <n>, <start_date>, etc.) is a placeholder. It represents a variable that can change depending on the user's actual DataFrame or query.
    
    Give the pandas expression of every input.
    """
    mmr_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix="Input: {input}\nOutput:",
        input_variables=["input"],
    )

    mmr_prompt_output = mmr_prompt.format(
        input="Which names have the highest salaries?"
    )
    print(mmr_prompt_output)

    # another option is to add a PandasOutputParser to determine if response is valid pandas
    # expression, for example, a @field_validator on an expression field which loads
    # the returned string into eval to validate its valid pandas expression. If not,
    # then OutputParser will generate error and in a langgraph we can send the prompt
    # back to the language model with additional context to correct the mistake.
    chain = mmr_prompt | llm
    ai_message = chain.invoke({"input": "Which names have the highest salaries?"})

    df_content = eval(ai_message, {"df": df})

    assert len(df_content) > 0


def test_llm_invoke_with_callbacks(llm: HuggingFaceLLM):
    mock_handler = MockCallbackHandler()
    config = RunnableConfig(callbacks=[mock_handler])
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    chain.invoke({"input": "Memento"}, config=config)

    assert mock_handler.llm_end_data is not None


def test_llm_invoke_with_run_information(spy_llm: SpyHuggingFaceLLM):
    llm = spy_llm.configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )

    handler = ConfigurableCaptureCallbackHandler()

    run_uuid = uuid.uuid4()
    config = RunnableConfig(
        tags=["huggingface_llm", "llm_123"],
        metadata={"user_uuid": "12345", "model": "meta/llama-3.2-90b-vision-instruct"},
        run_name="huggingface_llm_invoke_role",
        max_concurrency=1,  # more applicable when batching or using composite chains
        configurable={
            "temperature": 0.3
        },  # more applicable when you have multiple models in chain and want some to have configurable attribute like temperature but not others
        run_id=run_uuid,
        callbacks=[handler],
    )
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    chain.invoke({"input": "Memento"}, config=config)

    assert (
        handler.captured_temp == 0.3
    ), f"Expected temperature to be 0.3, got {spy_llm.last_used_temperature}"


def test_llm_invoke_with_token_usage_in_response(llm: HuggingFaceLLM):
    usage_collector = UsageCollectorWithChainID()
    run_uuid = uuid.uuid4()
    config = RunnableConfig(
        run_id=run_uuid, tags=["huggingface_llm_role"], callbacks=[usage_collector]
    )

    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = chain.invoke({"input": "Memento"}, config=config)

    usage_info = usage_collector.usage_by_run_id.get(run_uuid)

    structured_output = {"answer": ai_message, "token_usage": usage_info}

    assert all(structured_output["token_usage"].values())


@pytest.mark.skip(
    reason="LLM model only supports text inputs, image-to-text is not implemented."
)
def test_llm_invoke_with_image_to_text():
    """
    `HuggingFaceLLM` only supports text input.

    For multimodal and tool calling, use the `HuggingFaceChatModel` class of the huggingface_inference_kit package.
    """
    pass


@pytest.mark.skip(
    reason="LLM model only supports text inputs, tool calling is not implemented."
)
def test_llm_invoke_with_tool_calling():
    """
    `HuggingFaceLLM` only supports text input.

    For multimodal and tool calling, use the `HuggingFaceChatModel` class of the huggingface_inference_kit package.
    """
    pass


@pytest.mark.asyncio
async def test_llm_ainvoke(llm: HuggingFaceLLM):
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = await chain.ainvoke({"input": "Memento"})
    assert len(ai_message) > 0


def test_llm_stream(llm: HuggingFaceLLM):
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm

    ai_message = ""
    for chunk in chain.stream({"input": "Memento"}):
        ai_message += chunk

    assert len(ai_message) > 0


@pytest.mark.asyncio
async def test_llm_astream(llm: HuggingFaceLLM):
    prompt = PromptTemplate(
        input_variables=["input"], template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = ""
    async for chunk in chain.astream({"input": "Memento"}):
        ai_message += chunk

    assert len(ai_message) > 0


def test_llm_batch(llm: HuggingFaceLLM):
    """
    Batching support right now is basic.

    More features coming soon
    """
    ai_messages = llm.batch(
        ["Tell me about the movie Memento", "Tell me about the movie Reservoir Dogs"]
    )
    assert len(ai_messages) > 0


@pytest.mark.asyncio
async def test_llm_abatch(llm: HuggingFaceLLM):
    """
    Batching support right now is basic.

    More features coming soon
    """
    ai_messages = await llm.abatch(
        ["Tell me about the movie Memento", "Tell me about the movie Reservoir Dogs"]
    )
    assert len(ai_messages) > 0
