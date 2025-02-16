import pytest
import re
import os
import uuid
from uuid import UUID
import itertools
from typing import Iterator, List, Optional, Dict
from faker import Faker
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_redis import RedisConfig
from langchain_redis import RedisVectorStore
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.runnables.utils import ConfigurableField
from ..huggingface_inference_server_config import HuggingFaceTGIConfig
from ..huggingface_llm import HuggingFaceLLM
from ..huggingface_embeddings import HuggingFaceEmbeddings
from ..huggingface_inference_server_config import HuggingFaceTEIConfig
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer 
from .corpus import examples

@pytest.fixture
def tgi_self_hosted_config() -> HuggingFaceTGIConfig:
    return HuggingFaceTGIConfig(
        name='meta-llama/Meta-Llama-3.1-70B-Instruct',
        url='http://3.210.60.7:8080/',
        auth_token='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHAiOiJzdmMtY2hhdC10ZXN0Iiwic3ViIjoibjFtNCIsIm1haWwiOiJqb2huLmRvZUBiY2JzZmwuY29tIiwic3JjIjoiam9obi5kb2VAYmNic2ZsLmNvbSIsInJvbGVzIjpbIiJdLCJpc3MiOiJQTUktVGVzdCIsImF0dHJpYnV0ZXMiOlt7ImdpdmVubmFtZSI6IkpvaG4ifSx7InNuIjoiSkRvZSJ9LHsibWFpbCI6ImpvaG4uZG9lQGJjYnNmbC5jb20ifSx7ImRpc3BsYXluYW1lIjoiSkRvZSwgSm9obiJ9LHsiYmNic2ZsLWlkbVBpY3R1cmVVUkwiOiIifV0sImF1ZCI6ImNoYXRhcHAtdHN0YS50aHJvdGwuY29tIiwiZ2l2ZW5uYW1lIjoiSm9obiIsImRpc3BsYXluYW1lIjoiRG9lLCBKb2huIiwic24iOiJKRG9lIiwiaWRtX3BpY3R1cmVfdXJsIjoiIiwiZXhwIjoxODkzNDU2MDAwLCJpYXQiOjE3MTQxNDQ4NDEsInNlc3Npb25faWQiOiIiLCJqdGkiOiIifQ.rxHyA_WeMprlMtDsTGPvqgjRbQ2qT7VkiT6Ak1aSQmTl3nOFR_v0ev2AmUogUHXJi9CmGZcw3i-Wsis86ggOJKl4e7TwuKSBqt-s81jzGePI2yIsyKInEXwieKHXpWl1JFMtSkDpkRBeaiSlM1qpJ33BJLekRRkW-mDhV-yG5VVxyOWxRZDSfXRgrQ3CoNzChvITqdC1VOCeMAMI5Vg5zvo9bNOjOqOCLEtncsHdDiD7gYmPsGWeR9eXcT0y2-KONa0LvsYBewBcXjvJE63xe3XViiQ3HQPayjA1UAxWekD83_Kq7y-LJEjrQNNphEq_XyocpzvlmK-tlf59UGJJcw',
        max_input_tokens=12582,
        max_total_tokens=16777,
        max_batch_prefill_tokens=12582+50,
        payload_limit=5_000_000
    )

@pytest.fixture
def llm(tgi_self_hosted_config: HuggingFaceTGIConfig) -> HuggingFaceLLM:
    return HuggingFaceLLM(
        base_url=tgi_self_hosted_config.url,
        credentials=tgi_self_hosted_config.auth_token,
        tgi_config=tgi_self_hosted_config,
        max_tokens=tgi_self_hosted_config.available_generated_tokens,
        temperature=0.8 
    )

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
    config = RedisConfig(
        index_name="test1",
        redis_url=os.environ['REDIS_URL'],
        metadata_schema=[
            {"name": "input", "type": "text"},
            {"name": "output", "type": "text"},
        ],
        # setting this avoids unnecessary request for embeddings
        embedding_dimensions=tokenizer.dimensions
    )

    store = RedisVectorStore(embeddings, config=config)

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
        """
        Load usage statistics in all runs part of
        the composite Runnable
        """
        usage = response.llm_output.get('token_usage')
        for run_id in self.chain_run_ids:
            self.usage_by_run_id[run_id] = usage

class ConfigurableCaptureCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.captured_temp = None

    def on_llm_end(self, response: LLMResult, run_id=None, **kwargs):
        if response.llm_output is not None:
            self.captured_temp = response.llm_output.get('final_temp')

@pytest.fixture
def spy_llm(tgi_self_hosted_config: HuggingFaceTGIConfig) -> SpyHuggingFaceLLM:
    return SpyHuggingFaceLLM(
        base_url=tgi_self_hosted_config.url,
        credentials=tgi_self_hosted_config.auth_token,
        tgi_config=tgi_self_hosted_config,
        max_tokens=tgi_self_hosted_config.available_generated_tokens,
        temperature=0.8 
    )

def test_llm_type(llm: HuggingFaceLLM):
    assert getattr(llm, '_llm_type') == 'huggingface_llm'

def test_identifying_params(llm: HuggingFaceLLM):
    assert getattr(llm, '_identifying_params') == {'endpoint_url': 'http://3.210.60.7:8080/', 'model_kwargs': {}}

def test_llm_invoke(llm: HuggingFaceLLM):
    """
    LLMs accept strings as inputs, or objects which can be coerced to string 
    prompts, including List[BaseMessage] and PromptValue.
    """
    ai_message = llm.invoke('What is Generative AI?')
    assert len(ai_message) > 0

def test_llm_invoke_with_prompt_template(llm: HuggingFaceLLM):
    prompt = PromptTemplate(
        input_variables=['input'],
        template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = chain.invoke({'input': 'Memento'})
    assert len(ai_message) > 0

def test_llm_invoke_with_output_parser(llm: HuggingFaceLLM):
    """
    Output Parser generates instructions injected in prompt, prompting
    LLM to generate json using Pydantic model schema. JSON Parser 
    validates and fixes the JSON string into actual JSON, and Pydantic
    Parser validates the actual JSON data against the Pydantic Schema.
    The result of the Runnable is a valid Pydantic model.

    Some LLMs are inferior to others when generating json. For the Meta
    Llama models, I found I had to include 'Return ONLY a single valid 
    JSON object.' in the prompt to get a single json response back.
    """
    output_parser = PydanticOutputParser(pydantic_object=MovieSummary)

    prompt = PromptTemplate(
        template="Tell me about the movie {input}.\nReturn ONLY a single valid JSON object.\n{format_instructions}\n",
        input_variables=["input"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    print(prompt.format(input='Memento'))

    chain = prompt | llm | output_parser
    ai_message = chain.invoke({'input': 'Memento'})
    assert isinstance(ai_message, MovieSummary)
    assert ai_message.release_year == 2000
    assert ai_message.director == 'Christopher Nolan'
    assert len(ai_message.plot_summary) > 1

def test_llm_invoke_with_few_shot_prompt(
        llm: HuggingFaceLLM, 
        vectorstore: Iterator[RedisVectorStore],
        sample_population: List[str]):
    """
    """
    def example_to_text(
        example: dict[str, str], 
    ) -> str:
        sorted_keys = sorted(example.keys())
        return " ".join(example[k] for k in sorted_keys)

    string_examples = [example_to_text(eg) for eg in examples]

    index_ids = vectorstore.add_texts(string_examples, metadatas=examples)
    print(index_ids)

    example_selector = MaxMarginalRelevanceExampleSelector(
        vectorstore=vectorstore,
        k=3
    )

    selector_output = example_selector.select_examples({'input': 'Which names have the highest salaries?'})
    print(selector_output)
    
    # Once we have the selected examples example selector returned, then we format them for the eventual prompt sent to the model
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
    )

    salaries = itertools.count(start=10_000, step=10_000)
    ages = itertools.count(start=25, step=5)
    data = zip(sample_population, ages, salaries)
    df = pd.DataFrame(data, columns=['Name', 'Age', 'Salary'])

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

    mmr_prompt_output = mmr_prompt.format(input='Which names have the highest salaries?')
    print(mmr_prompt_output)

    # another option is to add a PandasOutputParser to determine if response is valid pandas
    # expression, for example, a @field_validator on an expression field which loads
    # the returned string into eval to validate its valid pandas expression. If not,
    # then OutputParser will generate error and in a langgraph we can send the prompt
    # back to the language model with additional context to correct the mistake.
    chain = mmr_prompt | llm
    ai_message = chain.invoke({ 'input': 'Which names have the highest salaries?' })
    
    df_content = eval(ai_message, {'df': df})

    assert len(df_content) > 0

def test_llm_invoke_with_callbacks(llm: HuggingFaceLLM):
    mock_handler = MockCallbackHandler()
    config = RunnableConfig(callbacks=[mock_handler])
    prompt = PromptTemplate(
        input_variables=['input'],
        template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    chain.invoke({'input': 'Memento'}, config=config)

    assert mock_handler.llm_end_data is not None

def test_llm_invoke_with_run_information(spy_llm: SpyHuggingFaceLLM):
    """
    configurable_fields returns a new RunnableSerializable object
    with newly configured fields
    
    Most of behavior is in RunnableConfigurableFields with extends
    DynamicRunnable

    Note the change in Configurable is ephemeral. It does not affect
    the original llm object, only for an ephemeral operation on the 
    chain to use the changed field, such as temperature from 0.8 to
    0.3 temporarily.
    """
    llm = spy_llm.configurable_fields(
        temperature=ConfigurableField(
            id='temperature',
            name='LLM Temperature',
            description='The temperature of the LLM'
        )
    )

    handler = ConfigurableCaptureCallbackHandler()
    
    run_uuid = uuid.uuid4()
    config = RunnableConfig(
        tags=['huggingface_llm', 'llm_123'],
        metadata={'user_uuid': '12345', 'model': 'meta/llama-3.2-90b-vision-instruct'},
        run_name='huggingface_llm_invoke_role',
        max_concurrency=1, # more applicable when batching or using composite chains
        configurable={'temperature': 0.3}, # more applicable when you have multiple models in chain and want some to have configurable attribute like temperature but not others
        run_id=run_uuid,
        callbacks=[handler], 
    )
    prompt = PromptTemplate(
        input_variables=['input'],
        template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    chain.invoke({'input': 'Memento'}, config=config)

    assert handler.captured_temp == 0.3, (
        f"Expected temperature to be 0.3, got {spy_llm.last_used_temperature}"
    )

def test_llm_invoke_with_token_usage_in_response(llm: HuggingFaceLLM):
    """
    run id is ephemeral. It is consumed by the first runner, RunnableSequence
    below in the composite pattern.

    Therefore, we generate a callback id in order to capture token usage
    in callback handler.
    """
    usage_collector = UsageCollectorWithChainID()
    run_uuid = uuid.uuid4()
    config = RunnableConfig(
        run_id=run_uuid, 
        tags=['huggingface_llm_role'], 
        callbacks=[usage_collector]
    )
    
    prompt = PromptTemplate(
        input_variables=['input'],
        template="Tell me about the movie {input}."
    )

    chain = prompt | llm
    ai_message = chain.invoke({'input': 'Memento'}, config=config)

    usage_info = usage_collector.usage_by_run_id.get(run_uuid)

    structured_output = {
        'answer': ai_message,
        'token_usage': usage_info
    }

    assert all(structured_output['token_usage'].values())

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_image_to_text():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_tool_calling():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_ainvoke():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_stream():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_astream():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_batch():
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_abatch():
    ...



    # TODO: chat_model 
    # huggingface bot
    # then the example selectors in the dataframe expression tool tests    
    # document loader with metadata and smart vector retriever
    # langgraph