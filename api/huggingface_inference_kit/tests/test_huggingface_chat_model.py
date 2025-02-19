import pytest
import os
import re
import uuid
from uuid import UUID
import itertools
from typing import Iterator, List, Optional, Dict
from faker import Faker
import pandas as pd
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
from pydantic import BaseModel, Field, field_validator
from ..huggingface_inference_server_config import HuggingFaceTGIConfig
from ..huggingface_llm import HuggingFaceLLM
from ..huggingface_chat_model import HuggingFaceChatModel
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
    """
    When using chat model in runnable chain, kwargs are often lost
    For example, RunnableParallel and BasePromptTemplate ignore kwargs
    param and so it will never make it to your model. 
    
    You have two options.

    You can define model params during instantiation and these attributes
    will be marked as inheritable by the runnable chains.

    However, you can also specify model params during invocation time to
    specify parameters for one specific run. This is accomplished using
    `RunnableConfig.configurable`. Example:

    config = RunnableConfig(
        tags=['huggingface_chat_model', 'chat_model_123'],
        metadata={'user_uuid': '12345', 'model': 'meta/llama-3.2-90b-vision-instruct'},
        run_name='huggingface_chat_model_invoke_role',
        max_concurrency=1,
        configurable={'temperature': 0.3},
        run_id=run_uuid,
        callbacks=[handler], 
    )
    chain.invoke({'input': 'Hello'}, config=config)  
    """
    return HuggingFaceLLM(
        base_url=tgi_self_hosted_config.url,
        credentials=tgi_self_hosted_config.auth_token,
        tgi_config=tgi_self_hosted_config,
        max_tokens=tgi_self_hosted_config.available_generated_tokens,
        temperature=0.8,
        logprobs=True
    )

@pytest.fixture
def chat_model(llm: HuggingFaceLLM) -> HuggingFaceChatModel:
    return HuggingFaceChatModel(llm=llm)

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
def spy_llm(tgi_self_hosted_config: HuggingFaceTGIConfig) -> SpyHuggingFaceLLM:
    return SpyHuggingFaceLLM(
        base_url=tgi_self_hosted_config.url,
        credentials=tgi_self_hosted_config.auth_token,
        tgi_config=tgi_self_hosted_config,
        max_tokens=tgi_self_hosted_config.available_generated_tokens,
        temperature=0.8 
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

def test_chat_model_type(chat_model: HuggingFaceChatModel):
    assert getattr(chat_model, '_llm_type') == 'huggingface_chat_model'

def test_chat_model_invoke(chat_model: HuggingFaceChatModel):
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content='What is Generative AI?')
    ]
    ai_message = chat_model.invoke(messages)
    assert ai_message.type == 'ai'
    assert len(ai_message.content) > 0

def test_chat_model_invoke_with_prompt_template(chat_model: HuggingFaceChatModel):
    """
    Use `HumanMessagePromptTemplate.from_template` instead of `HumanMessage`
    if you desire placeholder to be filled in
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You're a helpful assistant"),
        HumanMessagePromptTemplate.from_template('Tell me about the movie {input}.')
    ])

    formatted_messages = chat_prompt.format_messages(input='Memento')
    for message in formatted_messages:
        print(f'{message.type}: {message.content}')

    chain = chat_prompt | chat_model
    
    ai_message = chain.invoke({'input': 'Memento'})
    assert ai_message.type == 'ai'
    assert len(ai_message.content) > 0

def test_chat_model_invoke_with_prompt_template2(chat_model: HuggingFaceChatModel):
    """
    You can also use tuple structure if you desire placeholders to be filled in 
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ('system', "You're a helpful assistant"),
        ('human', 'Tell me about the movie {input}.')
    ])

    formatted_messages = chat_prompt.format_messages(input='Memento')
    for message in formatted_messages:
        print(f'{message.type}: {message.content}')

    chain = chat_prompt | chat_model

    ai_message = chain.invoke({'input': 'Memento'})
    assert ai_message.type == 'ai'
    assert len(ai_message.content) > 0        

def test_chat_model_invoke_with_history(chat_model: HuggingFaceChatModel):
    chat_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', 'Who is the main character in the movie {input}?')
    ])

    ai_history = """
    Memento (2000) is a psychological thriller directed by Christopher Nolan 
    about a man with short-term memory loss who uses tattoos and notes to 
    track down his wife's killer, unfolding in a nonlinear narrative that 
    reveals the mystery in reverse.
    """

    chat_history = [
        ('human', 'Tell me about the movie Memento.'),
        ('assistant', ai_history)
    ]

    formatted_messages = chat_prompt.format_messages(
        chat_history=chat_history,
        input='Memento'
    )

    for message in formatted_messages:
        print(f'{message.type}: {message.content}')

    chain = chat_prompt | chat_model

    ai_message = chain.invoke({'chat_history': chat_history, 'input': 'Memento'})
    assert ai_message.type == 'ai'
    assert len(ai_message.content) > 0 

def test_chat_model_invoke_with_output_parser(chat_model: HuggingFaceChatModel):
    """
    Output parsers accept a string or BaseMessage and, therefore, work with both
    LLM and Chat Model components
    """
    strict_parser = PydanticOutputParser(pydantic_object=MovieSummary)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            (
                "human", 
                "Tell me about the movie {input}.\n"
                "Return ONLY a single valid JSON object.\n"
                "{format_instructions}\n"
            ),
        ],
    )

    prompt = prompt.partial(format_instructions=strict_parser.get_format_instructions())

    print(prompt.format(input='Memento'))

    retry_parser = RetryOutputParser.from_llm(
        parser=strict_parser,
        llm=chat_model# ideally change the temperature at least if you are not going to use different model: chat_model(temperature=0.0)
    )

    completion_chain = prompt | chat_model

    main_chain = RunnableParallel(
        completion=completion_chain,
        prompt_value=prompt
    ) 

    pipeline = main_chain | RunnableLambda(
        lambda x: retry_parser.parse_with_prompt(
            x['completion'].content,
            x['prompt_value']
        )
    )

    ai_message = pipeline.invoke({'input': 'Memento'})
    assert isinstance(ai_message, MovieSummary)
    assert ai_message.release_year == 2000
    assert ai_message.director == 'Christopher Nolan'
    assert len(ai_message.plot_summary) > 1

def test_chat_model_invoke_with_output_parser2(chat_model: HuggingFaceChatModel):
    """
    Output token usage with Output Parser
    """
    strict_parser = PydanticOutputParser(pydantic_object=OutputMessage)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            (
                'human', 
                "{input}.\n"
                "{format_instructions}\n"
            ),
        ],
    )
    
    prompt = prompt.partial(format_instructions=strict_parser.get_format_instructions())

    print(prompt.format(input='Tell me about the movie Memento'))

    retry_parser = RetryOutputParser.from_llm(
        parser=strict_parser,
        llm=chat_model# ideally change the temperature at least if you are not going to use different model: chat_model(temperature=0.0)
    )    

    completion_chain = prompt | chat_model

    main_chain = RunnableParallel(
        completion=completion_chain,
        prompt_value=prompt
    )

    pipeline = main_chain | RunnableLambda(
        lambda x: {
            'model_structured_output': retry_parser.parse_with_prompt(
                x['completion'].content,
                x['prompt_value']
            ),
            'usage_stats': x['completion'].additional_kwargs.get('token_usage'),
        }
    )

    output_message = pipeline.invoke({'input': 'Tell me about the movie Memento'})
    assert isinstance(output_message['model_structured_output'], OutputMessage)
    assert len(output_message['model_structured_output'].summary) > 0
    assert len(output_message['model_structured_output'].content) > 0
    assert output_message['usage_stats']['finish_reason'] == 'stop'
    assert output_message['usage_stats']['prompt_tokens'] > 0
    assert output_message['usage_stats']['completion_tokens'] > 0
    assert output_message['usage_stats']['total_tokens'] > 0
    assert -1.0 <= output_message['usage_stats']['mean_logprob'] <= 1.0

def test_chat_model_invoke_with_few_shot_prompt(
    chat_model: HuggingFaceChatModel, 
    vectorstore: Iterator[RedisVectorStore],
    sample_population: List[str]):
    ...
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
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ('human', "{input}"),
            ('ai', "{output}"),
        ]
    )

    few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
        input_variables=['input'],
        example_selector=example_selector,
        example_prompt=example_prompt
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', prefix),
            few_shot_chat_prompt,
            ('human', "{input}"),
        ]
    )

    chain = final_prompt | chat_model
    ai_message = chain.invoke({ 'input': 'Which names have the highest salaries?' })

    df_content = eval(ai_message.content, {'df': df})

    assert len(df_content) > 0

def test_chat_model_invoke_with_callbacks(chat_model: HuggingFaceLLM):
    mock_handler = MockCallbackHandler()
    config = RunnableConfig(callbacks=[mock_handler])

    prompt = ChatPromptTemplate.from_messages([
        ('system', "You're a helpful assistant"),
        ('human', "Tell me about the movie {input}.")
    ])

    chain = prompt | chat_model
    chain.invoke({'input': 'Memento'}, config=config)

    assert mock_handler.llm_end_data is not None

def test_llm_invoke_with_run_information(spy_chat_model: HuggingFaceChatModel):
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
    chat_model = spy_chat_model.configurable_fields(
        temperature=ConfigurableField(
            id='temperature',
            name='LLM Temperature',
            description='The temperature of the LLM'
        )
    )

    handler = ConfigurableCaptureCallbackHandler()
    
    run_uuid = uuid.uuid4()
    config = RunnableConfig(
        tags=['huggingface_chat_model', 'chat_model_123'],
        metadata={'user_uuid': '12345', 'model': 'meta/llama-3.2-90b-vision-instruct'},
        run_name='huggingface_chat_model_invoke_role',
        max_concurrency=1, # more applicable when batching or using composite chains
        configurable={'temperature': 0.3}, # more applicable when you have multiple models in chain and want some to have configurable attribute like temperature but not others
        run_id=run_uuid,
        callbacks=[handler], 
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            ('human', "Tell me about the movie {input}.")
        ]
    )

    chain = prompt | chat_model
    chain.invoke({'input': 'Memento'}, config=config)

    assert handler.captured_temp == 0.3, (
        f"Expected temperature to be 0.3, got {spy_chat_model.last_used_temperature}"
    )


    # TODO: chat_model, chat_bot (which is an abstraction over chat_model)
    # dataframe expression tool   
    # document loader with metadata and smart vector retriever
    # langgraph

    




# def test_chat_model_invoke_with_structured_output(chat_model: HuggingFaceChatModel):
#     """with_structured_output method to return json"""
#     structured_llm = chat_model.with_structured_output(MovieSummary)
#     ai_message = structured_llm.invoke('Tell me about the movie Memento')
#     assert isinstance(ai_message, MovieSummary)
#     assert ai_message.release_year == 2000
#     assert ai_message.director == 'Christopher Nolan'
#     assert len(ai_message.plot_summary) > 1   

# def store_state_in_case_of_fatal_error():
#     """
#     Since a RunnableSequence is a kind of RunnableSerializable
#     it has a to_json() method to serialize the state of the composite
#     parts (the Runnables)
#     """
