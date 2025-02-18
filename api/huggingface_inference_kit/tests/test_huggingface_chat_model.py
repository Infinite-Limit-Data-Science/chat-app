import pytest
import re
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from ..huggingface_inference_server_config import HuggingFaceTGIConfig
from ..huggingface_llm import HuggingFaceLLM
from ..huggingface_chat_model import HuggingFaceChatModel

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
def chat_model(llm: HuggingFaceLLM) -> HuggingFaceChatModel:
    return HuggingFaceChatModel(llm=llm)

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
        ('ai', ai_history)
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
    output_parser = PydanticOutputParser(pydantic_object=MovieSummary)

def test_chat_model_invoke_with_structured_output(chat_model: HuggingFaceChatModel):
    """Try with the with_structured_output method to return json"""
    ...

    # TODO: chat_model, chat_bot (which is an abstraction over chat_model)
    # dataframe expression tool   
    # document loader with metadata and smart vector retriever
    # langgraph



# def test_chat_model_invoke_with_output_parser(chat_model: HuggingFaceChatModel):
#     ...

# def test_chat_model_invoke_with_few_shot_prompt(
#         chat_model: HuggingFaceChatModel, 
#         vectorstore: Iterator[RedisVectorStore],
#         sample_population: List[str]):
#     ...

# def test_chat_model_invoke_with_callbacks(chat_model: HuggingFaceChatModel):
#     ...

# def test_chat_model_invoke_with_callbacks(chat_model: HuggingFaceChatModel):
#     ...

# def test_chat_model_invoke_with_run_information(spy_chat_model: SpyHuggingFaceChatModel):
#     """
#     This allows us to store uuids as metadata
#     and then we can use langsmith to check
#     performance bottlenecks and usage for specific 
#     uuids (users)
#     """
#     ...

# def test_chat_model_invoke_with_token_usage_in_response(chat_model: HuggingFaceChatModel):
#     ...




# def store_state_in_case_of_fatal_error():
#     """
#     Since a RunnableSequence is a kind of RunnableSerializable
#     it has a to_json() method to serialize the state of the composite
#     parts (the Runnables)
#     """
