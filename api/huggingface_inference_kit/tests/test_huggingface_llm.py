import pytest
from langchain_core.prompts import PromptTemplate
from ..huggingface_inference_server_config import HuggingFaceTGIConfig
from ..huggingface_llm import HuggingFaceLLM

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

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_output_parser(llm: HuggingFaceLLM):
    """
    START HERE
    """
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_few_shot_prompt(llm: HuggingFaceLLM):
    """
    Use Example Selector, FewShotPromptTemplate, MaximumMarginalRelevance
    """
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_callbacks():
    # from langchain_core.callbacks import StdOutCallbackHandler
    # config = RunnableConfig(callbacks=[StdOutCallbackHandler()])
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_invoke_with_run_information():
    # tags=config.get("tags"),
    # metadata=config.get("metadata"),
    # run_name=config.get("run_name"),
    # run_id=config.pop("run_id", None),
    # config = RunnableConfig(tags=["my_custom_chain", "summarization"])
    # config = RunnableConfig(metadata={"user_id": "12345", "model": "gpt-4"})
    # config = RunnableConfig(run_name="SummarizationRun")
    # config = RunnableConfig(run_id=uuid4())
    ...

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_ainvoke():
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

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_llm_with_few_shot():
    ...



    # TODO: chat_model and llm classes
    # then the example selectors in the dataframe expression tool tests
    #def test_max_marginal_relevance_selector():
        # Similarity Selector
        # Max Marginal Relevance (MMR) Selector
        # N-gram Overlap Selector  
        # pass  
