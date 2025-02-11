import pytest
from typing import List
import asyncio
import base64
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from huggingface_hub.inference._generated.types import ChatCompletionOutput
from ..huggingface_inference_client import HuggingFaceInferenceClient
from ..huggingface_transformer_tokenizers import BgeLargePretrainedTokenizer
from ..huggingface_inference_server_config import HuggingFaceTEIConfig, HuggingFaceTGIConfig

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
def tgi_inference_client(tgi_self_hosted_config: HuggingFaceTGIConfig) -> HuggingFaceInferenceClient:
    return HuggingFaceInferenceClient(
        base_url=tgi_self_hosted_config.url,
        credentials=tgi_self_hosted_config.auth_token,
        tgi_config=tgi_self_hosted_config
    )

@pytest.fixture
def tei_inference_client(tei_self_hosted_config: HuggingFaceTEIConfig) -> HuggingFaceInferenceClient:
    return HuggingFaceInferenceClient(
        base_url=tei_self_hosted_config.url,
        credentials=tei_self_hosted_config.auth_token,
        tei_config=tei_self_hosted_config
    )

@pytest.fixture
def bge() -> BgeLargePretrainedTokenizer:
    return BgeLargePretrainedTokenizer()

@pytest.fixture
def corpus() -> str:
    text = """
    The quick brown fox jumps over the lazy dog near the riverbank. 
    As the sun set, the sky turned shades of orange and pink, casting a golden glow on the water. 
    Birds chirped melodiously in the background while a gentle breeze rustled the leaves.
    """
    return text

def test_inference_client_feature_extraction(tei_inference_client: HuggingFaceInferenceClient, corpus: str):
    """
    Invokes the feature-extraction task of the HuggingFace TEI. Note
    this expects the TEI is running in a self-hosted environment.
    A single text or a list of texts are accepted for embedding.
    Internally calls InferenceClient of langchain_huggingface,
    which uses the requests package to send a post request to TEI.

    Note requests package has a json= parameter which will be sent 
    as part of post request to TEI. The format of the json field changes  
    whether using a single text or list of texts:

    single text: '{"inputs":"What is Deep Learning?"}'
    list of texts: '{"inputs":["Today is a nice day", "I like you"]}'
    
    The json= parameter takes a dictionary and it internally gets converted 
    to an actual json string

    TEI returns a byte string of high-dimensional vectors from the requests 
    package post request.
    Example: b'[[0.006687683,-0.02827644,0.021200065,-0.006164584, ... ]'

    Curl examples:

    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
    curl 127.0.0.1:8080/embed \
        -X POST \
        -d '{"inputs":["Today is a nice day", "I like you"]}' \
        -H 'Content-Type: application/json'    

    Note the InferenceClient doesn't append /embed but rather specifies
    the task 'feature_extraction'
        
    feature_extract converts the bytes to a numpy array of dtype float32. 
    These floats are representations of dense numerical vectors of the 
    original input text. If there were six inputs of text, then there will 
    be 6 arrays of floats returned. Each array will be of size 1024, if 
    the dimensions of the model are 1024, such as in BGE-Large. In effect,
    each array is a vector.

    In the case of BGE-Large and most modern embedding models, the entire 
    query text is converted into a single vector representation rather than 
    breaking it into individual word vectors like word2vec or TF-IDF.

    Unlike traditional word embeddings (like word2vec), BGE-Large and 
    transformer-based models use contextual embeddings where the meaning 
    of a word depends on the surrounding words. The model processes the 
    entire query as a sequence and outputs a single vector representing 
    the meaning of the whole query.
    """
    embeddings = tei_inference_client.feature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_feature_extraction_trunc(tei_inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    corpus = " ".join([corpus] * 10)
    embeddings = tei_inference_client.feature_extraction(corpus, truncate=True)

    assert embeddings.size == bge.dimensions

def test_inference_client_feature_extraction_not_tokens(tei_inference_client: HuggingFaceInferenceClient, corpus: str, bge: BgeLargePretrainedTokenizer):
    """
    The output of feature_extraction() is not tokens of text, 
    but rather dense numerical vector representations of the 
    input text.
    
    feature_extraction returns a high-dimensional vector 
    (1024 features for BGE-Large).
    
    The tokenizer converts text into token IDs. The token ids 
    are indices in the tokenizer's vocabulary. But that's not 
    what is returned here. 
    
    Hence, the vectors do not correspond to token IDs

    As such, you cannot directly do anything with the vectors. 
    You cannot convert them back into tokens or text. You can, 
    however, embed a query and then use the vectors of that 
    query to find semantically similar vectors from the original 
    text.
    """
    tokens  = bge.tokenizer.encode(corpus, add_special_tokens=True)
    embeddings = tei_inference_client.feature_extraction(corpus, truncate=True)
    decoded = bge.tokenizer.decode(embeddings[0])

    assert tokens != decoded

@pytest.mark.asyncio
async def test_async_inference_client_feature_extraction(tei_inference_client: HuggingFaceInferenceClient, corpus: str):
    embeddings = await tei_inference_client.afeature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_chat_completion(tgi_inference_client: HuggingFaceInferenceClient):
    """
    Basic request to self-hosted TGI chat completion endpoint

    The returned structured output of chat_completion encapsulates
    a ChatCompletionOutput which contains one or more choices represented
    as a ChatCompletionOutputComplete object, comprising of a finish
    reason (e.g. eos token, max length of token limit reached), an index,
    a message,and logprobs.
    """
    chat_completion_output = tgi_inference_client.chat_completion(
        messages = [
            {
                'role': 'user',
                'content': 'What is Generative AI?'
            }
        ],
        stream=False, # not needed, defaults to False
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        temperature=0.8 # add randomness
    )

    assert len(chat_completion_output.choices) == 1
    # finish_reason shouldn't be 'length' because we are using the available tokens 
    # we specified when configuring the tgi
    assert chat_completion_output.choices[0].finish_reason in ('stop', 'eos_token')
    assert chat_completion_output.choices[0].message.role == 'assistant'
    assert len(chat_completion_output.choices[0].message.content) > 0

def test_inference_client_chat_completion_with_multiple_candidates(tgi_inference_client: HuggingFaceInferenceClient):
    """
    Each ChatCompletionOutputComplete of choices represents a completion
    candidate. Having `num_generations` > 1 specifies to have multiple 
    responses generated for a given input. 

    Multiple candidates allow for better selection, diversity, and reranking.    
    """
    chat_completion_output = tgi_inference_client.chat_completion(
        messages = [
            {
                'role': 'user',
                'content': 'What is Generative AI?'
            }
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        num_generations=3,
        temperature=0.8
    )

    # once multiple candidate completions is supported, change below to ==
    assert len(chat_completion_output.choices) != 3

    for choice in chat_completion_output.choices:
        assert choice.finish_reason in ('stop', 'eos_token')
        assert choice.message.role == 'assistant'
        assert len(choice.message.content) > 0

@pytest.mark.asyncio
async def test_inference_client_chat_completion_with_logprobs(tgi_inference_client: HuggingFaceInferenceClient):
    """
    Each ChatCompletionOutputComplete choice, in addition to a message,
    index, and finish reason, contains a logprobs. logprobs contain
    logarithmic probabilities of tokens. Every word or subword in a 
    generation is sampled from a probability distribution.

    High logprob values (closer to 0) indicate higher confidence in 
    the token.

    Lower logprob values (large negative numbers) indicate lower 
    confidence in the token.

    logprobs are useful in Multiple Candidate Completions when performing
    reranking. Since currently HuggingFace TGI Messages API Chat Completion
    endpoint does not suport multiple candidate completions, we can
    vary temperature, top_p, frequency_penalty to send multiple requests
    and then compare logprobs from the results.

    Remember the Messages API is simply Hugging Face's attempt to mimic
    the behavior of OpenAI's Chat Completions API, but currently is not
    a 1:1 match in features.

    In terms of variation between responses, we can leverage temperature
    or top_p. It's not recommended to alter both of them together.

    Temperature uses Softmax scaling. Higher temperatures (1.2 to 2.0)
    mean the model takes more risks, picking less probable words more often.
    Lower temperatures (0.2 to 0.5) mean the model is more conservative, 
    sticking to high-probability words.

    With temperature, the model still considers all words but scales their 
    probabilities.

    top_p uses nucleus sampling. Instead of considering all tokens, top_p 
    picks only the most likely ones and discards the rest. Lower top_p 
    (e.g., 0.5) means more strictness, keeping only the highest-probability 
    words. Higher top_p (e.g., 0.9) means more flexibility, allowing more 
    variety in token selection.

    Note in the circumstance where you want generation to produce multiple
    responses that use the same temperature or top_p, then the solution is
    to use few shot prompts to send to the model, not to alter these0
    parameters.

    Note every token in the output of ChatCompletionOutputLogprob will have 
    a logprob.
    """
    messages = [{'role': 'user', 'content': 'What is Generative AI?'}]
    max_tokens = tgi_inference_client.tgi_config.available_generated_tokens
    params = [
        {'top_p': 0.5},
        {'top_p': 0.9}
    ]

    async def fetch_chat_completion(top_p_value):
        return tgi_inference_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=top_p_value,
            logprobs=True
        )

    tasks = [fetch_chat_completion(p['top_p']) for p in params]

    chat_completion_outputs: List[ChatCompletionOutput] = await asyncio.gather(*tasks)

    for output in chat_completion_outputs:
        for choice in output.choices:
            logprobs = [logprob.logprob for logprob in choice.logprobs.content]
            mean_logprob = np.mean(logprobs)
            assert -2.5 <= mean_logprob <= 1.0, f'Mean logprob {mean_logprob} out of range'

@pytest.mark.asyncio
async def test_inference_client_chat_completion_with_reranking(tgi_inference_client: HuggingFaceInferenceClient):
    """
    When generating multiple candidate completions (n > 1), the model can 
    return multiple outputs, which can then be ranked based on various 
    criteria such as log probability scores, semantic similarity, or 
    external evaluation metrics.

    Log Probability-Based Reranking (Logprobs)

    If logprobs=True, the model provides log probability scores for 
    each generated token. The overall log probability of a sequence 
    is the sum (or mean) of individual token log probabilities.

    The response with the highest log probability is typically considered 
    the most likely (or best) response.
    """
    messages = [{'role': 'user', 'content': 'What is Generative AI?'}]
    max_tokens = tgi_inference_client.tgi_config.available_generated_tokens
    params = [
        {'top_p': 0.5},
        {'top_p': 0.9}
    ]

    async def fetch_chat_completion(top_p_value):
        return tgi_inference_client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=top_p_value,
            logprobs=True
        )

    tasks = [fetch_chat_completion(p['top_p']) for p in params]

    chat_completion_outputs: List[ChatCompletionOutput] = await asyncio.gather(*tasks)

    logprob_scores = []
    for output in chat_completion_outputs:
        for choice in output.choices:
            logprobs = [logprob.logprob for logprob in choice.logprobs.content]
            mean_logprob = np.mean(logprobs)
            logprob_scores.append((choice, mean_logprob))

    logprob_scores.sort(key=lambda x: x[1], reverse=True)

    best_choice = logprob_scores[0][0]

    assert len(best_choice.message.content) > 0 

def test_inference_client_chat_completion_with_image_to_text(tgi_inference_client: HuggingFaceInferenceClient):
    """
    Multimodals support more than text generation. For example, the
    multimodal Llama 3.2 11B/90B Vision Instruct models support image
    to text. The HuggingFace Messages API Chat Completions endpoint
    supports passing base64 encoded image strings as part of the
    content of the message using an image_url key.
    """
    image_path = Path(__file__).parent / 'assets' / 'baby.jpg'
    with image_path.open('rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    image_url = f'data:image/jpeg;base64,{base64_image}'

    chat_completion_output = tgi_inference_client.chat_completion(
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': image_url},
                    },
                    {
                        'type': 'text',
                        'text': 'Describe this image.'
                    }
                ]
            }
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        temperature=0.8,
        logprobs=True
    )

    for choice in chat_completion_output.choices:
        assert choice.finish_reason in ('stop', 'eos_token')
        assert choice.message.role == 'assistant'
        assert len(choice.message.content) > 0

def test_inference_client_chat_completion_with_output_usage(tgi_inference_client: HuggingFaceInferenceClient):
    """
    The ChatCompletionOutput object comprises of critical fields, such
    as choices, and another one is ChatCompletionOutputUsage. This object 
    contains token usage statistics.

    It includes:
    - Number of tokens generated by the model.
    - Number of tokens in the input prompt.
    - Total tokens used.
    """
    chat_completion_output = tgi_inference_client.chat_completion(
        messages = [
            {
                'role': 'user',
                'content': 'What is Generative AI?'
            }
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        temperature=0.8
    )

    assert chat_completion_output.usage.prompt_tokens > 0
    assert chat_completion_output.usage.completion_tokens > 0
    assert chat_completion_output.usage.total_tokens > 0

# @pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_inference_client_chat_completion_with_tool_calling(tgi_inference_client: HuggingFaceInferenceClient):
    """
    The Hugging Face Model Hub (huggingface_hub) has a collection of APIs 
    to interface with the models on Hugging Face Hub. This includes the
    ChatCompletionOutputComplete, ChatCompletionOutputMessage, and the new
    one, ChatCompletionOutputToolCall. These are objects of the Hugging Face
    Messages API, which seeks to mimic OpenAI's Chat Completions API.

    A `ChatCompletionOutputMessage` (the message of the completion candidate) 
    of a `ChatCompletionOutputComplete` (the completion candidate) contains 
    a tool_calls attribute (in addition to the standard role and content 
    attributes). The tool_calls attribute is a list of `ChatCompletionOutputToolCall`. 
    It is available when the model "calls" a tool. These classes are available
    in huggingface_hub (a package of Hugging Face Model Hub).
   
    In collaboration, Hugging Face TGI supports function calling (tool calling) by 
    forcing the model to generate structured outputs based on your own predefined 
    output schemas.

    TGI supports function calling through a feature known as Guidance. Guidance is 
    a feature that allows users to constrain the generation of a large language 
    model with a specified grammar. This feature is particularly useful when you 
    want to generate text that follows a specific structure or uses a specific 
    set of words or produce output in a specific format. A prominent example is 
    JSON grammar, where the model is forced to output valid JSON.

    Guidance accomplishes this by including a grammar with a generation request 
    that is compiled, and used to modify the chosen tokens. The model does a 
    forward pass over the batch. This returns probabilities for each token in the 
    vocabulary for each request in the batch. The process of choosing one of those 
    tokens is called sampling. The model samples from the distribution of 
    probabilities to choose the next token. In TGI all of the steps before sampling 
    are called processor. Grammars are applied as a processor that masks out tokens 
    that are not allowed by the grammar. Guidance is possible using the 
    /chat/completion endpoint with tools.

    Under the hood tools are a special case of grammars that allows the model to 
    choose one or none of the provided tools.

    A Grammar would look like this in the Messages API:

    schema = {
        "properties": {
            "location": {"title": "Location", "type": "string"},
            "activity": {"title": "Activity", "type": "string"},
            "animals_seen": {
                "maximum": 5,
                "minimum": 1,
                "title": "Animals Seen",
                "type": "integer",
            },
            "animals": {"items": {"type": "string"}, "title": "Animals", "type": "array"},
        },
        "required": ["location", "activity", "animals_seen", "animals"],
        "title": "Animals",
        "type": "object",
    }

    Tools are a set of user defined functions that can be used in tandem with the chat 
    functionality to enhance the LLM's capabilities. Functions, similar to grammar are 
    defined as JSON schema and can be passed as part of the parameters to the Messages 
    API.

    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location."
                    }
                },
                "required": ["location", "format"]
            }
        }
    }

    Not all models on the Hugging Face Model Hub support Tool Calling but the
    Meta Llama models support it.The larger ones, such as Llama 3.1 70B support
    it better than the smaller ones, such as 8B.
    """
    class WeatherForecastRequest(BaseModel):
        location: str = Field(..., description="The city and state, e.g., 'San Francisco, CA'")
        format: str = Field(..., description="Temperature unit: 'celsius' or 'fahrenheit'", enum=["celsius", "fahrenheit"])
        num_days: int = Field(..., description="Number of days to forecast (1-7)", ge=1, le=7)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather_forecast",
                "description": "Get an N-day weather forecast",
                "parameters": WeatherForecastRequest.model_json_schema(),
            },
        }
    ]

    chat_completion_output: ChatCompletionOutput = tgi_inference_client.chat_completion(
        messages = [
            {'role': 'system', 'content': 'Use the provided tools to answer user questions.'},
            {'role': 'user', 'content': "What's the weather like in New York for the next 3 days?"}
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        tools=tools,
        tool_choice='auto',
        temperature=0.8
    )  

    for tool_call in chat_completion_output.choices[0].message.tool_calls:
        assert tool_call.type == 'function'
        assert tool_call.function.name == 'get_weather_forecast'
        assert tool_call.function.arguments['num_days'] == 3
        assert tool_call.function.arguments['location'] == 'New York'
        assert tool_call.function.arguments['format'] == 'celsius'

@pytest.mark.asyncio
async def test_async_inference_client_chat_completion(tgi_inference_client: HuggingFaceInferenceClient):
    """
    HuggingFaceInferenceClient supports returning a coroutine object
    """
    chat_completion_output = await tgi_inference_client.achat_completion(
        messages = [
            {
                'role': 'user',
                'content': 'What is Generative AI?'
            }
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        temperature=0.8
    )

    assert len(chat_completion_output.choices) == 1
    assert chat_completion_output.choices[0].finish_reason in ('stop', 'eos_token')
    assert chat_completion_output.choices[0].message.role == 'assistant'
    assert len(chat_completion_output.choices[0].message.content) > 0

@pytest.mark.asyncio
async def test_async_streaming_inference_client_chat_completion(tgi_inference_client: HuggingFaceInferenceClient):
    """
    Token streaming is the mode in which the server returns the tokens 
    one by one as the model generates them. This enables showing 
    progressive generations to the user rather than waiting for the 
    whole generation.

    `HuggingFaceInferenceClient` supports yielding AsyncGenerator objects
    during streaming. These are instances of `ChatCompletionStreamOutput`.
    Another object type provided by the Hugging Face Hub Messages API. 
    It holds a list of choices, similar to the ChatCompletionOutput in the
    non-streaming version. Instead of choices (multiple candidate 
    completions) being instances of `ChatCompletionOutputComplete`, the
    choices are instances of `ChatCompletionStreamOutputChoice`. This object
    contains a finish_reason, index, and logprobs like its counterpart. But
    instead of message attribute, it contains a delta attribute. The delta
    attribute represents the current token being streamed. 
    """
    chat_completion_output = await tgi_inference_client.achat_completion(
        messages = [
            {
                'role': 'user',
                'content': 'What is Generative AI?'
            }
        ],
        max_tokens=tgi_inference_client.tgi_config.available_generated_tokens,
        temperature=0.8,
        stream=True,
        logprobs=True
    )

    async for chunk in chat_completion_output:
        for choice in chunk.choices:
            assert -10 <= choice.logprobs.content[0].logprob <= 1.0, f'logprob out of range'
            assert choice.delta.role == 'assistant'
            assert len(choice.delta.content) >= 0 # content represents streamed token