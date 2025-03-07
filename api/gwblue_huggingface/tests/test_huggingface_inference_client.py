import pytest
from typing import List
import asyncio
import base64
from pathlib import Path
import numpy as np
import os
import json
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from huggingface_hub.inference._generated.types import ChatCompletionOutput
from ..huggingface_inference_client import HuggingFaceInferenceClient
from ..huggingface_transformer_tokenizers import VLM2VecFullPretrainedTokenizer
from ..huggingface_inference_server_config import HuggingFaceTEIConfig, HuggingFaceTGIConfig

load_dotenv()

def _model_config(model_type=str) -> str:
    models = json.loads(os.environ[model_type])
    
    return {
        'name': models[0]['name'],
        'url': models[0]['endpoints'][0]['url']
    }

@pytest.fixture
def tgi_self_hosted_config() -> HuggingFaceTGIConfig:
    config = _model_config('MODELS')

    return HuggingFaceTGIConfig(
        name=config['name'],
        url=config['url'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_input_tokens=12582,
        max_total_tokens=16777,
        max_batch_prefill_tokens=12582+50,
        payload_limit=5_000_000
    )

@pytest.fixture
def tei_self_hosted_config() -> HuggingFaceTEIConfig:
    config = _model_config('EMBEDDING_MODELS')

    return HuggingFaceTEIConfig(
        name=config['name'],
        url=config['url'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
        max_batch_tokens=32768,
        max_client_batch_size=128,
        max_batch_requests=64,
        auto_truncate=True
    )

@pytest.fixture
def tei_self_hosted_config_vision() -> HuggingFaceTEIConfig:
    config = _model_config('EMBEDDING_MODELS')
    
    return HuggingFaceTEIConfig(
        name=config['name'],
        url=config['url'],
        auth_token=os.environ['TEST_AUTH_TOKEN'],
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
def tei_inference_client_vision(tei_self_hosted_config: HuggingFaceTEIConfig) -> HuggingFaceInferenceClient:
    return HuggingFaceInferenceClient(
        base_url=tei_self_hosted_config.url,
        credentials=tei_self_hosted_config.auth_token,
        tei_config=tei_self_hosted_config,
        model='TIGER-Lab/VLM2Vec-Full'
    )

@pytest.fixture
def vlm2vec() -> VLM2VecFullPretrainedTokenizer:
    return VLM2VecFullPretrainedTokenizer()

@pytest.fixture
def corpus() -> str:
    text = """
    The quick brown fox jumps over the lazy dog near the riverbank. 
    As the sun set, the sky turned shades of orange and pink, casting a golden glow on the water. 
    Birds chirped melodiously in the background while a gentle breeze rustled the leaves.
    """
    return text

def test_inference_client_feature_extraction(tei_inference_client: HuggingFaceInferenceClient, corpus: str):
    embeddings = tei_inference_client.feature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_feature_extraction_trunc(
    tei_inference_client: HuggingFaceInferenceClient, 
    corpus: str, 
    vlm2vec: VLM2VecFullPretrainedTokenizer
):
    corpus = " ".join([corpus] * 10)
    embeddings = tei_inference_client.feature_extraction(corpus, truncate=True)

    assert embeddings.size == vlm2vec.dimensions

def test_inference_client_feature_extraction_not_tokens(
    tei_inference_client: HuggingFaceInferenceClient, 
    corpus: str, 
    vlm2vec: VLM2VecFullPretrainedTokenizer
):
    tokens  = vlm2vec.tokenizer.encode(corpus, add_special_tokens=True)
    embeddings = tei_inference_client.feature_extraction(corpus, truncate=True)
    decoded = vlm2vec.tokenizer.decode(embeddings[0])

    assert tokens != decoded

def test_inference_client_feature_extraction_vision(
    tei_inference_client_vision: HuggingFaceInferenceClient, 
    vlm2vec: VLM2VecFullPretrainedTokenizer
):
    image_path = Path(__file__).parent / 'assets' / 'baby.jpg'
    with image_path.open('rb') as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    image_url = f'data:image/jpeg;base64,{base64_image}'

    tokens  = vlm2vec.tokenizer.encode(corpus, add_special_tokens=True)
    embeddings = tei_inference_client_vision.feature_extraction({
        "model": "TIGER-Lab/VLM2Vec-Full",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe the image."
                    },
                ],
            }
        ],
        "encoding_format": "float",
        "mm_processor_kwargs": {},
    }, truncate=True)
    decoded = vlm2vec.tokenizer.decode(embeddings[0])

    assert tokens != decoded

@pytest.mark.asyncio
async def test_async_inference_client_feature_extraction(tei_inference_client: HuggingFaceInferenceClient, corpus: str):
    embeddings = await tei_inference_client.afeature_extraction(corpus)
    assert embeddings.dtype == 'float32'

def test_inference_client_chat_completion(tgi_inference_client: HuggingFaceInferenceClient):
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