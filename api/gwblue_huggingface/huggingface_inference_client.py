from typing import (
    Protocol, 
    Annotated, 
    Self, 
    Optional, 
    Dict, 
    Union,
    List, 
    Iterable,
    Literal, 
    overload,
    runtime_checkable,
    AsyncIterable,
    Any,
    TypedDict,
    TypeAlias,
)
from typing_extensions import Doc
from urllib.parse import urlparse
import numpy as np
from pydantic import model_validator, field_validator, Field, ConfigDict
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput, 
    ChatCompletionStreamOutput,
    ChatCompletionInputGrammarType, 
    ChatCompletionInputStreamOptions,
    ChatCompletionInputToolChoiceClass, 
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionInputTool
)
from huggingface_hub import (
    InferenceClient, 
    AsyncInferenceClient,
)
from huggingface_hub.inference._providers import get_provider_helper
import huggingface_hub.inference._providers as providers
from huggingface_hub.inference._common import _import_numpy, _bytes_to_dict, RequestParameters
from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import (
    TaskProviderHelper,
    filter_none,
)
from .inference_schema import HuggingFaceInferenceServerMixin
from.huggingface_inference_server_config import HuggingFaceInferenceConfig, HuggingFaceEmbeddingsConfig
from .providers.vllm import VLLMEmbeddingTask

providers.PROVIDERS["vllm"] = {
    "embedding": VLLMEmbeddingTask()
}

_SUPPORTED_VISION_EMBEDDINGS = ["TIGER-Lab/VLM2Vec-Full"]

class ContentItemImageUrl(TypedDict):
    type: Literal["image_url"]
    image_url: dict

class ContentItemText(TypedDict):
    type: Literal["text"]
    text: str

ContentItem = Union[ContentItemImageUrl, ContentItemText]

class ChatCompletionMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: list[ContentItem]

EmbeddingsInputLike: TypeAlias = Union[str, list[str], List[ChatCompletionMessage]]

class HuggingFaceBaseInferenceClient(HuggingFaceInferenceServerMixin):
    client: Optional[InferenceClient] = Field(description='A low-level Inference Client supported by Hugging Face Hub', default=None)
    async_client: Optional[AsyncInferenceClient] = Field(description='A low-level Async Inference Client supported by Hugging Face Hub', default=None)

    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        if parsed_url.path not in ('', '/'):
            raise ValueError(f'Invalid base_url: {value}. Must not contain extra path segments.')
        return value
    
class HuggingFaceInferenceClient(HuggingFaceBaseInferenceClient):
    inference_config: Optional[HuggingFaceInferenceConfig] = None
    embeddings_config: Optional[HuggingFaceEmbeddingsConfig] = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        try:
            client = InferenceClient(
                model=self.base_url,
                api_key=self.credentials,
                timeout=self.timeout,
                headers=self.headers,
                provider=self.provider,
            )

            self.client = client

            async_client = AsyncInferenceClient(
                base_url=self.base_url,
                api_key=self.credentials,
                timeout=self.timeout,
                headers=self.headers,
                provider=self.provider,
            )

            self.async_client = async_client
        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        
        return self

    def _prepare_request_params(
        self,
        texts: str | List[str],
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None, 
    ) -> RequestParameters:
        provider_helper = get_provider_helper('hf-inference', task='feature-extraction')
        request_parameters = provider_helper.prepare_request(
            inputs=texts,
            parameters={
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
            },
            headers=self.headers or {},
            model=self.base_url,
            api_key=self.credentials,
        )

        return request_parameters

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[False] = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionOutput: 
        """
        If stream False, we return `ChatCompletionOutput` object
        """
        ...

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[True] = True,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Iterable[ChatCompletionStreamOutput]:
        """
        If stream True, we return an iterable, `ChatCompletionStreamOutput`
        """
        ...

    def chat_completion(
        self,
        messages: List[Dict],
        *,
        stream: Optional[bool] = False,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[List[float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]:
        """
        A normal ChatCompletionOutput if stream=False
        A generator object that implements Iterable[ChatCompletionStreamOutput] protocol if stream=True,
        Hence, `chat_completion` can return either a regular object or a generator object
        
        If stream True, it returns an generator object—a special object implementing 
        the iteration protocol (__iter__/__next__).

        Example:
        ```python
        
        def _async_stream_chat_completion_response() -> Iterable[ChatCompletionStreamOutput]::
            yield ChatCompletionStreamOutput1
            yield ChatCompletionStreamOutput2

        gen_obj = _stream_chat_completion_response()
        ```
        """
        if self.inference_config:
            if not max_tokens:
                max_tokens = self.inference_config.available_generated_tokens

            if not (1 <= max_tokens <= self.inference_config.available_generated_tokens):
                raise ValueError(f'max_tokens must be between 1 and {self.inference_config.available_generated_tokens}, got {max_tokens}')

        if num_generations:
            print('Warning: The OpenAI API `n` option is unsupported by Hugging Face TGI (even if supported by Hugging Face Hub Messages API). Ignoring the num_generations option..')

        return self.client.chat_completion(
            messages=messages,
            stream=stream,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=num_generations,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options, # returns tokens used if set
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt=tool_prompt,
            top_logprobs=top_logprobs,
            top_p=top_p
        )

    def feature_extraction(
        self,
        inputs: EmbeddingsInputLike,
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
    ) -> np.ndarray:
        if self.model in _SUPPORTED_VISION_EMBEDDINGS and self.client.provider == "vllm":
            payload = {
                "model": self.model,
                "text": inputs,
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
            }
            provider_helper = get_provider_helper(self.client.provider, "embedding")
            request_params = provider_helper.prepare_request(
                inputs=payload,
                parameters={}, 
                headers=self.headers or {},
                model=self.base_url,
                api_key=self.credentials,
            )

            resp_bytes = self.client._inner_post(request_params, stream=False)
            embeddings = provider_helper.get_response(resp_bytes)

            np = _import_numpy()
            return np.array(embeddings, dtype="float32")

        request_parameters = self._prepare_request_params(
            inputs, 
            normalize=normalize, 
            prompt_name=prompt_name,
            truncate=truncate,
            truncation_direction=truncation_direction
        )
        response = self.client._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(_bytes_to_dict(response), dtype="float32")
    
    @overload
    async def achat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[False] = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionOutput:
        """
        Streaming False, returns ChatCompletionOutput
        """
        ...

    @overload
    async def achat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[True] = True,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncIterable[ChatCompletionStreamOutput]: 
        """
        Streaming True, returns async iterable, ChatCompletionStreamOutput
        """
        ...

    async def achat_completion(
        self,
        messages: List[Dict],
        *,
        stream: bool = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, AsyncIterable[ChatCompletionStreamOutput]]:
        """
        A normal ChatCompletionOutput if stream=False
        An async generator object that implements AsyncIterable[ChatCompletionStreamOutput] protocol if stream=True,
        Hence, `achat_completion` can return either a regular object or an async generator object
        
        If stream True, it returns an async generator object—a special object implementing 
        the asynchronous iteration protocol (__aiter__/__anext__).

        Example:
        ```python
        
        async def _async_stream_chat_completion_response() -> AsyncIterable[ChatCompletionStreamOutput]::
            yield ChatCompletionStreamOutput1
            yield ChatCompletionStreamOutput2

        gen_obj = _async_stream_chat_completion_response()
        ```
        """
        return await self.async_client.chat_completion(
            messages=messages,
            stream=stream,
            frequency_penalty=frequency_penalty,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=num_generations,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options, # returns tokens used if set
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt=tool_prompt,
            top_logprobs=top_logprobs,
            top_p=top_p
        )
    
    async def afeature_extraction(
        self,
        inputs: EmbeddingsInputLike,
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
    ) -> "np.ndarray":
        if self.model in _SUPPORTED_VISION_EMBEDDINGS and self.client.provider == "vllm":
            payload = {
                "model": self.model,
                "text": inputs,
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
            }
            provider_helper = get_provider_helper(self.client.provider, "embedding")
            request_params = provider_helper.prepare_request(
                inputs=payload,
                parameters={}, 
                headers=self.headers or {},
                model=self.base_url,
                api_key=self.credentials,
            )

            resp_bytes = await self.async_client._inner_post(request_params, stream=False)
            embeddings = provider_helper.get_response(resp_bytes)

            np = _import_numpy()
            return np.array(embeddings, dtype="float32")

        request_parameters = self._prepare_request_params(
            inputs, 
            normalize=normalize, 
            prompt_name=prompt_name,
            truncate=truncate,
            truncation_direction=truncation_direction
        )
        response = await self.async_client._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(_bytes_to_dict(response), dtype="float32")