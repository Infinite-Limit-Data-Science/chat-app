import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk, Generation, LLMResult
from langchain_core.utils import get_pydantic_field_names
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self
from huggingface_hub.inference._generated.types import (
    ChatCompletionInputGrammarType,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum
)
from .inference_schema import HuggingFaceInferenceServerMixin
from .huggingface_inference_client import HuggingFaceInferenceClient
from .huggingface_inference_server_config import HuggingFaceInferenceConfig
from .helpers.chat_completion_helper import (
    postprocess_chat_completion_output,
    postprocess_chat_completion_stream_output,
    strip_stop_sequences,
    truncate_at_stop_sequence,
)
from .helpers.run_manager_helper import (
    handle_sync_run_manager,
    handle_async_run_manager
)

logger = logging.getLogger(__name__)

class HuggingFaceLLM(LLM, HuggingFaceInferenceServerMixin):
    client: Optional[HuggingFaceInferenceClient] = Field(description='Hugging Face Inference Client to interface with Hugging Face Hub Messages API', default=None)
    # InferenceClient parameters (in addition to HuggingFaceInferenceServerMixin)
    timeout: float = 120

    # chat_completion parameters supported by Hugging Face Hub Messages API and Hugging Face TGI
    # common parameters are provided explicitly, the remaining parameters to HuggingFaceInferenceClient.chat_completion are provided through `model_kwargs`
    # max_tokens is a critical parameter. And it is often overriden. Here we default to 512 tokens, but you may want to increase it.
    max_tokens: Optional[int] = Field(description='Maximum number of tokens allowed in the response. That is, the maximum number of tokens that can be generated in the chat completion.', default=512)
    num_generations: Optional[int] = Field(description='The number of completions to generate for each prompt. Specifies how many responses the model should generate for a single input prompt.', default=None)
    response_format: Optional[ChatCompletionInputGrammarType] = Field(description='Constrain the model\'s output to adhere to a defined structure or pattern, enhancing the precision and reliability of the generated responses.', default=None)
    stream_options: Optional[ChatCompletionInputStreamOptions] = Field(description='Configure how streaming responses are returned. If include_usage=True, an additional chunk is streamed before the final [DONE] message, containing total token usage.', default=None)
    temperature: Optional[float] = Field(description='Controls randomness of the generations', default=0.8)
    tools: Optional[List[ChatCompletionInputTool]] = Field(description='list of tools the model may "call".', default=None)
    tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = Field(description='Determines how the model selects a tool from the available tools list.', default=None)
    tool_prompt: Optional[str] = Field(description='Append extra instructions before the tool is executed', default=None)  
    stop: List[str] = Field(description='Model stop sequences represented as tokens', default_factory=list)
    # all remaining chat completions parameters supported by HuggingFaceInferenceClient can be passed through model_kwargs 
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    streaming: Optional[bool] = Field(description='Enable realtime streaming of responses', default=False)

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode='before')
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please make sure that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        
        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
        
        values["model_kwargs"] = extra
        return values

    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        client = HuggingFaceInferenceClient(
            base_url=self.base_url,
            credentials=self.credentials,
            timeout=self.timeout,
            headers=self.headers,
            provider=self.provider,
            model=self.model,
        )
        self.client = client

        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            'max_tokens': self.max_tokens,
            'num_generations': self.num_generations,
            'response_format': self.response_format,
            'stream_options': self.stream_options,
            'temperature': self.temperature,
            'tools': self.tools,
            'tool_choice': self.tool_choice,
            'tool_prompt': self.tool_prompt,
            'stop': self.stop,
            **self.model_kwargs,
        }
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.base_url },
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_llm"
    
    def _invocation_params(
        self, runtime_stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        params['stop'] = params['stop'] + (runtime_stop or [])
        return params

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        try:
            if self.streaming:
                completion = ""
                for chunk in self._stream(prompt, stop, run_manager, **invocation_params):
                    completion += chunk.text
                return completion
            else:
                chat_completion_output = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    **invocation_params
                )
                response_text, token_usage = postprocess_chat_completion_output(
                    chat_completion_output, invocation_params
                )
                if run_manager:
                    handle_sync_run_manager(run_manager, response_text, token_usage)
                return strip_stop_sequences(response_text, invocation_params["stop"])
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise    

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        try:
            if self.streaming:
                completion = ""
                async for chunk in self._astream(
                    prompt, stop, run_manager, **invocation_params
                ):
                    completion += chunk.text
                return completion
            chat_completion_output = await self.client.achat_completion(
                messages=[{"role": "user", "content": prompt}],
                **invocation_params
            )
            text, token_usage = postprocess_chat_completion_output(
                chat_completion_output, invocation_params
            )
            if run_manager:
                await handle_async_run_manager(run_manager, text, token_usage)
            return strip_stop_sequences(text, invocation_params["stop"])
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        for chat_completion_stream_output in self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **invocation_params,
            stream=True
        ):
            text_chunk, token_usage = postprocess_chat_completion_stream_output(
                chat_completion_stream_output, invocation_params
            )

            text_chunk, found_stop = truncate_at_stop_sequence(
                text_chunk,
                invocation_params.get("stop", [])
            )

            chunk = GenerationChunk(
                text=text_chunk,
                generation_info=token_usage
            )
            
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, **chunk.generation_info)

            if text_chunk:
                yield chunk

            if found_stop:
                break

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)

        streaming_completion = await self.client.achat_completion(
            messages=[{"role": "user", "content": prompt}],
            **invocation_params,
            stream=True
        )
        async for chat_completion_stream_output in streaming_completion:
            text_chunk, token_usage = postprocess_chat_completion_stream_output(
                chat_completion_stream_output, invocation_params
            )

            text_chunk, found_stop = truncate_at_stop_sequence(
                text_chunk,
                invocation_params.get("stop", [])
            )

            chunk = GenerationChunk(
                text=text_chunk,
                generation_info=token_usage
            )
            
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk, **chunk.generation_info)
            
            if text_chunk:
                yield chunk

            if found_stop:
                break