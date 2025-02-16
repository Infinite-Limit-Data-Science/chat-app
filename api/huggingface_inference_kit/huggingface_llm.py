import inspect
import json  # type: ignore[import-not-found]
import logging
import os
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
    ChatCompletionOutput,
    ChatCompletionInputGrammarType,
    ChatCompletionInputStreamOptions,
    ChatCompletionInputTool,
    ChatCompletionInputToolChoiceClass,
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionStreamOutput
)
from .inference_schema import HuggingFaceInferenceServerMixin
from .huggingface_inference_client import HuggingFaceInferenceClient
from .huggingface_inference_server_config import HuggingFaceTGIConfig

logger = logging.getLogger(__name__)

# from langchain_huggingface import HuggingFaceEndpoint

class HuggingFaceLLM(LLM, HuggingFaceInferenceServerMixin):
    client: Optional[HuggingFaceInferenceClient] = Field(description='Hugging Face Inference Client to interface with Hugging Face Hub Messages API', default=None)
    # InferenceClient parameters (in addition to HuggingFaceInferenceServerMixin)
    timeout: float = 120
    tgi_config: Optional[HuggingFaceTGIConfig] = None

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
            tgi_config=self.tgi_config,
            timeout=self.timeout,
            headers=self.headers
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

    def _postprocess_chat_completion_output(
        self,
        chat_completion_output: ChatCompletionOutput,
        invocation_params: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Extract the text from `chat_completion_output` and gather token usage."""
        completion_candidate = chat_completion_output.choices[0]
        response_text = completion_candidate.message.content
        finish_reason = completion_candidate.finish_reason

        token_usage = {}
        if finish_reason in ("stop", "eos_token") and chat_completion_output.usage:
            token_usage["prompt_tokens"] = chat_completion_output.usage.prompt_tokens
            token_usage["completion_tokens"] = chat_completion_output.usage.completion_tokens
            token_usage["total_tokens"] = chat_completion_output.usage.total_tokens

        if invocation_params.get("logprobs", False) and completion_candidate.logprobs:
            import numpy as np
            content = completion_candidate.logprobs.content
            logprobs = [logprob.logprob for logprob in content]
            mean_logprob = np.mean(logprobs)
            token_usage["mean_logprob"] = mean_logprob

        return response_text, token_usage

    def _handle_sync_run_manager(
        self,
        run_manager: Optional[CallbackManagerForLLMRun],
        response_text: str,
        token_usage: Dict[str, Any],
    ) -> None:
        if run_manager is None:
            return
        llm_result = LLMResult(
            generations=[[Generation(text=response_text)]],
            llm_output={"token_usage": token_usage},
        )
        run_manager.on_llm_end(llm_result)

    async def _handle_async_run_manager(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        response_text: str,
        token_usage: Dict[str, Any],
    ) -> None:
        if run_manager is None:
            return
        llm_result = LLMResult(
            generations=[[Generation(text=response_text)]],
            llm_output={"token_usage": token_usage},
        )
        await run_manager.on_llm_end(llm_result)

    def _strip_stop_sequences(self, text: str, stop_sequences: List[str]) -> str:
        for stop_seq in stop_sequences:
            if text.endswith(stop_seq):
                text = text[: -len(stop_seq)]
        return text

    def _process_stream_output(
        self,
        chat_completion_stream_output: ChatCompletionStreamOutput,
        invocation_params: Dict[str, Any],
    ) -> Tuple[Optional[GenerationChunk], bool, Dict[str, Any]]:
        completion_candidate = chat_completion_stream_output.choices[0]
        token = completion_candidate.delta.content or ""

        stop_seq_found: Optional[str] = None
        for stop_seq in invocation_params.get("stop", []):
            if stop_seq in token:
                stop_seq_found = stop_seq
                break

        text = token
        if stop_seq_found:
            idx = text.index(stop_seq_found)
            text = text[:idx]

        chunk: Optional[GenerationChunk] = None
        if text:
            chunk = GenerationChunk(text=text)

        finish_reason = completion_candidate.finish_reason
        token_usage: Dict[str, Any] = {}
        if finish_reason in ("stop", "eos_token") and chat_completion_stream_output.usage:
            token_usage["prompt_tokens"] = chat_completion_stream_output.usage.prompt_tokens
            token_usage["completion_tokens"] = chat_completion_stream_output.usage.completion_tokens
            token_usage["total_tokens"] = chat_completion_stream_output.usage.total_tokens

        if invocation_params.get("logprobs", False) and completion_candidate.logprobs:
            import numpy as np
            content = completion_candidate.logprobs.content
            logprobs = [logprob.logprob for logprob in content]
            token_usage["mean_logprob"] = float(np.mean(logprobs))

        return chunk, (stop_seq_found is not None), token_usage

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
                response_text, token_usage = self._postprocess_chat_completion_output(
                    chat_completion_output, invocation_params
                )
                self._handle_sync_run_manager(run_manager, response_text, token_usage)
                return self._strip_stop_sequences(response_text, invocation_params["stop"])
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
            response_text, token_usage = self._postprocess_chat_completion_output(
                chat_completion_output, invocation_params
            )
            await self._handle_async_run_manager(run_manager, response_text, token_usage)
            return self._strip_stop_sequences(response_text, invocation_params["stop"])
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
            chunk, found_stop, token_usage = self._process_stream_output(
                chat_completion_stream_output, invocation_params
            )
            if chunk:
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, **token_usage)
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

        async for chat_completion_stream_output in self.client.achat_completion(
            messages=[{"role": "user", "content": prompt}],
            **invocation_params,
            stream=True
        ):
            chunk, found_stop, token_usage = self._process_stream_output(
                chat_completion_stream_output, invocation_params
            )
            if chunk:
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, **token_usage)
                yield chunk

            if found_stop:
                break