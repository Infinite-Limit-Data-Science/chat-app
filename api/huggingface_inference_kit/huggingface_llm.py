import inspect
import json  # type: ignore[import-not-found]
import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
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

logger = logging.getLogger(__name__)

class HuggingFaceLLM(LLM, HuggingFaceInferenceServerMixin):
    # InferenceClient parameters (in addition to HuggingFaceInferenceServerMixin)
    timeout: float = 120

    # chat completion parameters (since chat completion is called inside `_call` (which is in turned called by `invoke`))
    # common parameters are provided explicitly, the remaining parameters to HuggingFaceInferenceClient.chat_completion is provided through `model_kwargs`
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

    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        client = HuggingFaceInferenceClient(
            base_url=self.base_url,
            credentials=self.credentials,
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

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.streaming:
            completion = ""
            # for chunk in self._stream(prompt, stop, run_manager, **invocation_params):
            #     completion += chunk.text
            # return completion
        else:
            invocation_params['stream'] = False
            response = self.client.chat_completion(
                [('human', prompt)],
                **invocation_params
            )
            response_text = str(response)

            for stop_seq in invocation_params['stop']:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text