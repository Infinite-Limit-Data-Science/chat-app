import base64
from typing import Optional, List, Any, Dict, Iterator, AsyncIterator, Self, Tuple, TypeAlias
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs.generation import GenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.passthrough import RunnablePassthrough, RunnableAssign
from langchain_core.utils.utils import get_pydantic_field_names
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import Field, model_validator

VLLMOpenAI: TypeAlias = OpenAI 
AsyncVLLMOpenAI: TypeAlias = AsyncOpenAI 

_MODEL_NAME = 'microsoft/Phi-3.5-vision-instruct'

_UNSUPPORTED_KEYS = {'tools', 'tool_choice'}

def _message_transform(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """transform Langchain-style message structure to OpenAI Chat Completion"""
    role_mapping = {
        'system': 'system',
        'human': 'user',
        'ai': 'assistant',
    }
    
    messages = []
    for message in messages:
        role, content = message
        converted_role = role_mapping.get(role, 'user')
        converted_message = {'role': converted_role, 'content': []}

        if isinstance(content, str):
            converted_message['content'].append({
                'type': 'text',
                "text": content
            })
        elif isinstance(content, dict) and 'image_url' in content:
            converted_message['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': content['image_url']
                }
            })
        elif isinstance(content, dict):
            for key, value in content.items():
                converted_message['content'].append({
                    'type': key,
                    key: value
                })
    
        messages.append(converted_message)
    return messages

class MyVLLM(BaseLLM):
    """
    This is an implementation of `LLM` component of the Runnable Interface
    Component: LLM
    Input Type: Single string, list of chat messages or a PromptValue
    Output Type: String

    The existing VLLM module available at vllm-project that integrates with BaseLLM requires
    explicit connection to HF model id or local files. It does not have an Inference Client.

    Note this class extends BaseLLM, a foundational class for general-purpose language models 
    that accept text and generate text. It focuses on handling prompts as strings and generating
    textual completions. Use MyChatVLLM for multimodality.

    Note this is not intended as a conversational class. See ChatMultiModalVLLM for a class designed 
    to deal with conversations involving roles like "human: " or "AI: ".
    """
    client: VLLMOpenAI
    endpoint_url: str
    huggingfacehub_api_token: str
    model: str = _MODEL_NAME
    # 
    max_tokens: int = 1024
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    logprobs: Optional[int] = None
    stream: bool = False
    stop: List[str] = Field(default_factory=list)
    server_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get('model_kwargs', {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f'Found {field_name} supplied twice.')
            if field_name not in all_required_field_names:
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f'Parameters {invalid_model_kwargs} should be specified explicitly. '
                f'Instead they were passed in as part of `model_kwargs` parameter.'
            )

        if _UNSUPPORTED_KEYS.intersection(extra.keys()):
            raise ValueError('vLLM does not support this feature')
        values['model_kwargs'] = extra
        
        return values

    @model_validator(mode='after')
    @classmethod
    def validate_environment(self) -> Self:
        AsyncOpenAI
        self.client = OpenAI(
            api_key=self.huggingfacehub_api_token, 
            base_url=self.endpoint_url)

        return self

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling vLLM Inference API."""
        return {
            'max_tokens': self.max_tokens,
            'n': self.n,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'logprobs': self.logprobs,
            'stream': self.stream,
            'stop': self.stop,
            **self.model_kwargs,
        }

    def _invocation_params(
        self, runtime_stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Combine the default parameters and any runtime-supplied kwargs
        Add the runtime-supplied stop sequences to the default stop sequences
        """
        params = {**self._default_params, **kwargs}
        params['stop_sequences'] = params['stop_sequences'] + (runtime_stop or [])
        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'vllm_endpoint'
    
    def _text_completion(self, prompt: str) -> Tuple[str, str]:
        """Single-turn prompt"""
        return ('human', prompt)

    def _run_client(self, message, **invocation_params) -> ChatCompletion:
        chat_completion = self.client.chat.completions.create(
            [message],
            **invocation_params)
        return chat_completion

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to vLLM Inference Server on chain `invoke` invocation."""
        invocation_params = self._invocation_params(stop, **kwargs)
        if self.stream:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **invocation_params):
                completion += chunk.text
            return completion
        else:
            message = _message_transform(self._text_completion(prompt))
            chat_completion = self._run_client(message, **invocation_params)
            response_text = chat_completion.choices[0].message.content
            
            for stop_seq in invocation_params['stop_sequences']:
                if response_text[-len(stop_seq) :] == stop_seq:
                    response_text = response_text[: -len(stop_seq)]
            return response_text

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        pass

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)
        message = _message_transform(self._text_completion(prompt))
        chat_stream = self._run_client(message, **invocation_params)

        for response in chat_stream:
            stop_seq_found: Optional[str] = None
            for stop_seq in invocation_params['stop_sequences']:
                if stop_seq in response:
                    stop_seq_found = stop_seq

            text: Optional[str] = None
            if stop_seq_found:
                text = response[: response.index(stop_seq_found)]
            else:
                text = response

            if text:
                chunk = GenerationChunk(text=text)

                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk

            if stop_seq_found:
                break

    def _astream():
        pass

class MyChatVLLM(BaseChatModel):
    """
    This is an implementation of `ChatModel` component of the Runnable Interface
    Component: ChatModel
    Input Type: Single string, list of chat messages or a PromptValue
    Output Type: ChatMessage
    
    MyChatVLLM is designed for handling conversational language models. 
    
    It is specialized for models that support multi-turn conversations 
    where the input consists of a sequence of messages with distinct roles 
    (e.g., "system", "human", "AI").
    """
    pass