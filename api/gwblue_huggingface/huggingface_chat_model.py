from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    Iterator,
    AsyncIterator,
    TypeAlias,
    Tuple
)
import uuid
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import (
    ChatGeneration, 
    ChatGenerationChunk, 
    ChatResult, 
    LLMResult
)
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import model_validator
from typing_extensions import Self
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput,
    ChatCompletionOutputMessage,
    ChatCompletionStreamOutputDelta,
)
from .huggingface_llm import HuggingFaceLLM
from .huggingface_transformer_tokenizers import get_tokenizer_class_by_prefix
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

ChatCompletionOutputContentLike: TypeAlias = ChatCompletionOutputMessage | ChatCompletionStreamOutputDelta

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""

def _convert_message_to_chat_message(
    message: BaseMessage,
) -> Dict:
    if isinstance(message, ChatMessage):
        return dict(role=message.role, content=message.content)
    elif isinstance(message, HumanMessage):
        return dict(role="user", content=message.content)
    elif isinstance(message, AIMessage):
        if "tool_calls" in message.additional_kwargs:
            tool_calls = [
                {
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                }
                for tc in message.additional_kwargs["tool_calls"]
            ]
        else:
            tool_calls = None
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": tool_calls,
        }
    elif isinstance(message, SystemMessage):
        return dict(role="system", content=message.content)
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")

def corrected_functions(fn_string: str) -> str:
    import json, ast

    try:
        py_obj = ast.literal_eval(fn_string)
        corrected_json = json.dumps(py_obj)
    except Exception:
        corrected_json = fn_string.replace("'", '"')

    return corrected_json

def _convert_tgi_message_to_langchain_message(
    message: ChatCompletionOutputContentLike,
    token_usage: Dict[str, any]
) -> Tuple[str, Dict[str, any]]:
    role = message.role
    assert role == 'assistant', f"Expected role to be 'assistant', got {role}"
    content = cast(str, message.content)
    if content is None:
        content = ""
    additional_kwargs: Dict = {}
    additional_kwargs['token_usage'] = token_usage
    if tool_calls := message.tool_calls:
        if 'arguments' in tool_calls[0]['function']:
            functions_string = str(tool_calls[0]['function'].pop('arguments'))
            tool_calls[0]['function']['arguments'] = corrected_functions(functions_string)
        additional_kwargs['tool_calls'] = tool_calls
    return content, additional_kwargs

def _convert_tgi_message_to_lc_ai_message(
    message: ChatCompletionOutputContentLike,
    token_usage: Dict[str, any]    
) -> AIMessage:
    content, additional_kwargs = _convert_tgi_message_to_langchain_message(
        message,
        token_usage      
    )
    return AIMessage(content=content, additional_kwargs=additional_kwargs)

def _convert_tgi_message_to_lc_ai_message_chunk(
    message: ChatCompletionOutputContentLike,
    token_usage: Dict[str, any]    
) -> AIMessageChunk:
    content, additional_kwargs = _convert_tgi_message_to_langchain_message(
        message,
        token_usage  
    )
    additional_kwargs['token_usage'] = { 'chunks': [additional_kwargs['token_usage']] }
    return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)    

class HuggingFaceChatModel(BaseChatModel):
    llm: HuggingFaceLLM
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_name: Optional[str] = None

    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        if not isinstance(self.llm, HuggingFaceLLM):
            raise TypeError(
                "Expected llm to be of type `HuggingFaceLLM`"
                f"received {type(self.llm)}"
            )

        tokenizer_name = self.llm.inference_config.name or self.model_name
        if not tokenizer_name:
            raise TypeError(
                "Expected model name to be defined"
                f"received {type(tokenizer_name)}"
            )
        self.tokenizer = get_tokenizer_class_by_prefix(tokenizer_name)

        return self

    @property
    def _llm_type(self) -> str:
        return 'huggingface_chat_model'
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            'max_tokens': self.llm.max_tokens,
            'num_generations': self.llm.num_generations,
            'response_format': self.llm.response_format,
            'stream_options': self.llm.stream_options,
            'temperature': self.llm.temperature,
            'tools': self.llm.tools,
            'tool_choice': self.llm.tool_choice,
            'tool_prompt': self.llm.tool_prompt,
            'stop': self.llm.stop,
            **self.llm.model_kwargs,
        }

    def _invocation_params(
        self, runtime_stop: Optional[List[str]], **kwargs: Any
    ) -> Dict[str, Any]:
        if isinstance(self.llm, RunnableBinding):
            ephemeral = self.llm.kwargs or {}
        else:
            ephemeral = {}
        
        bound_params = {**self._default_params, **ephemeral, **kwargs}
        bound_params['stop'] = bound_params['stop'] + (runtime_stop or [])
        return bound_params

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> List[Dict[Any, Any]]:
        message_dicts = [_convert_message_to_chat_message(m) for m in messages]
        return message_dicts
    
    def _create_chat_result(self, message: ChatCompletionOutputMessage, token_usage: Dict[str, any]) -> ChatResult:
        generations = []
        gen = ChatGeneration(
            message=_convert_tgi_message_to_lc_ai_message(message, token_usage),
            generation_info=token_usage,
        )
        generations.append(gen)
        model_object = self.llm.client.inference_config.name if self.llm.client.inference_config else None
        llm_output = {"token_usage": token_usage, "model": model_object}
        return ChatResult(generations=generations, llm_output=llm_output)
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        invocation_params = self._invocation_params(stop, **kwargs)
        message_dicts = self._create_message_dicts(messages, stop)

        try:
            chat_completion_output: ChatCompletionOutput = self.llm.client.chat_completion(
                messages=message_dicts,
                **invocation_params 
            )
            _, token_usage = postprocess_chat_completion_output(
                chat_completion_output, invocation_params
            )

            message: ChatCompletionOutputMessage = chat_completion_output.choices[0].message
            if message.content:
                # if not message content, then it is a tool response and we don't want to strip out anything for tool responses
                message.content = strip_stop_sequences(message.content, invocation_params["stop"])

            chat_result = self._create_chat_result(message, token_usage)

            handle_sync_run_manager(
                run_manager, 
                chat_result.generations[0].message.content, 
                token_usage
            )

            return chat_result
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise  

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        invocation_params = self._invocation_params(stop, **kwargs)
        message_dicts = self._create_message_dicts(messages, stop)

        try:
            chat_completion_output: ChatCompletionOutput = await self.llm.client.achat_completion(
                messages=message_dicts,
                **invocation_params 
            )
            _, token_usage = postprocess_chat_completion_output(
                chat_completion_output, invocation_params
            )

            message: ChatCompletionOutputMessage = chat_completion_output.choices[0].message
            if message.content:
                # if not message content, then it is a tool response and we don't want to strip out anything for tool responses
                message.content = strip_stop_sequences(message.content, invocation_params["stop"])

            chat_result = self._create_chat_result(message, token_usage)

            await handle_async_run_manager(
                run_manager, 
                chat_result.generations[0].message.content, 
                token_usage
            )

            return chat_result
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise            

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)
        message_dicts = self._create_message_dicts(messages, stop)
        completion_id = str(uuid.uuid4())

        try:
            for chat_completion_stream_output in self.llm.client.chat_completion(
                messages=message_dicts,
                **invocation_params,
                stream=True
            ):
                text_chunk, token_usage = postprocess_chat_completion_stream_output(
                    chat_completion_stream_output, invocation_params
                )                    

                text_chunk, found_stop = truncate_at_stop_sequence(
                    text_chunk,
                    invocation_params.get('stop', [])
                )
                
                lc_message = _convert_tgi_message_to_lc_ai_message_chunk(
                    chat_completion_stream_output.choices[0].delta,
                    token_usage
                )
                lc_message.content = text_chunk
                lc_message.additional_kwargs['uuid'] = completion_id

                chat_chunk = ChatGenerationChunk(
                    message=lc_message,
                    generation_info={ 'chunks': [token_usage] }
                )
                
                if run_manager:
                    run_manager.on_llm_new_token(
                        chat_chunk.text, 
                        chat_chunk, 
                        **chat_chunk.generation_info
                    )
                
                if text_chunk:
                    yield chat_chunk

                if found_stop:
                    break
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        invocation_params = self._invocation_params(stop, **kwargs)
        message_dicts = self._create_message_dicts(messages, stop)
        completion_id = str(uuid.uuid4())

        try:
            streaming_completion = await self.llm.client.achat_completion(
                messages=message_dicts,
                **invocation_params,
                stream=True
            )
            async for chat_completion_stream_output in streaming_completion:
                text_chunk, token_usage = postprocess_chat_completion_stream_output(
                    chat_completion_stream_output, invocation_params
                )

                text_chunk, found_stop = truncate_at_stop_sequence(
                    text_chunk,
                    invocation_params.get('stop', [])
                )

                lc_message = _convert_tgi_message_to_lc_ai_message_chunk(
                    chat_completion_stream_output.choices[0].delta,
                    token_usage
                )
                lc_message.content = text_chunk
                lc_message.additional_kwargs['uuid'] = completion_id

                chat_chunk = ChatGenerationChunk(
                    message=lc_message,
                    generation_info={ 'chunks': [token_usage] }
                )

                if run_manager:
                    run_manager.on_llm_new_token(
                        chat_chunk.text, 
                        chat_chunk, 
                        **chat_chunk.generation_info
                    )
                
                if text_chunk:
                    yield chat_chunk

                if found_stop:
                    break
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise            

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        'type': 'function',
                        'function': {'name': tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]['function']['name']
                    != tool_choice['function']['name']
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs['tool_choice'] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)
