from typing import (
    Dict,
    Tuple,
    TypeAlias,
    Any,
    cast,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutputMessage,
    ChatCompletionStreamOutputDelta,
)

ChatCompletionOutputContentLike: TypeAlias = (
    ChatCompletionOutputMessage | ChatCompletionStreamOutputDelta
)

def convert_message_to_chat_message(
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


def convert_tgi_message_to_langchain_message(
    message: ChatCompletionOutputContentLike, token_usage: Dict[str, any]
) -> Tuple[str, Dict[str, Any]]:
    role = message.role
    assert role == "assistant", f"Expected role to be 'assistant', got {role}"
    content = cast(str, message.content)
    if content is None:
        content = ""
    additional_kwargs: Dict = {}
    additional_kwargs["token_usage"] = token_usage
    if tool_calls := message.tool_calls:
        if "arguments" in tool_calls[0]["function"]:
            functions_string = str(tool_calls[0]["function"].pop("arguments"))
            tool_calls[0]["function"]["arguments"] = corrected_functions(
                functions_string
            )
        additional_kwargs["tool_calls"] = tool_calls
    return content, additional_kwargs


def convert_tgi_message_to_lc_ai_message(
    message: ChatCompletionOutputContentLike, token_usage: Dict[str, any]
) -> AIMessage:
    content, additional_kwargs = convert_tgi_message_to_langchain_message(
        message, token_usage
    )
    return AIMessage(content=content, additional_kwargs=additional_kwargs)


def convert_tgi_message_to_lc_ai_message_chunk(
    message: ChatCompletionOutputContentLike, token_usage: Dict[str, any]
) -> AIMessageChunk:
    content, additional_kwargs = convert_tgi_message_to_langchain_message(
        message, token_usage
    )
    additional_kwargs["token_usage"] = {"chunks": [additional_kwargs["token_usage"]]}
    return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
