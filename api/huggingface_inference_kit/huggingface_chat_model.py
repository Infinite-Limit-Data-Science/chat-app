from dataclasses import dataclass
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
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import model_validator
from typing_extensions import Self

from ..huggingface_inference_kit.huggingface_llm import HuggingFaceLLM
from ..huggingface_inference_kit.huggingface_transformer_tokenizers import get_tokenizer_class_by_prefix

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""

class HuggingFaceChatModel(BaseChatModel):
    llm: HuggingFaceLLM
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None
    model_name: Optional[str] = None

    @model_validator(mode='after')
    def validate_environment(self) -> Self:
        if not isinstance(self.llm, HuggingFaceLLM):
            raise TypeError(
                "Expected llm to be one of type `HuggingFaceLLM`"
                f"received {type(self.llm)}"
            )

        tokenizer_name = self.llm.tgi_config.name or self.model_name
        if not tokenizer_name:
            raise TypeError(
                "Expected model name to be defined"
                f"received {type(tokenizer_name)}"
            )
        self.tokenizer = get_tokenizer_class_by_prefix(tokenizer_name)

        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages, stop)
        answer = self.llm.client.chat_completion(messages=message_dicts, **kwargs)
        return self._create_chat_result(answer)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)



from langchain_huggingface import ChatHuggingFace
