from typing import Dict
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation, LLMResult


def handle_sync_run_manager(
    run_manager: CallbackManagerForLLMRun,
    response_text: str,
    token_usage: Dict[str, any],
) -> None:
    llm_result = LLMResult(
        generations=[[Generation(text=response_text, generation_info=token_usage)]],
        llm_output={"token_usage": token_usage},
    )
    run_manager.on_llm_end(llm_result)


async def handle_async_run_manager(
    run_manager: AsyncCallbackManagerForLLMRun,
    response_text: str,
    token_usage: Dict[str, any],
) -> None:
    llm_result = LLMResult(
        generations=[[Generation(text=response_text, generation_info=token_usage)]],
        llm_output={"token_usage": token_usage},
    )
    await run_manager.on_llm_end(llm_result)
