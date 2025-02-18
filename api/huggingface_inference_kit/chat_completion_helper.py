from typing import Dict, Tuple, Optional
import numpy as np
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import GenerationChunk, Generation, LLMResult

def postprocess_chat_completion_output(
    chat_completion_output: ChatCompletionOutput,
    invocation_params: Dict[str, any],
) -> Tuple[str, Dict[str, any]]:
    completion_candidate = chat_completion_output.choices[0]
    response_text = completion_candidate.message.content
    finish_reason = completion_candidate.finish_reason

    token_usage = {}
    token_usage['finish_reason'] = finish_reason
    # if finish_reason is None, it means streaming and hence we are mid-generation and no finish reason yet
    if finish_reason in ('stop', 'eos_token', 'length') and chat_completion_output.usage:
        token_usage['prompt_tokens'] = chat_completion_output.usage.prompt_tokens
        token_usage['completion_tokens'] = chat_completion_output.usage.completion_tokens
        token_usage['total_tokens'] = chat_completion_output.usage.total_tokens

    if invocation_params.get('logprobs', False) and completion_candidate.logprobs:
        content = completion_candidate.logprobs.content
        logprobs = [logprob.logprob for logprob in content]
        mean_logprob = np.mean(logprobs)
        token_usage['mean_logprob'] = mean_logprob

    return response_text, token_usage

def process_stream_output(
    chat_completion_stream_output: ChatCompletionStreamOutput,
    invocation_params: Dict[str, any],
) -> Tuple[Optional[GenerationChunk], bool, Dict[str, any]]:
    """
    Hugging Face Text Generation Inference (TGI) does not support
    multiple completion candidates for the `chat_completion` task
    as of yet.

    We pull the first and only chat completion stream output choice.
    """
    completion_candidate: ChatCompletionStreamOutputChoice = chat_completion_stream_output.choices[0]
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

    finish_reason = completion_candidate.finish_reason
    token_usage = {}
    token_usage['finish_reason'] = finish_reason
    # if finish_reason is None, it means streaming and hence we are mid-generation and no finish reason yet
    if finish_reason in ('stop', 'eos_token', 'length') and chat_completion_stream_output.usage:
        token_usage['prompt_tokens'] = chat_completion_stream_output.usage.prompt_tokens
        token_usage['completion_tokens'] = chat_completion_stream_output.usage.completion_tokens
        token_usage['total_tokens'] = chat_completion_stream_output.usage.total_tokens

    if invocation_params.get('logprobs', False) and completion_candidate.logprobs:        
        content = completion_candidate.logprobs.content
        logprobs = [logprob.logprob for logprob in content]
        token_usage['mean_logprob'] = float(np.mean(logprobs))

    chunk: Optional[GenerationChunk] = None
    if text:
        chunk = GenerationChunk(
            text=text,
            generation_info=token_usage
        )

    return chunk, (stop_seq_found is not None)
    
def handle_sync_run_manager(
    run_manager: Optional[CallbackManagerForLLMRun],
    response_text: str,
    token_usage: Dict[str, any],
) -> None:
    if run_manager is None:
        return
    llm_result = LLMResult(
        generations=[
            [
                Generation(text=response_text, generation_info=token_usage)
            ]
        ],
        llm_output={'token_usage': token_usage}
    )
    run_manager.on_llm_end(llm_result)

async def handle_async_run_manager(
    run_manager: Optional[AsyncCallbackManagerForLLMRun],
    response_text: str,
    token_usage: Dict[str, any],
) -> None:
    if run_manager is None:
        return
    llm_result = LLMResult(
        generations=[
            [
                Generation(text=response_text, generation_info=token_usage)
            ]
        ],
        llm_output={'token_usage': token_usage}
    )
    await run_manager.on_llm_end(llm_result)