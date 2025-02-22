from typing import Dict, Tuple, Optional, TypeAlias, List
import numpy as np
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta
)

ChatCompletionOutputLike: TypeAlias = ChatCompletionOutput | ChatCompletionStreamOutput
ChatCompletionOutputChoiceLike: TypeAlias = ChatCompletionOutputComplete | ChatCompletionStreamOutputChoice

def _process_token_usage(
    output: ChatCompletionOutputLike,
    completion_candidate: ChatCompletionOutputChoiceLike,
    invocation_params: Dict[str, any]
) -> Dict[str, any]:
    finish_reason = completion_candidate.finish_reason

    token_usage = {}
    token_usage['finish_reason'] = finish_reason
    # if finish_reason is None, it means streaming and hence we are mid-generation and no finish reason yet
    if finish_reason in ('stop', 'eos_token', 'length') and output.usage:
        token_usage['prompt_tokens'] = output.usage.prompt_tokens
        token_usage['completion_tokens'] = output.usage.completion_tokens
        token_usage['total_tokens'] = output.usage.total_tokens

    if invocation_params.get('logprobs', False) and completion_candidate.logprobs:
        content = completion_candidate.logprobs.content
        logprobs = [logprob.logprob for logprob in content]
        mean_logprob = np.mean(logprobs)
        token_usage['mean_logprob'] = mean_logprob
    
    return token_usage

def postprocess_chat_completion_output(
    chat_completion_output: ChatCompletionOutput,
    invocation_params: Dict[str, any],
) -> Tuple[str, Dict[str, any]]:
    # we specify first choice only because Hugging Face TGI doesn't support multiple candidate completions
    completion_candidate: ChatCompletionOutputComplete = chat_completion_output.choices[0]
    message: ChatCompletionOutputMessage = completion_candidate.message
    response_text = message.content or ""

    token_usage = _process_token_usage(
        chat_completion_output, 
        completion_candidate,
        invocation_params
    )

    return response_text, token_usage

def postprocess_chat_completion_stream_output(
    chat_completion_stream_output: ChatCompletionStreamOutput,
    invocation_params: Dict[str, any],
) -> Tuple[str, Dict[str, any]]:
    completion_candidate: ChatCompletionStreamOutputChoice = chat_completion_stream_output.choices[0]
    delta: ChatCompletionStreamOutputDelta = completion_candidate.delta
    response_text = delta.content or ""

    token_usage = _process_token_usage(
        chat_completion_stream_output, 
        completion_candidate,
        invocation_params
    )

    return response_text, token_usage

def strip_stop_sequences(text: str, stop_sequences: List[str]) -> str:
    """
    Removes stop sequences at end

    So in example, "Hello STOP somewhere", string does not end with "STOP", 
    so it does not remove anything, returning "Hello STOP somewhere".
    """
    for stop_seq in stop_sequences:
        if text.endswith(stop_seq):
            text = text[: -len(stop_seq)]
    return text

def truncate_at_stop_sequence(text: str, stop_sequences: List[str]) -> Tuple[str, bool]:
    """
    Removes everything after stop sequence, including stop sequence anywhere

    So in example, "Hello STOP somewhere", _truncate_at_stop_sequence sees 
    "STOP" in the middle and cuts off everything from "STOP" onward, returning 
    "Hello ".
    """        
    stop_seq_found: Optional[str] = None
    for stop_seq in stop_sequences:
        if stop_seq in text:
            stop_seq_found = stop_seq
            break

    if stop_seq_found:
        idx = text.index(stop_seq_found)
        text = text[:idx]
    return text, stop_seq_found