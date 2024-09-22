from typing import List, Optional, Any, Iterator, AsyncIterator
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk

class MyChatHuggingFace(ChatHuggingFace):
    """Wrapper to ChatHuggingFace. See https://github.com/langchain-ai/langchain/issues/17779"""
    def _resolve_model_id(self) -> None:
        self.model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Stream using HuggingFaceEndpoint's client
        message_dicts = self._create_message_dicts(messages, stop)
        for chunk in self.llm.client.chat_stream(messages=message_dicts, **kwargs):
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # Convert messages to prompt using the tokenizer
        prompt = self._to_chat_prompt(messages)

        # Prepare invocation parameters for HuggingFaceEndpoint's streaming
        invocation_params = self.llm._invocation_params(stop, **kwargs)

        # Await the coroutine and retrieve the result, assuming it returns an async iterable
        response_stream = await self.llm.async_client.text_generation(
            prompt, **invocation_params, stream=True
        )

        # Iterate over the streamed responses
        async for response in response_stream:
            stop_seq_found: Optional[str] = None

            # Check for stop sequences in the response
            for stop_seq in invocation_params["stop_sequences"]:
                if stop_seq in response:
                    stop_seq_found = stop_seq

            # Identify the portion of the response to yield
            text: Optional[str] = None
            if stop_seq_found:
                text = response[:response.index(stop_seq_found)]
            else:
                text = response

            if text:
                message_chunk = AIMessageChunk(content=text)
                chunk = ChatGenerationChunk(message=message_chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.message.content)

                yield chunk

            if stop_seq_found:
                break