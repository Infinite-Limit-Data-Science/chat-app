import logging
from typing import List, Optional, Any, Iterator, AsyncIterator
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from orchestrators.chat.utils.my_llms import assert_valid_hf_llm as validate_llm

class MyChatHuggingFace(ChatHuggingFace):
    """Wrapper to ChatHuggingFace. See https://github.com/langchain-ai/langchain/issues/17779"""
    def _resolve_model_id(self) -> None:
        pass
    
    @property
    def sequence_tokens(self) -> List[str]:
        return [self.tokenizer.bos_token, self.tokenizer.eos_token]

    @staticmethod
    def to_openai(input: dict) -> dict:
        output = {
            'max_tokens': input.get('max_new_tokens'),
            'top_p': input.get('top_p'),
            'temperature': input.get('temperature'),
            'seed': input.get('seed'),
            'logit_bias': input.get('logit_bias', [])
        }
    
        return output


    def handle_smarty_pants():
        """Custom logic to determine model filling"""
        pass

    @validate_llm
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        stop_tokens = stop or self.sequence_tokens

        message_dicts = self._create_message_dicts(messages, stop)
        invocation_params = self.to_openai(self.llm._invocation_params(stop, **kwargs))

        answer = self.llm.client.chat_completion(messages=message_dicts, stream=True, stop=stop_tokens, **invocation_params)

        for response_chunk in answer:
            text = response_chunk['choices'][0]['delta']['content']
            
            if any(stop_token in text for stop_token in stop_tokens):
                text = text.split(next(stop_token for stop_token in stop_tokens if stop_token in text))[0]
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                break
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    @validate_llm
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        stop_tokens = stop or self.sequence_tokens
        
        message_dicts = self._create_message_dicts(messages, stop)
        invocation_params = self.to_openai(self.llm._invocation_params(stop, **kwargs))

        response_stream = await self.llm.async_client.chat_completion(message_dicts, stream=True, stop=stop_tokens, **invocation_params)

        async for response in response_stream:
            text = response["choices"][0]['delta']['content']

            if any(stop_token in text for stop_token in stop_tokens):
                text = text.split(next(stop_token for stop_token in stop_tokens if stop_token in text))[0]
                message_chunk = AIMessageChunk(content=text)
                chunk = ChatGenerationChunk(message=message_chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.message.content)

                yield chunk
                break
            else:
                message_chunk = AIMessageChunk(content=text)
                chunk = ChatGenerationChunk(message=message_chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk.message.content)

                yield chunk

    def __str__(self):
        return str(self.tokenizer.build_inputs_with_special_tokens([self.tokenizer.encode(self.__class__.__name__)]))