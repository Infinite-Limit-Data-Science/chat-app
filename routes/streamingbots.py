import os
import asyncio
from typing import Any, List, Optional, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from transformers import AutoTokenizer
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from typing import Iterator, AsyncIterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from fastapi import APIRouter

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all the question to the best of your ability."),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}")
    ]
)

# See https://github.com/langchain-ai/langchain/issues/17779
class MyChatHuggingFace(ChatHuggingFace):
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
        
def run_llm() -> RunnableWithMessageHistory:
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    new_folder_path = os.path.join(current_directory, 'local_tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(new_folder_path)

    streaming_handler = StreamingStdOutCallbackHandler()
    callbacks = [streaming_handler]
    llm = HuggingFaceEndpoint(
        streaming=True, 
        callbacks=callbacks, 
        endpoint_url="http://100.28.34.190:8080/",
        max_new_tokens=512,
        temperature=0.01,
        repetition_penalty=1.2,
    )

    chat = MyChatHuggingFace(llm=llm, tokenizer=tokenizer)
    chain = prompt | chat

    store={}

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in store:
            store[session_id]=ChatMessageHistory()
        return store[session_id]
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key='input', 
        history_messages_key='chat_history')
    
    return chain_with_history

router = APIRouter(
    prefix='/streamingbots', 
    tags=['streamingbot'],
)

@router.post("/stream")
async def stream_llm(request: Request):
    # Extract the input from the request, if necessary
    # input_data = await request.json()
    user_input = {}.get('input', 'What is Langchain?')

    chain_with_history = run_llm()
    # Define the FastAPI streaming generator
    async def llm_stream():
        stop_token = "<|eot_id|>"
        async for s in chain_with_history.astream(
                        {'input': user_input},
                        config={'session_id': 'chat1'}):
            # Remove the stop token if it's present
            if stop_token in s.content:
                s.content = s.content.replace(stop_token, "")
            yield s.content

    return StreamingResponse(llm_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no"})

# curl -N -X POST http://localhost:8000/stream