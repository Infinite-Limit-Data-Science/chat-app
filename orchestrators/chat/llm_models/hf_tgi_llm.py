import logging
import os
import sys
from typing import Any
from dataclasses import dataclass
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.outputs import LLMResult
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.my_chat_huggingface import MyChatHuggingFace

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
new_folder_path = os.path.join(current_directory, 'local_tokenizer')
tokenizer = AutoTokenizer.from_pretrained(new_folder_path)

@dataclass(kw_only=True, slots=True)
class HFTGI(LLM):
    def __post_init__(self) -> None:
        streaming_handler = StreamingStdOutCallbackHandler()
        callbacks = [streaming_handler]
        llm = HuggingFaceEndpoint(
            streaming=True, 
            callbacks=callbacks, 
            **{'endpoint_url': self.endpoint['url'], **self.parameters, 'server_kwargs': dict(self.server_kwargs)})
        
        chat = MyChatHuggingFace(llm=llm, tokenizer=tokenizer, model_id=self.name)
        self.endpoint_object = chat

        summary_llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint['url'],
            max_new_tokens=10,
            temperature=0.1,
            repetition_penalty=1.2,
            task='summarization',
            server_kwargs=dict(self.server_kwargs)
        )
        self.summary_object = MyChatHuggingFace(llm=summary_llm, tokenizer=tokenizer, model_id=self.name)