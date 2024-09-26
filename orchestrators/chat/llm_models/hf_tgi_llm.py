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

class InspectionStreamingHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            **kwargs (Any): Additional keyword arguments.
        """
        logging.warning(f'WHAT IS THE CURRENT TOKEN {token}')
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response from the LLM.
            **kwargs (Any): Additional keyword arguments.
        """
        logging.warning(f'WHAT IS THE LLM RESULT {response}')

@dataclass(kw_only=True, slots=True)
class HFTGI(LLM):
    def __post_init__(self) -> None:
        streaming_handler = InspectionStreamingHandler()
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
        self.summary_object = MyChatHuggingFace(llm=summary_llm, model_id=self.name)