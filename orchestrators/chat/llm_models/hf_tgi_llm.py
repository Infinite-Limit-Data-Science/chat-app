import logging
import os
from dataclasses import dataclass
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.outputs import LLMResult
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.chat.llm_models.my_chat_huggingface import MyChatHuggingFace

_current_file_path = os.path.abspath(__file__)
_current_directory = os.path.dirname(_current_file_path)
_new_folder_path = os.path.join(_current_directory, 'local_tokenizer')

_summarizable_models = {'text-generation'}

@dataclass(kw_only=True, slots=True)
class HFTGI(LLM):
    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(_new_folder_path, self.name.split(os.path.sep)[-1]))
        return tokenizer

    def load_summarizable_model(self):
        summary_llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint['url'],
            max_new_tokens=10,
            temperature=0.1,
            repetition_penalty=1.2,
            task='summarization',
            server_kwargs=dict(self.server_kwargs)
        )
        self.summary_object = MyChatHuggingFace(llm=summary_llm, tokenizer=self._load_tokenizer(), model_id=self.name)

    def __post_init__(self) -> None:
        streaming_handler = StreamingStdOutCallbackHandler()
        callbacks = [streaming_handler] if self.stream else []
        llm = HuggingFaceEndpoint(
            streaming=self.stream,
            callbacks=callbacks, 
            **{'endpoint_url': self.endpoint['url'], **self.parameters, 'server_kwargs': dict(self.server_kwargs)})
        chat = MyChatHuggingFace(llm=llm, tokenizer=self._load_tokenizer(), model_id=self.name)
        self.endpoint_object = chat
        if _summarizable_models:
             self.load_summarizable_model()