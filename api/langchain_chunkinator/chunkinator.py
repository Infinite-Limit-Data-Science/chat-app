from __future__ import annotations

import os
import json
import importlib
from typing import List
from collections import namedtuple
import itertools as it
from functools import lru_cache
from transformers import PreTrainedTokenizerBase
from langchain.text_splitter import RecursiveCharacterTextSplitter as Splitter
from langchain_core.documents import Document
from ..langchain_doc.vector_stores import AbstractVectorStore
from .embedding_like import EmbeddingLike

_tokenizer_dir = 'local_tokenizer'

_MAX_BATCH_TOKENS = 32768

_MAX_BATCH_REQUESTS = 64

@lru_cache(maxsize=1)
def _tokenizer(embedding_name: str):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),_tokenizer_dir, embedding_name, 'tokenizer_config.json'), 'r') as f:
        tokenizer_config = json.load(f)
        tokenizer_class_name: PreTrainedTokenizerBase = tokenizer_config['tokenizer_class']
        transformers_module = importlib.import_module('transformers')
        cls = getattr(transformers_module, tokenizer_class_name)
        tokenizer = cls.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)),_tokenizer_dir, embedding_name))
        return tokenizer

class Chunkinator:
    Expo = namedtuple('Expo', ['x0', 'x1', 'x2', 'x3'])

    class Base:
        def __init__(self, documents: List[Document], vector_store: AbstractVectorStore):
            self.documents = documents
            self.embedding = vector_store.embeddings
            self.embedding_name = vector_store.embeddings_model_config.name.split('/')[1]
            self.tokenizer = _tokenizer(self.embedding_name)
            self.len_func = lambda text: len(self.tokenizer.encode(text))
            self.request_tokens = int(_MAX_BATCH_TOKENS / _MAX_BATCH_REQUESTS)
            self.expo = Chunkinator.Expo(*tuple(map(pow, it.repeat(4,times=4), it.count())))

        def chunk(self) -> List[str]:
            max_chars = self.request_tokens * self.expo.x1
            token_size = max_chars - max_chars % 100
            token_overlap = token_size * 0.01

            splitter = Splitter(
                length_function=self.len_func, 
                chunk_size=(self.request_tokens + 100), 
                chunk_overlap=token_overlap)
            return splitter.split_documents(self.documents)

        def encode(self, document: Document) -> str:
            return self.tokenizer.encode(document.page_content)

        def decode(self, tokens: List[str]) -> List[str]:
            return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)