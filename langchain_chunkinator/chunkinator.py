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
from .embedding_like import EmbeddingLike

_tokenizer_dir = 'local_tokenizer'

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

    class BinPack:
        def __init__(self, documents: List[Document], embedding: EmbeddingLike):
            self.documents = documents
            self.embedding = embedding
            self.embedding_name = self.embedding.name.split('/')[1]
            self.tokenizer = _tokenizer(self.embedding_name)            

        def chunk(self, documents: List[Document], max_tokens: int) -> List[List[List[str]]]:
            sorted_documents = sorted(documents, 
                key=lambda doc: len(doc.page_content), reverse=True)
            batches = []
            current_batch = []
            current_batch_tokens = 0

            for document in sorted_documents:
                token_content = self.tokenizer.encode(document.page_content)
                token_len = len(document.page_content)

                if current_batch_tokens + token_len <= max_tokens:
                    current_batch.append(token_content)
                    current_batch_tokens += token_len
                
                elif token_len > max_tokens:
                    start = 0
                    while start < token_len:
                        remaining_space = max_tokens - current_batch_tokens

                        if remaining_space > 0:
                            token_slice = token_content[start:start+remaining_space]
                            current_batch.append(token_slice)
                            current_batch_tokens += len(token_slice)
                            start += len(token_slice)
                        else:
                            batches.append(current_batch)
                            current_batch = []
                            current_batch_tokens = 0
                else:
                    batches.append(current_batch)
                    current_batch = [token_content]
                    current_batch_tokens = token_len

            if current_batch:
                batches.append(current_batch)

            return batches

    class Base:
        def __init__(self, documents: List[Document], embedding: EmbeddingLike):
            self.documents = documents
            self.embedding = embedding
            self.embedding_name = self.embedding.name.split('/')[1]
            self.tokenizer = _tokenizer(self.embedding_name)
            self.len_func = lambda text: len(self.tokenizer.encode(text))
            self.request_tokens = int(self.embedding.max_batch_tokens / self.embedding.max_batch_requests)
            self.expo = Chunkinator.Expo(*tuple(map(pow, it.repeat(4,times=4), it.count())))

        def chunk(self) -> List[str]:
            max_chars = self.request_tokens * self.expo.x1
            token_size = max_chars - max_chars % 100
            token_overlap = token_size * 0.01            
            splitter = Splitter(
                length_function=self.len_func, 
                chunk_size=self.request_tokens, 
                chunk_overlap=token_overlap)
            return splitter.split_documents(self.documents)

        def encode(self, document: Document) -> str:
            return self.tokenizer.encode(document.page_content)

        def decode(self, tokens: List[str]) -> List[str]:
            return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)