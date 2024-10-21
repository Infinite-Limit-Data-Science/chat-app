from __future__ import annotations

import logging
import os
import json
import importlib
from typing import List
from enum import Enum
from functools import lru_cache
from transformers import PreTrainedTokenizerBase
from langchain.text_splitter import RecursiveCharacterTextSplitter as Splitter
from langchain_core.documents import Document
from orchestrators.doc.embedding_models.embedding import BaseEmbedding

_tokenizer_dir = 'local_tokenizer'

@lru_cache(maxsize=1)
def _tokenizer(embedding_name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),_tokenizer_dir, embedding_name, 'tokenizer_config.json'), 'r') as f:
        tokenizer_config = json.load(f)
        tokenizer_class_name: PreTrainedTokenizerBase = tokenizer_config['tokenizer_class']
        transformers_module = importlib.import_module('transformers')
        cls = getattr(transformers_module, tokenizer_class_name)
        tokenizer = cls.from_pretrained(os.path.join(os.path.dirname(os.path.abspath(__file__)),_tokenizer_dir, embedding_name))
        return tokenizer

class Chunkinator2:   
    class Document:
        def __init__(self, document: Document, embedding_name: str):
            self.document = document
            self.embedding_name = embedding_name

        @property
        def as_tokens(self) -> List[str]:
            return _tokenizer(self.embedding_name).encode(self.document.page_content)

        def __len__(self) -> int:
            return len(self.as_tokens)

        def __lt__(self, other) -> bool:
            return len(self) < len(other)

        def __le__(self, other) -> bool:
            return len(self) <= len(other)

        def __gt__(self, other) -> bool:
            return len(self) > len(other)

        def __ge__(self, other) -> bool:
            return len(self) >= len(other)
        Input = Document
    
    class CharFactor(Enum):
        x: int = 4
        x2: int = 4**2
        x3: int = 4**3

    class BinPack:
        @staticmethod
        def pack(chunks: List[Chunkinator2.Document], max_tokens: int) -> List[List[List[str]]]:
            sorted_chunks = sorted(chunks, reverse=True)
            batches = []
            current_batch = []
            current_batch_tokens = 0

            for chunk in sorted_chunks:
                token_content = chunk.as_tokens
                token_len = len(chunk)

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

    class KMeansPack:
        @staticmethod
        def pack(chunks: List[Chunkinator2.Document], max_tokens: int) -> List[List[List[str]]]:
            from sklearn.cluster import KMeans
            import numpy as np

            token_lengths = np.array([len(chunk) for chunk in chunks]).reshape(-1, 1)
            total_tokens = sum(token_lengths)[0]

            num_clusters = int(np.ceil(total_tokens / max_tokens))

            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(token_lengths)

            cluster_labels = kmeans.labels_

            batches = [[] for _ in range(num_clusters)]
            current_batch_tokens = [0] * num_clusters

            for idx, chunk in enumerate(chunks):
                cluster_id = cluster_labels[idx]
                token_content = chunk.as_tokens
                token_len = len(chunk)

                if current_batch_tokens[cluster_id] + token_len <= max_tokens:
                    batches[cluster_id].append(token_content)
                    current_batch_tokens[cluster_id] += token_len
                else:
                    start = 0
                    while start < token_len:
                        remaining_space = max_tokens - current_batch_tokens[cluster_id]

                        if remaining_space > 0:
                            token_slice = token_content[start:start+remaining_space]
                            batches[cluster_id].append(token_slice)
                            current_batch_tokens[cluster_id] += len(token_slice)
                            start += len(token_slice)
                        else:
                            cluster_id += 1
                            batches.append([])
                            current_batch_tokens.append(0)

            return batches

    class Engine:
        def __init__(self, corpus: List[Document], embedding: BaseEmbedding):
            self.corpus = corpus
            self.embedding = embedding
            self.embedding_name = self.embedding.name.split('/')[1]
            self.bin_pack = Chunkinator2.BinPack()
            self.k_means_pack = Chunkinator2.KMeansPack()

        def encode(self) -> List[List[List[str]]]:
            request_tokens = int(self.embedding.max_batch_tokens / self.embedding.max_batch_requests)
            max_chars = request_tokens * Chunkinator2.CharFactor.x.value
            chunk_size = max_chars - max_chars % 100
            chunk_overlap = chunk_size * 0.1
            
            text_splitter = Splitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            chunks = text_splitter.split_documents(self.corpus)
            chunks = [Chunkinator2.Document(chunk, self.embedding_name) for chunk in chunks]            
            result = self.bin_pack.pack(chunks, request_tokens)
            
            total_tokens_per_batch = [sum(len(token_slice) for token_slice in batch) for batch in result]
            for idx, tokens in enumerate(total_tokens_per_batch):
                logging.warning(f'Batch {idx + 1} has {tokens} tokens.')

            return result

        def decode(self, batches: List[List[List[str]]]) -> List[str]:
            decoded_texts = [
                " ".join([
                    _tokenizer(self.embedding_name).decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for tokens in batch
                ])
                for batch in batches
            ]
            
            return decoded_texts

        def chunk(self) -> List[str]:
            encoded = self.encode()
            decoded = self.decode(encoded)
            return decoded