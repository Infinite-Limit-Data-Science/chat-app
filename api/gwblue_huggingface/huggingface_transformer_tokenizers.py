import os
import json
import re
import importlib
from pathlib import Path
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Protocol, Annotated, Tuple
from typing_extensions import Doc
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import numpy as np

T = TypeVar("T")
U = TypeVar("U")

_UPPER_BOUNDS = 24_256

"""
Tokenizers convert raw text into a sequence of token IDs - 
the "words" or "subwords" the model actually consumes.

Transformer models understand these integer IDs representing
tokens.

We cache the Tokenizers
"""
TRANSFORMER_TOKENIZER_CACHE = "transformers/tokenizers/cache"


def transform_name(s: str) -> str:
    parts = s.split("/", 1)
    remainder = parts[1] if len(parts) > 1 else parts[0]

    result = re.sub(r"[.\-]", "", remainder)
    return result


def get_tokenizer_class_by_prefix(prefix: str):
    class_name = f"{transform_name(prefix)}PretrainedTokenizer"
    try:
        return globals()[class_name]
    except KeyError:
        raise ImportError(f"Could not find class {class_name} in {__name__}")


def load_local_tokenizer_and_config(name: str) -> Tuple[PreTrainedTokenizerBase, dict]:
    """
    Load a tokenizer + its config from the local cache directory,
    falling back to AutoTokenizer if the explicit class fails.
    Returns (tokenizer, config dict).
    """
    current_file_dir = Path(__file__).resolve().parent
    local_dir = current_file_dir / TRANSFORMER_TOKENIZER_CACHE / name

    tokenizer_config_path = local_dir / "tokenizer_config.json"
    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)
    tokenizer_class_name: str = tokenizer_config.get("tokenizer_class", "")

    tokenizer = None
    try:
        transformers_module = importlib.import_module("transformers")
        cls = getattr(transformers_module, tokenizer_class_name)
        tokenizer = cls.from_pretrained(
            local_dir, local_files_only=True, trust_remote_code=True
        )
    except Exception as e:
        print(
            f"Failed to load tokenizer class '{tokenizer_class_name}' from config. "
            f"Falling back to AutoTokenizer.\nError: {e}"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir, local_files_only=True, trust_remote_code=True
        )

    config_path = local_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return tokenizer, config


class BaseLocalTokenizer(ABC):
    def __init__(self, name: str):
        self._name = name
        self._tokenizer = None
        self._sequence_length = None
        self._dimensions = None
        self._initialized = False

    def _initialize_if_needed(self):
        if not self._initialized:
            tokenizer, config = load_local_tokenizer_and_config(self._name)

            self._tokenizer = tokenizer
            self._sequence_length = tokenizer.model_max_length
            self._dimensions = self.extract_dimensions(config)

            self._initialized = True

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._initialize_if_needed()
        return self._tokenizer

    @property
    def sequence_length(self) -> int:
        """
         sequence length refers to the maximum number of tokens (or positions) 
         a model can handle in one forward pass of the neural network, where
         the input (sequence of tokens) flows through the input layer, then all
         the hidden layers, and finally the final output layer. The output being
         next-token preditions.


        """
        self._initialize_if_needed()
        return self._sequence_length

    @property
    def dimensions(self) -> int:
        self._initialize_if_needed()
        return self._dimensions

    @property
    def positional_embeddings(self) -> Tuple[int, int]:
        """
        Unlike an RNN that processes tokens one by one in order,
        a Transformer sees the entire sequence at once and needs
        a way to encode the position of each token.

        Each row corresponds to one "position" in the input.
        Each row is a length-dimensions vector
        vllm2vec example:
        [131072, 3072]
        """
        self._initialize_if_needed()
        return np.random.randn(self._sequence_length, self._dimensions)

    @property
    def empirical_sequence_length(self) -> int:
        """
        Running the model at the max sequence length can produce
        degraded performance for extremely large inputs, or it might
        not be thoroughly tested at the full theoretical limit.

        Use the empirical sequence length as the recommended max length
        that is smaller than the max positional embeddings.
        """
        self._initialize_if_needed()
        models_list = json.loads(os.environ["MODELS"])
        embedding_models_list = json.loads(os.environ["EMBEDDING_MODELS"])

        full_models = models_list + embedding_models_list
        empirical_sequence_len = next(
            (
                model["sequence_length"]
                for model in full_models
                if model["name"] == self._name
            ),
            None,
        )
        if not empirical_sequence_len:
            return self._sequence_length

        return empirical_sequence_len

    @property
    def recommended_token_batch(self) -> int:
        """
        How many tokens can be combined across all user requests in a
        single GPU forward pass. if multiple user requests come in
        simultaneously, their combined input lengths cannot exeed
        this limit.

        This limit is about throughput and GPU memory management.
        It doesn't change the model's internal sequence length.

        Each token in the input is assigned a position index (0, 1, 2,
        …), and for each token-position pair, the model looks up the
        corresponding row in the positional embedding table. However,
        this happens per sequence in the batch.

        If you have one sequence of length 100 tokens, those tokens map
        to positions 0 through 99. Each position looks up a row in the
        table.

        If you batch 5 sequences each of length 100 tokens, the model
        processes them as a tensor of shape [5,100].
        """
        if self.empirical_sequence_length > _UPPER_BOUNDS:
            return _UPPER_BOUNDS
        else:
            return self.empirical_sequence_length

    @abstractmethod
    def extract_dimensions(self, config: dict) -> int: ...


class BgeLargePretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "baai_bge_large_en_v1.5")

    def extract_dimensions(self, config: dict) -> int:
        return config.get("hidden_size")


class NomicPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "nomic_ai_nomic_embed_text_v1.5")

    def extract_dimensions(self, config: dict) -> int:
        return config.get("n_embd")


class MetaLlama3170BInstructPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "meta_llama_3_1_70B_Instruct")

    def extract_dimensions(self, config: dict) -> int:
        return config.get("hidden_size")


class Llama3290BVisionInstructPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "meta_llama_3_2_90B_Vision_Instruct")

    def extract_dimensions(self, config: dict) -> int:
        text_config = config.get("text_config", {})
        return text_config.get("hidden_size")


class Llama3211BVisionInstructPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "meta_llama_3_2_11B_Vision_Instruct")

    def extract_dimensions(self, config: dict) -> int:
        text_config = config.get("text_config", {})
        return text_config.get("hidden_size")


class LlamaGuard38BPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "meta_llama_guard_3_8B")

    def extract_dimensions(self, config: dict) -> int:
        text_config = config.get("text_config", {})
        return text_config.get("hidden_size")


class LlamaGuard311BVisionPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "meta_llama_guard_3_11B")

    def extract_dimensions(self, config: dict) -> int:
        text_config = config.get("text_config", {})
        return text_config.get("hidden_size")


class VLM2VecFullPretrainedTokenizer(BaseLocalTokenizer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "tiger_lab_vlm2vec_full")

    def extract_dimensions(self, config: dict) -> int:
        return config.get("hidden_size")
