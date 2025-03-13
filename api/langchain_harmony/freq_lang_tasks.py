import string
from typing import Literal
from collections import defaultdict
from enum import Enum
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ngrams
from nltk.corpus import words, stopwords
from .task import BaseTask
from .nlp_functools import (
    rm_stopwords,
)

_task_type = "freqlang"


class FrequencyType(Enum):
    LOW = "low"
    HIGH = "high"


class FreqHeuristicsTask(BaseTask):
    temperature_range = (0.7, 1.0)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal["low, high"]:
        max_repeats = 5
        tagged_words = pos_tag(word_tokenize(corpus))
        consecutive_pofs_counts = defaultdict(int)
        max_consecutive = defaultdict(lambda: max_repeats)

        max_consecutive.update(
            {
                "IN": 2,
                "CC": 2,
                "UH": 1,
                "PUNCT": 5,
                "SYM": 3,
            }
        )

        puncts = set(string.punctuation) | {"..."}

        last_tag = None
        for word, tag in tagged_words:
            if word in puncts:
                tag_category = "PUNCT"
            else:
                tag_category = tag[:2] if tag[:2] in consecutive_pofs_counts else tag

            if tag_category == last_tag:
                consecutive_pofs_counts[tag_category] += 1
            else:
                consecutive_pofs_counts[tag_category] = 1

            if consecutive_pofs_counts[tag_category] > max_consecutive[tag_category]:
                return FrequencyType.HIGH.value

            last_tag = tag_category

        return FrequencyType.LOW.value


class FreqNLPTask(BaseTask):
    temperature_range = (0.3, 0.69)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal["low, high"]:
        n = 3
        max_repeats = 30
        en_words = set(words.words("en"))
        es_stopwords = set(stopwords.words("spanish"))

        tokens = rm_stopwords(word_tokenize(corpus))

        if any(token.lower() in es_stopwords for token in tokens):
            return FrequencyType.LOW

        non_en_words = [
            token.lower()
            for token in tokens
            if token.isalpha() and token.lower() not in en_words
        ]
        trigrams = list(ngrams(non_en_words, n))

        consecutive_count = 1
        for i in range(len(trigrams) - n):
            if trigrams[i : i + n] == trigrams[i + n : i + 2 * n]:
                consecutive_count += 1
                if consecutive_count >= max_repeats:
                    return FrequencyType.HIGH.value

        return FrequencyType.LOW.value


class FreqMLTask(BaseTask):
    temperature_range = (0.0, 0.29)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal["low, high"]:
        """Implementation coming soon"""
        pass
