import os
import string
import types
from typing import Callable, List, Dict, Tuple, Any, Optional, Self, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from .task import BaseTask
from .freq_lang_tasks import FrequencyType
from .natural_lang_tasks import LanguageType

nltk.data.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data'))
    
class LexicalSoup(BaseModel):
    """
    Lexical Database processing for NLP
    
    Use only lower temperatures in production.
    Higher temperatures are designed for dev testing and heuristic results
    """
    corpus: str | List[str] | Dict[str, str] = Field(description='Corpus which has a vocabulary')
    language: Optional[str] = Field(None, description='Natural language of the corpus')
    punct: set = set(string.punctuation) | {'...'}
    weight: Optional[str] = Field(None, description='The weighted truth of the result')
    temperature: float = Field(description='The temperature to apply to the corpus', ge=0, le=1.0, default=0.01)
    # tagged_sents: Optional[List[str]] = Field(None, description='Sentence tokenizers')
    # tagged_words: Optional[List[Tuple[str, str]]] = Field(None, description='Word tokenizers')

    @field_validator('corpus', mode='before')
    def normalize_corpus(cls, v: str | List[str] | Dict[str, str]) -> str:
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            if all(isinstance(i, str) for i in v):
                return "\n".join(v)
            else:
                raise TypeError('All elements in the list must be strings')
        elif isinstance(v, dict):
            if all(isinstance(k, str) and isinstance(v, str) for k, v in v.items()):
                return "\n".join(value for _, value in v.items())
            else:
                raise TypeError('All keys and values in the dict must be strings.')
        else:
            raise TypeError('`corpus` must be a str, list of str, or dict of str to str.')
        
    @model_validator(mode='after')
    def tokenize_corpus(self) -> Self:
        self.process_language()
        self.process_weight()
        # self.tagged_sents = sent_tokenize(self.corpus)
        # self.tagged_words = [pos_tag(word_tokenize(sentence)) for sentence in self.tagged_sents]
        
        return self

    def process_language(self) -> str:
        task = BaseTask.fetch(temperature=self.temperature, task_type='naturallang')
        self.language = task.perform(self.corpus)
        return self.language
    
    def process_weight(self) -> str:
        task = BaseTask.fetch(temperature=self.temperature, task_type='freqlang')
        self.weight = task.perform(self.corpus)
        return self.weight
    
    @property
    def is_code(self) -> bool:
        return self.language == LanguageType.CODE.value

    @property
    def is_natural(self) -> bool:
        return self.language == LanguageType.NATURAL.value

    @property
    def is_high_frequency(self) -> bool:
        return self.weight == FrequencyType.HIGH.value

    @property
    def is_low_frequency(self) -> bool:
        return self.weight == FrequencyType.LOW.value