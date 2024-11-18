from collections import Counter
import string
from typing import Literal
from enum import Enum
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk import Tree
from .task import BaseTask
from .nlp_functools import ngram_freqs

_task_type = 'naturallang'

class LanguageType(Enum):
    CODE = 'code'
    NATURAL = 'natural'

code_tree = Tree('CodePatterns', [
    Tree('FunctionDefinitions', [
        ('(', ')', 'FunctionCall'),
        ('function', '(', 'FunctionDefinition_JS'),
        ('def', '(', 'FunctionDefinition_Python'),
        ('fn', '(', 'FunctionDefinition_Rust'),
        ('func', '(', 'FunctionDefinition_Go'),
        ('class', '{', 'ClassDefinition_Generic'),
        ('public', 'class', 'ClassDefinition_Public'),
        ('private', 'class', 'ClassDefinition_Private')
    ]),
    
    Tree('ConditionalsAndLoops', [
        ('if', '(', 'IfStatement'),
        ('else', '{', 'ElseStatement_Block'),
        ('for', '(', 'ForLoop'),
        ('while', '(', 'WhileLoop'),
        ('elif', '(', 'ElifStatement_Python'),
        ('else', ':', 'ElseStatement_Python')
    ]),
    
    Tree('ComparisonAndAssignmentOperators', [
        ('=', '=', 'Assignment_Equality'),
        ('==', 'True', 'EqualityCheck_Boolean'),
        ('===', 'true', 'StrictEqualityCheck_JS'),
        ('!=', 'false', 'NotEqualityCheck_Boolean'),
        ('let', '=', 'VariableDeclaration_JS'),
        ('const', '=', 'ConstantDeclaration_JS'),
        ('var', '=', 'VariableDeclaration_Generic'),
        (':=', '=', 'WalrusOperator_Python')
    ]),
    
    Tree('BlockDelimitersAndBraces', [
        ('{', '}', 'BlockDelimiters'),
        ('[', ']', 'ArrayDelimiters'),
        ('=>', '{', 'ArrowFunction_Block_JS')
    ]),
    
    Tree('LogicalAndArithmeticOperators', [
        ('+', '=', 'AdditionAssignment'),
        ('-', '=', 'SubtractionAssignment'),
        ('*', '=', 'MultiplicationAssignment'),
        ('/', '=', 'DivisionAssignment'),
        ('&&', '||', 'LogicalOperators_AndOr'),
        ('&', '&', 'BitwiseAndOperator'),
        ('|', '|', 'BitwiseOrOperator')
    ])
])

class NaturalHeuristicsTask(BaseTask):
    temperature_range = (0.7, 1.0)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal['code', 'natural']:
        en_words = set(words.words())
        tokens = word_tokenize(corpus)

        en_word_count = sum(1 for token in tokens if token.lower() in en_words)
        non_en_count = len(tokens) - en_word_count

        if non_en_count > en_word_count:
            return LanguageType.CODE.value
        
        return self.statistical_freq(corpus, tokens)

    def statistical_freq(self, corpus: str) -> Literal['code', 'natural']:
        char_counts = Counter(corpus)
        total_chars = sum(char_counts.values())
        symbol_chars = set('{}[]()<>;=+-*/')

        symbol_count = sum(char_counts[char] for char in symbol_chars)
        alphabetic_count = sum(char_counts[char] for char in string.ascii_letters)

        symbol_ratio = symbol_count / total_chars
        alphabetic_ratio = alphabetic_count / total_chars

        if symbol_ratio > 0.1 and alphabetic_ratio < 0.7:
            return LanguageType.CODE.value
        else:
            return LanguageType.NATURAL.value

class NaturalNLPTask(BaseTask):
    temperature_range = (0.3, 0.69)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal['code', 'natural']:
        tokens = word_tokenize(corpus)
        bigram_freqs = ngram_freqs(tokens)
        heuristic_weight = 5
        weighted_sum = 0

        category_weights = {
            'FunctionDefinitions': 1.5,
            'ConditionalsAndLoops': 1.5,
            'ComparisonAndAssignmentOperators': 1.5,
            'BlockDelimitersAndBraces': 1.0,
            'LogicalAndArithmeticOperators': 1.0,
        }

        for bigram, count in bigram_freqs.items():
            cat = self.cat_from_tree(bigram)
            if cat:
                weighted_sum += count * category_weights[cat]

        if weighted_sum > heuristic_weight:
            return LanguageType.CODE.value
        else:
            return LanguageType.NATURAL.value

    def cat_from_tree(self, ngram):
        for category in code_tree:
            for pattern in category:
                if (pattern[0], pattern[1]) == ngram:
                    return category.label()
        return None

class NaturalMLTask(BaseTask):
    temperature_range = (0.0, 0.29)
    task_type = _task_type

    def perform(self, corpus: str) -> Literal['code', 'natural']:
        import logging
        logging.warning('Implementation coming soon')
        pass