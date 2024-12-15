from typing import List
from collections import Counter
import math
from nltk import ngrams
from nltk.corpus import stopwords

def rm_stopwords(tokens: List[str]) -> List[str]:
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def ngram_freqs(tokens: List[str], n: int = 2) -> Counter[tuple]:    
    ngrams_list = list(ngrams(tokens, n))
    ngram_counts = Counter(ngrams_list)
    return ngram_counts

def compute_tf(term, document):
    term_count = document.count(term)
    total_terms = len(document)
    return term_count / total_terms if total_terms > 0 else 0

def compute_idf(term, corpus):
    num_documents = len(corpus)
    num_documents_with_term = sum(1 for doc in corpus if term in doc)
    return math.log(num_documents / (1 + num_documents_with_term))

def compute_tf_idf(term, document, corpus):
    tf = compute_tf(term, document)
    idf = compute_idf(term, corpus)
    return tf * idf