from typing import List
from langchain_core.documents import Document
from orchestrators.doc.redistore import RediStore as VectorStore
from orchestrators.chat.llm_wrapper import LLMWrapper

class ChatBot:
    def __init__(self, llm: LLMWrapper):
        self.vector_store = VectorStore.instance()
        self.llm_wrapper = llm

    def cosine_similarity(self, query) -> List[Document]:
        self.documents = self.vector_store.similarity_search(query)
        return self.documents

    # retrieve relevant information from a knowledge base (in this case, the VectorStore) based on a query.
    # return a Retriever object that can be used to retrieve relevant vectors from the VectorStore.
    # The Retriever object provides methods for retrieving vectors based on a query
    # useful when you want to integrate the VectorStore with a larger RAG system, such as a language model or a question-answering system.
    def as_retriever(self, query):
        pass

    def __str__(self) -> str:
        if not self.documents:
            return ''
        raw_data = [document.page_content for document in self.documents]
        return ' '.join(raw_data)