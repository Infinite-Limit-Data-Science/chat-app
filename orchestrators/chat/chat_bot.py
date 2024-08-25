from typing import List
from langchain_core.documents import Document
from orchestrators.doc.redistore import RediStore as VectorStore

class ChatBot:
    def __init__(self, query):
        self.query = query
        self.vector_store = VectorStore.instance()

    def retrieve(self) -> List[Document]:
        return self.vector_store.search(self.query)