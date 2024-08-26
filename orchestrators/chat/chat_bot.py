from typing import List
from langchain_core.documents import Document
from orchestrators.doc.redistore import RediStore as VectorStore

class ChatBot:
    def __init__(self):
        self.vector_store = VectorStore.instance()

    def retrieve(self, query) -> List[Document]:
        self.documents = self.vector_store.search(query)
        return self.documents

    def __str__(self) -> str:
        if not self.documents:
            return ''
        raw_data = [document.page_content for document in self.documents]
        return ' '.join(raw_data)