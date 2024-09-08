from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatMessagePromptTemplate
from orchestrators.doc.redistore import RediStore as VectorStore
# from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.model_proxy import ModelProxy
from orchestrators.chat.messages.message_history import MessageHistory

class ChatBot(AbstractBot):
    def __init__(self, llm: ModelProxy, message_history: MessageHistory):
        self.vector_store = VectorStore
        self.message_history = message_history
        self.llm = llm

    def chat_prompt_template(self):
        self.prompt = ChatMessagePromptTemplate.from_template(
            """
                Answer the following question based only on the provided context:
                <context>
                {context}
                </context>
            """
        )

    
    def chain(self):
        self.prompt|self.llm|self.parser

    def cosine_similarity(self, query) -> List[Document]:
        """Perform a direct cosine similarity search on the VectorStore"""
        self.documents = self.vector_store.similarity_search(query)
        return self.documents

    def retrieval_chain():
        """Retrieve relevant vectors from VectorStore, useful in RAG and Q&A""" 

    def invoke():
        pass

    def __str__(self) -> str:
        if not self.documents:
            return ''
        raw_data = [document.page_content for document in self.documents]
        return ' '.join(raw_data)