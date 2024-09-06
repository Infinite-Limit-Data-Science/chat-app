import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatMessagePromptTemplate
from orchestrators.doc.redistore import RediStore as VectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.chat.abstract_chat import AbstractChat

class LLMChat(AbstractChat):
    def __init__(self, llm):
        self.prompt 
        self.vector_store = VectorStore
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

    def chat_prompt_template_(self):
        generic_template = os.environ['DEFAULT_PROMPT']
        [('system', generic_template), 'user', "{text}"]

    
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