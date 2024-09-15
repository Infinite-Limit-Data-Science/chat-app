from typing import List
from langchain_core.documents import Document
from orchestrators.doc.redistore import RediStore as VectorStore
# from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.model_proxy import ModelProxy
from orchestrators.chat.messages.prompt_template import PromptTemplate
from orchestrators.chat.messages.message_history import (
    MongoMessageHistory,
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    Sequence,
)

class ChatBot(AbstractBot):
    def __init__(self, prompt: str, llm: ModelProxy, message_history: MongoMessageHistory, vector_options: dict):
        self._prompt = PromptTemplate(prompt)
        self._llm = llm
        self._vector_store = VectorStore(vector_options['uuid'], vector_options['conversation_id'])
        self._message_history = message_history

    async def add_system_message(self, message: str) -> SystemMessage:
        """Add system message to data store"""
        system_message = await self._message_history.system(message)
        return system_message

    async def add_human_message(self, message: dict) -> HumanMessage:
        """add human message to data store"""
        human_message = await self._message_history.human(message)
        return human_message

    async def add_ai_message(self, message_schema: dict) -> AIMessage:
        """Add ai message to data store"""
        ai_message = await self._message_history.ai(message_schema)
        self._message_history.messages = ai_message
        return ai_message
    
    async def add_bulk_messages(self, messages: Sequence[BaseMessage]) -> True:
        """Store messages in bulk in data store"""
        return await self._message_history.bulk_add(messages)
    
    def cosine_similarity(self, query) -> List[Document]:
        """Perform a direct cosine similarity search on the VectorStore"""
        """This method is here for completeness but generally use runnable instead"""
        self.documents = self._vector_store.similarity_search(query)
        return self.documents

    # TODO: add trimmer runnable  
    async def runnable(self, **kwargs) -> AIMessage:
        """Invoke the chain"""
        chain = self._prompt.runnable() | self._llm.runnable() | self._vector_store.runnable()
        chain_with_history = self._message_history.runnable(chain)
        ai_response = await chain_with_history.ainvoke({'question': kwargs['message']}, {'session_id': kwargs['session_id']})
        return ai_response
    chat = runnable