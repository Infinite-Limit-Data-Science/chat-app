import logging
from typing import List
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from orchestrators.doc.redistore import RediStore as VectorStore
# from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.chat.messages.prompts import registry
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.model_proxy import ModelProxy
from orchestrators.chat.messages.message_history import (
    MongoMessageHistory,
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    Sequence,
)

class ChatBot(AbstractBot):
    def __init__(self, llm: ModelProxy, message_history: MongoMessageHistory, vector_options: dict):
        self._llm = llm
        self._message_history = message_history
        self._vector_store = VectorStore(vector_options['uuid'], vector_options['conversation_id'])
        self._rag_prompt_runnable = registry['rephrased_runnable']()

    async def add_system_message(self, message: str) -> SystemMessage:
        """Add system message to data store"""
        system_message = await self._message_history.system(message)
        return system_message

    async def add_human_message(self, message: dict) -> HumanMessage:
        """add human message to data store"""
        human_message = await self._message_history.human(message)
        return human_message

    async def add_ai_message(self, message: str) -> AIMessage:
        """Add ai message to data store"""
        ai_message = await self._message_history.ai(message)
        return ai_message
    
    async def add_bulk_messages(self, messages: Sequence[BaseMessage]) -> True:
        """Store messages in bulk in data store"""
        return await self._message_history.bulk_add(messages)
    
    def create_history_aware_chain(self):
        """If no chat_history, then input passed directly to the retriever. If chat_history,""" 
        """then the prompt and LLM used to generate a search query. That search query is then passed to the retriever."""
        return create_history_aware_retriever(
            self._llm.runnable(),
            self._vector_store.runnable(),
            self._rag_prompt_runnable)
        
    
    # TODO: add trimmer runnable  
    async def runnable(self, **kwargs) -> AIMessage:
        """Invoke the chain (Note system prompt stored in message history before runnable execution)"""
        history_aware_chain = self.create_history_aware_chain()

        def use_retriever_result(output):
            return output

        def fallback_to_llm(input):
            return self._llm.runnable().invoke([HumanMessage(content=input['input'])])

        routing_chain = RunnableLambda(
            lambda input: (
                history_aware_chain | RunnableLambda(
                    lambda output: use_retriever_result(output) if output else fallback_to_llm(input)
                )
            )
        )

        chain_with_history = self._message_history.runnable(routing_chain)
        ai_response = await chain_with_history.ainvoke(
            {"input": kwargs['message']},
            config={'session_id': kwargs['session_id']}
        )

        return await self.add_ai_message(ai_response)
    chat = runnable