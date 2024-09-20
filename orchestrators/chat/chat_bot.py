import logging
from typing import Callable, Tuple
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.doc.redistore import RediStore as VectorStore
from langchain_huggingface import HuggingFaceEndpoint
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingProxy
from orchestrators.chat.messages.prompts import registry
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.model_proxy import ModelProxy as LLMProxy
from orchestrators.chat.llm_models.llm import StreamingToClientCallbackHandler
from orchestrators.chat.messages.message_history import (
    MongoMessageHistory,
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    Sequence,
)

class ChatBot(AbstractBot):
    def __init__(self, llm: LLMProxy, embeddings: EmbeddingProxy, message_history: MongoMessageHistory, vector_options: dict):
        self._llm = llm.get()
        self._vector_store = VectorStore(embeddings, vector_options['uuid'], vector_options['conversation_id'])
        self._message_history = message_history
        self._contextualized_template = registry['contextualized_template']()
        self._qa_template = registry['qa_template']()
        self._llm_template = registry['llm_template']()

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
    
    def create_rag_chain_advanced(self, llm: HuggingFaceEndpoint):
        """Create history-aware RAG chain"""
        history_aware_retriever = create_history_aware_retriever(
            llm,
            self._vector_store.runnable(),
            self._contextualized_template)
        question_answer_chain = create_stuff_documents_chain(llm, self._qa_template)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def create_rag_chain(self, llm: HuggingFaceEndpoint):
        pass
    
    def create_llm_chain(self, input, llm: HuggingFaceEndpoint):
        chain = self._llm_template | llm
        return chain

    def available_vectors(self, context, metadata):
        count = len(self._vector_store.similarity_search(context, metadata))
        logging.warning(f'VECTOR COUNT REPORTING {count}')
        return count > 0

    # TODO: add trimmer runnable  
    async def runnable(self, **kwargs) -> Tuple[Callable[[], None], StreamingToClientCallbackHandler]:
        """Invoke the chain"""
        endpoint = self._llm.endpoint_object
        rag_chain = self.available_vectors(kwargs['message'], kwargs['metadata'])
        routing_chain = RunnableLambda(
            lambda input: (
                self.create_rag_chain_advanced(endpoint) if rag_chain
                else self.create_llm_chain(input['input'], endpoint)
            )
        )
        chain_with_history = self._message_history.get(routing_chain, rag_chain)
        def run_llm():
            chain_with_history.invoke(
                {'input': kwargs['message']},
                config={'session_id': kwargs['session_id']}
            )
        return run_llm, self._llm.streaming_handler

    chat = runnable