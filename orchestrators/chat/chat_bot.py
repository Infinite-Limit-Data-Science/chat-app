import logging
from typing import Callable, AsyncGenerator, Optional, List
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from orchestrators.doc.redistore import RediStore as VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers.schemas import Run
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingProxy
from orchestrators.chat.messages.prompts import registry
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.model_proxy import ModelProxy as LLMProxy
from orchestrators.chat.messages.message_history import (
    MongoMessageHistorySchema,
    MongoMessageHistory,
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    Sequence,
)

class ChatBot(AbstractBot):
    def __init__(
            self, 
            user_prompt_template: str, 
            llm: LLMProxy, 
            embeddings: EmbeddingProxy, 
            history_config: dict,
            vector_config: dict):
        self._user_template = user_prompt_template
        self._llm = llm.get()
        self._vector_store = VectorStore(embeddings, vector_config['uuid'], vector_config['conversation_id'])
        self._message_history = MongoMessageHistory(
            MongoMessageHistorySchema(session_id=vector_config['conversation_id'], **history_config))
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
    
    def create_rag_chain_advanced(self, llm: BaseChatModel) -> Runnable:
        """Create history-aware RAG chain"""
        history_aware_retriever = create_history_aware_retriever(
            llm,
            self._vector_store.retriever(),
            self._contextualized_template)
        question_answer_chain = create_stuff_documents_chain(llm, self._qa_template)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    def create_rag_chain(self, llm: BaseChatModel):
        pass
    
    def create_llm_chain(self, llm: BaseChatModel) -> Runnable:
        chain = self._llm_template | llm
        return chain

    def available_vectors(self, context, metadata):
        count = len(self._vector_store.similarity_search(context, metadata))
        logging.warning(f'VECTOR COUNT REPORTING {count}')
        return count > 0

    @staticmethod
    def runnable_config(options) -> dict:
        return {
            'configurable': {'session_id': options['session_id']}
        }

    def _trace_history_chain(self) -> None:
        def _historic_messages_by(n: int) -> List[BaseMessage]:
            messages = self._message_history.messages[-n:]
            logging.warning(f'Message History {messages}')
            return messages
        runnable = RunnableLambda(
            _historic_messages_by).with_config(run_name='trace_my_history')
        runnable.invoke(20)

    async def _aenter_chat_chain(self, run: Run, config: RunnableConfig) -> Optional[SystemMessage]:
        """On start runnable listener"""
        collection = self._message_history.chat_message_history.collection
        if(
            _ := collection.find_one({'type': 'system', 'content': self._user_template})
        ) is None:
            await self.add_system_message(self._user_template)

    async def rag_astream(self, chat_llm, options):
        chain = self.create_rag_chain_advanced(chat_llm)
        chain_with_history = self._message_history.get(chain, True)
        chain_with_history = chain_with_history.with_alisteners(on_start=self._aenter_chat_chain)
        config=self.runnable_config(options)
        async def llm_astream():
            stop_token = "<|eot_id|>"
            async for s in chain_with_history.astream(
                {'input': options['message']},
                config=config):
                if 'answer' in s:
                    s_content = s['answer']
                    if stop_token in s_content:
                        s_content = s_content.replace(stop_token, "")

                    yield s_content
        return llm_astream

    async def chat_astream(self, chat_llm, options):
        chain = self.create_llm_chain(chat_llm)
        chain_with_history = self._message_history.get(chain, False)
        chain_with_history = chain_with_history.with_alisteners(on_start=self._aenter_chat_chain)
        config=self.runnable_config(options)
        async def llm_astream():
            stop_token = "<|eot_id|>"
            async for s in chain_with_history.astream(
                {'input': options['message']},
                config=config):
                    # Remove the stop token if it's present
                    if stop_token in s.content:
                        s.content = s.content.replace(stop_token, "")
                    yield s.content
        return llm_astream

    # TODO: add trimmer runnable  
    async def astream(self, **kwargs) -> Callable[[], AsyncGenerator[str, None]]:
        """Invoke the chain"""
        self._trace_history_chain()
        chat_llm = self._llm.endpoint_object
        rag_chain = self.available_vectors(kwargs['message'], kwargs['metadata'])
        return await self.rag_astream(chat_llm, kwargs) if rag_chain else await self.chat_astream(chat_llm, kwargs)

    chat = astream