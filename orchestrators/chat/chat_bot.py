from __future__ import annotations

import logging
import os
import json
from typing import Callable, AsyncGenerator, Optional, List, Any, Dict
import re
from collections import deque
from pymongo import DESCENDING

from langchain_core.runnables import (
    Runnable, 
    RunnablePassthrough,
    RunnableLambda, 
    RunnableParallel, 
    RunnableBranch,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers.schemas import Run
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR, 
    DEFAULT_DOCUMENT_PROMPT,
)
from langchain_core.prompts import format_document

from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore, 
    create_filter_expression,
)
from orchestrators.doc.vector_stores.factories import STORE_FACTORIES, RETRIEVER_FACTORIES
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingsProxy
from orchestrators.doc.vector_stores.abstract_vector_retriever import AbstractVectorRetriever

from orchestrators.chat.messages.prompts import registry
from orchestrators.chat.abstract_bot import AbstractBot
from orchestrators.chat.llm_models.llm import LLM
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

from orchestrators.nlharmony.lexical_lang import LexicalLang

class ChatBot(AbstractBot):
    def __init__(self):
        """Composite parts"""
        self.vector_part: ChatBotBuilder.VectorPart = None
        self.llm_part: ChatBotBuilder.LLMPart = None
        self.guardrails_part: ChatBotBuilder.GuardrailsPart = None
        self.prompt_part: ChatBotBuilder.PromptPart = None
        self.message_part: ChatBotBuilder.MessagePart = None

    def _trace_history_chain(self) -> None:
        def _historic_messages_by(n: int) -> List[BaseMessage]:
            messages = self.message_part.message_history.messages[-n:]
            logging.info(f'Message History {messages}')
            return messages
        runnable = RunnableLambda(
            _historic_messages_by).with_config(run_name='trace_my_history')
        runnable.invoke(20)

    @staticmethod
    def preprompt_filter() -> RunnableLambda:
        def create_preprompt_filter(input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **input_data,
                'chat_history': [
                    message for message in input_data.get('chat_history', [])
                    if not isinstance(message, SystemMessage) or not message.additional_kwargs.get('preprompt', False)
                ]
            }
        
        return RunnableLambda(create_preprompt_filter).with_config(run_name='filter_preprompt_chain')

    def create_history_aware_retriever(
        self,
        llm: LanguageModelLike,
        retriever: RetrieverLike,
        prompt: BasePromptTemplate,
        preprompt_filter: Optional[Runnable] = None,
    ) -> Runnable:
        """Custom implementation to handle preprompt messages"""
        def validate_history(input_data: Dict[str, Any]) -> bool:
            # TGI is really slow to respond to streaming; temporarily disabling this
            # return not input_data.get('chat_history')
            return True
        
        retrieve_documents = (preprompt_filter or RunnablePassthrough()) | RunnableBranch(
            (
                validate_history,
                (lambda input_data: input_data['input']) | retriever,
            ),
            prompt
            | llm
            | StrOutputParser()
            | retriever,
        ).with_config(run_name="history_aware_retriever_chain")
        
        return retrieve_documents
    
    def create_stuff_documents_chain(
        self,
        llm: LanguageModelLike,
        prompt: BasePromptTemplate,
        preprompt_filter: Optional[Runnable] = None,
    ) -> Runnable[Dict[str, Any], Any]:
        """Custom implementation to handle preprompt messages"""        
        def format_docs(inputs: dict) -> str:
            return DEFAULT_DOCUMENT_SEPARATOR.join(
                format_document(doc, DEFAULT_DOCUMENT_PROMPT)
                for doc in inputs['context']
            )

        return (
            (preprompt_filter or RunnablePassthrough())
            | RunnablePassthrough.assign(context=format_docs).with_config(run_name='format_inputs')
            | prompt
            | llm
            | StrOutputParser()
        ).with_config(run_name='stuff_documents_chain')    

    def create_chain(self, llm: BaseChatModel) -> Runnable:
        chain = self.prompt_part.registry['chat_preprompt_template'](self.prompt_part.user_prompt) | llm
        return chain.with_config(run_name='prompt_llm_chain')

    def create_context_aware_chain(self, llm: BaseChatModel, source_retriever: AbstractVectorRetriever) -> Runnable:
        """ """
        history_aware_retriever = self.create_history_aware_retriever(
            llm,
            source_retriever.retriever,
            self.prompt_part.registry['contextualized_template'](),
            preprompt_filter=self.preprompt_filter())
        question_answer_chain = self.create_stuff_documents_chain(
            llm, 
            self.prompt_part.registry['qa_template'](self.prompt_part.user_prompt),
            preprompt_filter=self.preprompt_filter())
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def create_multi_retriever_chain(self, llm: BaseChatModel, source_retrievers: List[AbstractVectorRetriever]) -> Runnable:
        context_prompt = self.prompt_part.registry['contextualized_template']()
        retriever_map = {f'Source {source_retriever.source}': source_retriever.retriever for source_retriever in source_retrievers}
        parallel_retrieval = RunnableParallel(retriever_map)

        def combine_contexts(retrieved_results: dict, separator=DEFAULT_DOCUMENT_SEPARATOR) -> list:
            combined_results = []
            for key, docs in retrieved_results.items():
                combined_docs = separator.join(doc.page_content for doc in docs)
                combined_results.append(
                    Document(page_content=f'Context from {key}:\n{combined_docs}')
                )

            return combined_results
        
        combine_contexts_runnable = RunnableLambda(combine_contexts) \
            .with_config(run_name='combine_context_chain')

        retrieve_documents = self.create_history_aware_retriever(
            llm,
            parallel_retrieval,
            context_prompt,
            preprompt_filter=self.preprompt_filter())

        return retrieve_documents | combine_contexts_runnable
    
    def create_multi_stuff_chain(self, llm: BaseChatModel) -> Runnable:
        qa_template = self.prompt_part.registry['qa_template'](self.prompt_part.user_prompt)

        return self.create_stuff_documents_chain(
            llm,
            qa_template,
            preprompt_filter=self.preprompt_filter())      

    def create_multicontext_aware_chain(self, llm: BaseChatModel, source_retrievers: List[AbstractVectorRetriever]):
        multi_retriever_chain = self.create_multi_retriever_chain(llm, source_retrievers)
        stuffing_chain = self.create_multi_stuff_chain(llm)
        
        multicontext_aware_chain = (
            RunnablePassthrough.assign(
                context=multi_retriever_chain.with_config(run_name='retrieval_chain'),
            ).assign(answer=stuffing_chain)
        ).with_config(run_name='multicontext_aware_chain')

        return multicontext_aware_chain

    async def _aenter_chat_chain(self, run: Run, config: RunnableConfig) -> Optional[SystemMessage]:
        """On start runnable listener"""
        collection = self.message_part.message_history.chat_message_history.collection
        
        document = collection.find_one({
            'type': 'system', 
            'content': self.prompt_part.user_prompt, 
            self.message_part.message_schema.session_id_key: config['configurable']['session_id'],
        })
        
        if document is None:
            await self.message_part.aadd_system_message(self.prompt_part.user_prompt, additional_kwargs={'preprompt': True})
        else:
            history_data = json.loads(document['History'])
            
            additional_kwargs = history_data.get('data', {}).get('additional_kwargs', {})
            if not additional_kwargs.get('preprompt', False):
                await self.message_part.aadd_system_message(self.prompt_part.user_prompt, additional_kwargs={'preprompt': True})

    async def _aexit_chat_chain(self, run: Run, config: RunnableConfig) -> None:
        """On end runnable listener"""
        collection = self.message_part.message_history.chat_message_history.collection
        if(
            ai_message := collection.find_one(
                {
                    'type': { '$in': ['ai', 'AIMessageChunk'] }, 
                    self.message_part.message_schema.session_id_key: config['configurable']['session_id'],
                }, 
                sort=[("createdAt", DESCENDING)])
        ) is not None:
            chain = self.prompt_part.registry['summarization_template']() | self.llm_part.llm.summary_object
            summary = await chain.ainvoke({'input': ai_message['content']})
            self.message_part.message_history.chat_message_history.add_summary(summary.content)
    
    def format_cancel_message(self):
        pattern = r"<BEGIN UNSAFE CONTENT CATEGORIES>(.*?)<END UNSAFE CONTENT CATEGORIES>"
        match = re.search(pattern, self.prompt_part.registry['guardrails_template']().template, re.DOTALL)
        content = 'Some of the content in your prompt falls under standardized hazards taxonomy.\n'
        content += 'Please review the following hazard categories:\n\n'
        content += match.group(1).strip()
        return content
    
    async def calculate_vecs(self, message: str) -> bool:
        if self.vector_part.source_retrievers:
            from langchain_core.callbacks.base import AsyncCallbackHandler
            class ProcessDocumentsCallback(AsyncCallbackHandler):
                async def on_retriever_end(self, documents: List[Document], **kwargs: Any) -> None:
                    n = 2
                    num_steps = 4
                    indices = [n**i for i in range(1, num_steps + 1) if n**i < len(documents)]
                    retrieved_docs = [documents[i] for i in indices]
                    logging.info(f'retrieved docs subset {retrieved_docs}')

            import random
            source_retriever = random.choice(self.vector_part.source_retrievers)
            docs = await source_retriever.retriever.ainvoke(
                message,
                config={
                    'callbacks': [ProcessDocumentsCallback()]
                }
            )
            return len(docs) > 0

        return await self.vector_part.aavailable_vectors(message)
    
    def fetch_retrievers(self) -> List[AbstractVectorRetriever]:
        source_retrievers = self.vector_part.source_retrievers or [self.vector_part.no_doc_retriever]
        return source_retrievers

    async def cancel_astream(self) -> Callable[[], AsyncGenerator[str, None]]:
        import asyncio

        def stream_chunks(message: str, chunk_size: int = 10):
            for i in range(0, len(message), chunk_size):
                yield message[i:i + chunk_size]

        async def llm_astream() -> AsyncGenerator[str, None]:
            for chunk in stream_chunks(self.format_cancel_message()):
                await asyncio.sleep(0.1)
                yield chunk

        return llm_astream

    async def rag_astream(
        self, 
        chat_llm: BaseChatModel, 
        message: str,
        source_retrievers: List[AbstractVectorRetriever]
    ) -> Callable[[], AsyncGenerator[str, None]]:
        if len(source_retrievers) > 1:
            chain = self.create_multicontext_aware_chain(chat_llm, source_retrievers)
        else:
            chain = self.create_context_aware_chain(chat_llm, source_retrievers[0])

        chain_with_history = self.message_part.message_history.get(chain, True)
        chain_with_history = chain_with_history.with_alisteners(
            on_start=self._aenter_chat_chain,
            on_end=self._aexit_chat_chain)
        config = self.message_part.runnable_config
        async def llm_astream():
            token_buff = deque(maxlen=100)
            async for s in chain_with_history.astream(
                {'input': message},
                config=config):
                if 'answer' in s:
                    s_content = s['answer']
                    token_buff.append(str(s_content))
                    if len(token_buff) == token_buff.maxlen:
                        corpus = ''.join(token_buff)
                        ll = LexicalLang(corpus=corpus, temperature=0.3)
                        if ll.is_natural and ll.is_high_frequency:
                            logging.warning(f'Low P(A) natural language score, got {''.join(token_buff)}')
                            yield '<|model_error|>'
                            break
                    yield s_content

        return llm_astream

    async def chat_astream(
        self, 
        chat_llm: BaseChatModel, 
        message: str
    ) -> Callable[[], AsyncGenerator[str, None]]:
        chain = self.create_chain(chat_llm)
        chain_with_history = self.message_part.message_history.get(chain, False)
        chain_with_history = chain_with_history.with_alisteners(
            on_start=self._aenter_chat_chain,
            on_end=self._aexit_chat_chain)
        config = self.message_part.runnable_config
        async def llm_astream():
            token_buff = deque(maxlen=100)
            async for s in chain_with_history.astream(
                {'input': message},
                config=config):
                    s_content = s.content
                    token_buff.append(str(s_content))
                    if len(token_buff) == token_buff.maxlen:
                        corpus = ''.join(token_buff)
                        ll = LexicalLang(corpus=corpus, temperature=0.3)
                        if ll.is_natural and ll.is_high_frequency:
                            logging.warning(f'Low P(A) natural language score, got {''.join(token_buff)}')
                            yield '<|model_error|>'
                            break
                    yield s_content

        return llm_astream

    # TODO: add trimmer runnable  
    async def astream(self, message: str) -> Callable[[], AsyncGenerator[str, None]]:
        # await self.vector_part.inspect(message)
        self._trace_history_chain()

        if self.guardrails_part.llm:
            is_safe = await self.guardrails_part.content_safe(
                message, 
                self.prompt_part.registry['guardrails_template']())
            if not is_safe:
                return await self.cancel_astream()
            
        chat_llm = self.llm_part.llm.endpoint_object
        rag_chain = await self.calculate_vecs(message)

        return await self.rag_astream(chat_llm, message, self.fetch_retrievers()) if rag_chain else await self.chat_astream(chat_llm, message)

    chat = astream

class ChatBotBuilder:
    def __init__(self, chat_bot: ChatBot):
        self.chat_bot = chat_bot

    class VectorPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            store: str,
            source_retrievers: List[AbstractVectorRetriever],
            embeddings: List[BaseEmbedding],
            metadata: dict,
        ):
            if store not in STORE_FACTORIES.keys():
                raise ValueError(f'Vector Store {store} is not supported')
            
            self.store = store
            self.metadata = metadata
            vector_store_schema = json.loads(os.environ['VECTOR_STORE_SCHEMA'])
            self.filter = create_filter_expression(vector_store_schema, self.metadata)
            self.embeddings = EmbeddingsProxy(embeddings).get()
            self.vector_store: AbstractVectorStore = STORE_FACTORIES[store](
                vector_store_schema, self.embeddings)
            self.source_retrievers = source_retrievers
            chat_bot.vector_part = self
    
        async def aavailable_vectors(self, context) -> bool:
            """Async Available Vector Search"""
            vectors = await self.vector_store.asimilarity_search(context, filter=self.filter)
            count = len(vectors)
            return count > 0
        
        @property
        def no_doc_retriever(self) -> AbstractVectorRetriever:
            return RETRIEVER_FACTORIES[self.store](
                filter=self.filter,
                vector_store_proxy=self.vector_store,
                metadata=self.metadata,
            )

        async def inspect(self, context) -> str:
            return await self.vector_store.inspect(context, filter=self.filter)

    class LLMPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            llm: List[LLM]
        ):
            self.llm = LLMProxy(llm).get()
            chat_bot.llm_part = self

    class GuardrailsPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            llm: List[LLM]
        ):
            self.llm = LLMProxy(llm).get() if llm else []
            chat_bot.guardrails_part = self

        async def content_safe(self, message: str, prompt: BasePromptTemplate) -> bool:
            endpoint_object: BaseChatModel = self.llm.endpoint_object
            chain = prompt | endpoint_object

            response = await chain.ainvoke({'input': message, 'agent_type': 'user'})
            return response.content.strip() == 'safe'

    class PromptPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            user_prompt: str
        ):
            self.user_prompt = user_prompt
            self.registry = registry
            chat_bot.prompt_part = self

    class MessagePart:
        def __init__(self, chat_bot: ChatBot, history_config: dict, configurable: dict):
            if not configurable['session_id']:
                raise ValueError('Session ID Required for History')
            self.configurable = configurable
            message_schema = MongoMessageHistorySchema(
                session_id=self.configurable['session_id'], 
                **history_config)
            self.message_schema = message_schema
            self.message_history = MongoMessageHistory(self.message_schema)
            chat_bot.message_part = self

        @property
        def runnable_config(self) -> dict:
            return {
                'configurable': self.configurable
            }

        def add_system_message(self, message: str, **kwargs: Any) -> SystemMessage:
            """Add system message to data store"""
            system_message = self.message_history.system(message, **kwargs)
            return system_message
        
        async def aadd_system_message(self, message: str, **kwargs: Any) -> SystemMessage:
            """Add system message to data store"""
            system_message = await self.message_history.asystem(message, **kwargs)
            return system_message

        async def aadd_human_message(self, message: dict) -> HumanMessage:
            """add human message to data store"""
            human_message = await self.message_history.ahuman(message)
            return human_message

        async def aadd_ai_message(self, message: str) -> AIMessage:
            """Add ai message to data store"""
            ai_message = await self.message_history.aai(message)
            return ai_message
        
        async def aadd_bulk_messages(self, messages: Sequence[BaseMessage]) -> True:
            """Store messages in bulk in data store"""
            return await self.message_history.abulk_add(messages)

    def build_vector_part(
        self, 
        store: str,
        source_retrievers: List[AbstractVectorRetriever],
        embeddings: List[LLM], 
        metadata: dict
    ):
        return ChatBotBuilder.VectorPart(self.chat_bot, store, source_retrievers, embeddings, metadata)
    
    def build_llm_part(self, llm: List[LLM]):
        return ChatBotBuilder.LLMPart(self.chat_bot, llm)
    
    def build_guardrails_part(self, llm: List[LLM]):
        return ChatBotBuilder.GuardrailsPart(self.chat_bot, llm)
    
    def build_prompt_part(self, user_prompt: str):
        return ChatBotBuilder.PromptPart(self.chat_bot, user_prompt)

    def build_message_part(self, history_config: dict, configurable: dict):
        return ChatBotBuilder.MessagePart(self.chat_bot, history_config, configurable)