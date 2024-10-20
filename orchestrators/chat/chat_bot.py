from __future__ import annotations

import logging
import os
from typing import Callable, AsyncGenerator, Optional, List, Any, Dict
import re
import string
import nltk
from nltk import word_tokenize, pos_tag, ngrams
from collections import deque, defaultdict, Counter
from pymongo import DESCENDING
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tracers.schemas import Run
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.documents import Document
from langchain.chains.combine_documents.base import DEFAULT_DOCUMENT_SEPARATOR, DEFAULT_DOCUMENT_PROMPT
from langchain_core.prompts import format_document
from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore,
    AbstractFlexiSchemaFields,
)
from orchestrators.doc.vector_stores.factories import FACTORIES as V_FACTORIES
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingsProxy
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

nltk.data.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'nltk_data'))

class ChatBot(AbstractBot):
    def __init__(self):
        """Composite parts"""
        self.vector_part: ChatBotBuilder.VectorPart = None
        self.llm_part: ChatBotBuilder.LLMPart = None
        self.prompt_part: ChatBotBuilder.PromptPart = None
        self.message_part: ChatBotBuilder.MessagePart = None
    
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
        retrieve_documents: RetrieverOutputLike = RunnableBranch(
            (
                lambda input_data: not input_data.get("chat_history", False),
                (lambda input_data: input_data['input']) | retriever,
            ),
            (preprompt_filter or RunnablePassthrough())
            | prompt
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
        return chain

    def create_context_aware_chain(self, llm: BaseChatModel) -> Runnable:
        """ """
        history_aware_retriever = self.create_history_aware_retriever(
            llm,
            self.vector_part.vector_store.retriever(),
            self.prompt_part.registry['contextualized_template'](),
            preprompt_filter=self.preprompt_filter())
        question_answer_chain = self.create_stuff_documents_chain(
            llm, 
            self.prompt_part.registry['qa_template'](self.prompt_part.user_prompt),
            preprompt_filter=self.preprompt_filter())
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def create_multi_retriever_chain(self, llm: BaseChatModel, retrievers: List[VectorStoreRetriever]) -> Runnable:
        context_prompt = self.prompt_part.registry['contextualized_template']()
        retriever_map = {f'retriever_{i}': retriever for i, retriever in enumerate(retrievers)}
        parallel_retrieval = RunnableParallel(retriever_map)

        def combine_contexts(retrieved_results: dict, separator=DEFAULT_DOCUMENT_SEPARATOR) -> list:
            combined_results = []
            for key, docs in retrieved_results.items():
                combined_docs = separator.join(doc.page_content for doc in docs)
                combined_results.append(
                    Document(page_content=f'Context from {key}:\n{combined_docs}')
                )

            return combined_results
        
        combine_contexts_runnable = RunnableLambda(combine_contexts)

        retrieve_documents = self.create_history_aware_retriever(
            llm,
            parallel_retrieval,
            context_prompt,
            preprompt_filter=self.preprompt_filter())

        return retrieve_documents | combine_contexts_runnable.with_config(run_name='combine_context_chain')

    def create_multi_stuff_chain(self, llm: BaseChatModel) -> Runnable:
        qa_template = self.prompt_part.registry['qa_template'](self.prompt_part.user_prompt)

        return self.create_stuff_documents_chain(
            llm,
            qa_template,
            preprompt_filter=self.preprompt_filter())      

    def create_multicontext_aware_chain(self, llm: BaseChatModel, source_retrievers: List[VectorStoreRetriever]):
        multi_retriever_chain = self.create_multi_retriever_chain(llm, source_retrievers)
        stuffing_chain = self.create_multi_stuff_chain(llm)
        
        multicontext_aware_chain = (
            RunnablePassthrough.assign(
                context=multi_retriever_chain.with_config(run_name='retrieval_chain'),
            ).assign(answer=stuffing_chain)
        ).with_config(run_name='multicontext_aware_chain')

        return multicontext_aware_chain
    
    def _trace_history_chain(self) -> None:
        def _historic_messages_by(n: int) -> List[BaseMessage]:
            messages = self.message_part.message_history.messages[-n:]
            logging.warning(f'Message History {messages}')
            return messages
        runnable = RunnableLambda(
            _historic_messages_by).with_config(run_name='trace_my_history')
        runnable.invoke(20)

    async def _aenter_chat_chain(self, run: Run, config: RunnableConfig) -> Optional[SystemMessage]:
        """On start runnable listener"""
        import json
        collection = self.message_part.message_history.chat_message_history.collection
        
        document = collection.find_one({
            'type': 'system', 
            'content': self.prompt_part.user_prompt, 
            'conversation_id': config['configurable']['session_id']
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
                    'conversation_id': config['configurable']['session_id']
                }, 
                sort=[("createdAt", DESCENDING)])
        ) is not None:
            chain = self.prompt_part.registry['summarization_template']() | self.llm_part.llm.summary_object
            summary = await chain.ainvoke({'input': ai_message['content']})
            self.message_part.message_history.chat_message_history.add_summary(summary.content)

    @staticmethod
    def pos_check(corpus: str, max_repeats: int = 5) -> bool:
        """Validate corpus using pos tagging"""
        tagged_words = pos_tag(word_tokenize(corpus))
        consecutive_pofs_counts = defaultdict(int)
        max_consecutive = defaultdict(lambda: max_repeats)
        max_consecutive.update({
            'IN': 2,
            'CC': 2,
            'UH': 1,
            'PUNCT': 5,
            'SYM': 3,
        })

        punctuation_chars = {'.', ',', ':', '!', '?', '-', '(', ')', '[', ']', '{', '}', '"', "'", '...'}
        special_symbols = {'*', '&', '%', '$', '@', '#'}

        last_tag = None
        for word, tag in tagged_words:
            if word in punctuation_chars:
                tag_category = 'PUNCT'
            elif word in special_symbols:
                tag_category = 'SYM'
            else:
                tag_category = tag[:2] if tag[:2] in consecutive_pofs_counts else tag
            
            if tag_category == last_tag:
                consecutive_pofs_counts[tag_category] += 1
            else:
                consecutive_pofs_counts[tag_category] = 1

            if consecutive_pofs_counts[tag_category] > max_consecutive[tag_category]:
                return False

            last_tag = tag_category

        return True
    
    @staticmethod
    def seq_check(corpus: str, n: int = 3, max_repeats: int = 2) -> bool:
        """Validate corpus by sequences of vocabulary"""
        def clean_text(text: str) -> str:
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            return text
        
        cleaned_corpus = clean_text(corpus)
        words = word_tokenize(cleaned_corpus.lower())
        n_grams = list(ngrams(words, n))
        n_gram_counts = Counter(n_grams)
        
        for n_gram, count in n_gram_counts.items():
            if count > max_repeats:
                logging.warning(f"Repeated sequence detected: {' '.join(n_gram)} (repeated {count} times)")
                return False
        
        return True

    async def rag_astream(
        self, 
        chat_llm: BaseChatModel, 
        message: str,
        source_retrievers: List[VectorStoreRetriever]) -> Callable[[], AsyncGenerator[str, None]]:
        if len(source_retrievers) > 1:
            chain = self.create_multicontext_aware_chain(chat_llm, source_retrievers)
        else:
            chain = self.create_context_aware_chain(chat_llm)
        chain_with_history = self.message_part.message_history.get(chain, True)
        chain_with_history = chain_with_history.with_alisteners(
            on_start=self._aenter_chat_chain,
            on_end=self._aexit_chat_chain)
        config = self.message_part.runnable_config
        async def llm_astream():
            token_buff = deque(maxlen=5)
            vocab_buff = deque(maxlen=30)
            async for s in chain_with_history.astream(
                {'input': message},
                config=config):
                if 'answer' in s:
                    s_content = s['answer']
                    token_buff.append(str(s_content))
                    vocab_buff.append(str(s_content))
                    if len(token_buff) == token_buff.maxlen:
                        if not self.pos_check(''.join(token_buff)):
                            logging.warning(f'Incoherent POS {''.join(token_buff)}')
                            yield '<|model_error|>'
                            break
                    if len(vocab_buff) == vocab_buff.maxlen:
                        if not self.seq_check(''.join(vocab_buff)):
                            logging.warning(f'Incoherent Vocabulary {''.join(vocab_buff)}')
                            yield '<|model_error|>'
                            break

                    yield s_content
        return llm_astream

    async def chat_astream(
        self, 
        chat_llm: BaseChatModel, 
        message: str) -> Callable[[], AsyncGenerator[str, None]]:
        chain = self.create_chain(chat_llm)
        chain_with_history = self.message_part.message_history.get(chain, False)
        chain_with_history = chain_with_history.with_alisteners(
            on_start=self._aenter_chat_chain,
            on_end=self._aexit_chat_chain)
        config = self.message_part.runnable_config
        async def llm_astream():
            token_buff = deque(maxlen=5)
            vocab_buff = deque(maxlen=30)
            async for s in chain_with_history.astream(
                {'input': message},
                config=config):
                    s_content = s.content
                    token_buff.append(str(s_content))
                    vocab_buff.append(str(s_content))
                    if len(token_buff) == token_buff.maxlen:
                        if not self.pos_check(''.join(token_buff)):
                            logging.warning(f'Incoherent POS {''.join(token_buff)}')
                            yield '<|model_error|>'
                            break
                    if len(vocab_buff) == vocab_buff.maxlen:
                        if not self.seq_check(''.join(vocab_buff)):
                            logging.warning(f'Incoherent Vocabulary {''.join(vocab_buff)}')
                            yield '<|model_error|>'
                            break

                    yield s_content
        return llm_astream

    # TODO: add trimmer runnable  
    async def astream(self, message: str) -> Callable[[], AsyncGenerator[str, None]]:
        # await self.vector_part.vector_store.inspect(message)
        self._trace_history_chain()
        source_retrievers = self.vector_part.source_retrievers
        chat_llm = self.llm_part.llm.endpoint_object
        rag_chain = await self.vector_part.aavailable_vectors(message)
        
        return await self.rag_astream(chat_llm, message, source_retrievers) if rag_chain else await self.chat_astream(chat_llm, message)

    chat = astream

class ChatBotBuilder:
    def __init__(self, chat_bot: ChatBot):
        self.chat_bot = chat_bot

    class VectorPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            store: str,
            source_retrievers: List[VectorStoreRetriever],
            embeddings: List[BaseEmbedding], 
            metadata: List[AbstractFlexiSchemaFields]):
            if store not in V_FACTORIES.keys():
                raise ValueError(f'Vector Store {store} is not supported')
            self.embeddings = EmbeddingsProxy(embeddings).get()
            self.vector_store: AbstractVectorStore = V_FACTORIES[store](self.embeddings, metadata)
            self.source_retrievers = source_retrievers
            chat_bot.vector_part = self
    
        async def aavailable_vectors(self, context):
            """Async Available Vector Search"""
            vectors = await self.vector_store.asimilarity_search(context)
            count = len(vectors)
            return count > 0

    class LLMPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            llm: List[LLM]):
            self.llm = LLMProxy(llm).get()
            chat_bot.llm_part = self

    class PromptPart:
        def __init__(
            self, 
            chat_bot: ChatBot, 
            user_prompt: str):
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
            self.message_history = MongoMessageHistory(message_schema)
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
        source_retrievers: List[VectorStoreRetriever],
        embeddings: List[LLM], 
        metadata: List[AbstractFlexiSchemaFields]):
        return ChatBotBuilder.VectorPart(self.chat_bot, store, source_retrievers, embeddings, metadata)
    
    def build_llm_part(self, llm: List[LLM]):
        return ChatBotBuilder.LLMPart(self.chat_bot, llm)
    
    def build_prompt_part(self, user_prompt: str):
        return ChatBotBuilder.PromptPart(self.chat_bot, user_prompt)

    def build_message_part(self, history_config: dict, configurable: dict):
        return ChatBotBuilder.MessagePart(self.chat_bot, history_config, configurable)