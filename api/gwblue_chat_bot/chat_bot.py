from typing import (
    Annotated, 
    TypedDict, 
    Dict, 
    Literal, 
    Optional,
    TypeVar,
    Union,
    Iterator,
    AsyncIterator,
    Sequence,
    TypeAlias,
    override,
    Self,
    Any,
    List,
)
from bson import ObjectId
from collections import defaultdict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import (
    Runnable, 
    RunnableSerializable, 
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
    RunnableBranch,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.output_parsers.string import StrOutputParser
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR, 
    DEFAULT_DOCUMENT_PROMPT,
)
from langchain_core.prompts import format_document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, MessageLikeRepresentation
from langchain_core.messages.utils import AnyMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import (
    ChatGeneration, 
    ChatGenerationChunk
)
from langchain_core.runnables.config import RunnableConfig
from pydantic import (
    Field, 
    model_validator,
    ConfigDict
)
from langchain_core.retrievers import RetrieverLike

from .graph_state import State
from .language_models.huggingface import HuggingFaceInference
from .chat_bot_config import ChatBotConfig
# from .local_tools.route_query_tool import RouteQueryTool

from langchain_redis import RedisConfig
from redisvl.query.filter import Tag, FilterExpression
from ..gwblue_redis_vectoretriever.config import VectorStoreSchema
from ..gwblue_redis_vectoretriever.vectorstore import RedisVectorStoreTTL

from .prompts import registry
from .message_history import (
    MongoMessageHistorySchema, 
    MongoMessageHistory, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    BaseMessage, 
)

ChatGenerationLike: TypeAlias = ChatGeneration | Iterator[ChatGeneration] | AsyncIterator[ChatGenerationChunk]

I = TypeVar('I', bound=Union[PromptValue, str, Sequence[MessageLikeRepresentation]])
O = TypeVar('O', bound=ChatGenerationLike)
C = TypeVar('C', bound=BaseChatModel)
S = TypeVar('S', bound=BaseChatModel)

class ChatBot(RunnableSerializable[I, O]):
    config: ChatBotConfig
    graph: Optional[CompiledStateGraph] = None
    chat_model: BaseChatModel = Field(default=None, exclude=True)
    retry_model: BaseChatModel = Field(default=None, exclude=True)
    safety_model: BaseChatModel = Field(default=None, exclude=True)
    embeddings: Embeddings = Field(default=None, exclude=True)
    vector_store: RedisVectorStoreTTL = Field(default=None, exclude=True)
    message_history: MongoMessageHistory = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode='after')
    def load_environment(self) -> Self:
        graph = StateGraph(State)

        hf = HuggingFaceInference(config=self.config,model_types={})
        inference_engine = {
            'tgi': hf,
            'tei': hf,
            'vllm': None,
        }
        
        self.chat_model = inference_engine[self.config.llm.server]('chat_model')
        self.safety_model = inference_engine[self.config.guardrails.server]('guardrails')
        self.embeddings = inference_engine[self.config.embeddings.server]('embeddings')

        config = RedisConfig(**{
            'redis_client': self.config.vectorstore.client,
            'metadata_schema': self.config.vectorstore.metadata_schema,
            'embedding_dimensions': self.config.embeddings.dimensions
            **VectorStoreSchema().model_dump()
        })
            
        self.vector_store = RedisVectorStoreTTL(
            self.embeddings, 
            config=config
        )

        message_schema = MongoMessageHistorySchema(
            session_id=self.config.message_history.session_id,
            connection_string=self.config.message_history.url,
            database_name=self.config.message_history.name,
            collection_name=self.config.message_history.collection_name,
            session_id_key=self.config.message_history.session_id_key,
        )
        self.message_schema = message_schema
        self.message_history = MongoMessageHistory(self.message_schema)

        self.graph = self._compile(graph)

        return self
    
    @property
    @override
    def InputType(self) -> TypeAlias:
        from langchain_core.prompt_values import (
            ChatPromptValueConcrete,
            StringPromptValue,
        )

        return Union[
            str,
            Union[StringPromptValue, ChatPromptValueConcrete],
            list[AnyMessage],
        ]

    def invoke(
        self,
        input: I,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> O:
        return self.graph.invoke({
            'messages': input.messages,
            'docs': kwargs['docs'],
            'session_id': self.config.user_config.session_id,
            'retrieval_mode': kwargs['retrieval_mode'],
        })

    @staticmethod
    def preprompt_filter(state: State) -> RunnableLambda:
        def create_preprompt_filter(input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **input_data,
                'chat_history': [
                    message for message in input_data.get('chat_history', [])
                    if not isinstance(message, SystemMessage) or not message.additional_kwargs.get('preprompt', False)
                ]
            }
        
        return RunnableLambda(create_preprompt_filter).with_config(
            run_name=f'filter_preprompt_chain_{state['route']}',
            metadata=state['metadata']
        )
    
    @staticmethod
    def create_filter_expression(metadata: Dict[str, Any]) -> FilterExpression:
        from functools import reduce
        import operator

        tag_expressions = [
            Tag(key) == str(value)
            for key, value in metadata.items() 
        ]
        filter_expression = reduce(operator.and_, tag_expressions)
        return filter_expression

    def create_history_aware_retriever(
        self,
        retriever: RetrieverLike,
        prompt: BasePromptTemplate,
        preprompt_filter: Optional[Runnable] = None,
    ) -> Runnable:
        """Custom implementation to handle preprompt messages"""
        def validate_history(input_data: Dict[str, Any]) -> bool:
            # TGI is really slow to respond to streaming; temporarily disabling this with explicit True
            # return True
            return not input_data.get('chat_history')
            
        retrieve_documents = (preprompt_filter or RunnablePassthrough()) | RunnableBranch(
            (
                validate_history,
                (lambda input_data: input_data['input']) | retriever,
            ),
            prompt
            | self.chat_model
            | StrOutputParser()
            | retriever,
        ).with_config(run_name='history_aware_retriever_chain')
        
        return retrieve_documents

    def create_stuff_documents_chain(
        self,
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
            | self.chat_model
            | StrOutputParser()
        ).with_config(run_name='stuff_documents_chain')    

    def create_context_aware_chain(self, state: State) -> Runnable:
        metadata = state['metadata']
        system_prompt = state['messages'][0] # get current system prompt

        filter_expression = self.create_filter_expression(metadata)
        search_kwargs = {
            'k': 6,
            'filter': filter_expression,
        }
        if state['retrieval_mode'] == 'similarity_score_threshold':
            search_kwargs['score_threshold'] = 0.8

        retriever = self.vector_store.as_retriever(
            search_type=state['retrieval_mode'],
            search_kwargs=search_kwargs
        ).with_config(
            tags=[f'create_context_aware_chain_{state['route']}'],
            metadata=metadata,           
        )
        
        history_aware_retriever = self.create_history_aware_retriever(
            retriever,
            registry['contextualized_template'](),
            preprompt_filter=self.preprompt_filter(state),
        )

        question_answer_chain = self.create_stuff_documents_chain(
            registry['qa_template'](system_prompt),
            preprompt_filter=self.preprompt_filter()
        )
        
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def _compile(self, graph: StateGraph):
        def guardrails(state: State) -> Dict[str, any]:
            ai_message = self.safety_model.invoke([state['messages'][1]])
            return { 'messages': [ai_message] }

        def guardrails_condition(state: State) -> str:
            last_msg: AIMessage = state["messages"][-1]
            text = last_msg.content.lower().strip('\n')
            if 'safe' in text:
                return 'route_query'
            else:
                return 'not_safe'

        def not_safe(state: State):
            return {
                "messages": [
                    {
                        "role": "ai",
                        "content": "Your request cannot be processed. (Content flagged as not safe.)"
                    }
                ]
            }          
        def route_query(state: State):
            """
            Account for scenarios:
            - 'Explain this document' (where this is not specified and refers to upload)
            - 'Compare these two documents' (where these is not specified and refers to multiple uploads)
            - 'Explain why it has impact' (where it refers to previously vectorized data)
            - 'Compare this and that' (where this is current upload and that is previously vectorized data)
            - 'Explain something' (where no vector data but can tap into pretrained corpus of LLM)
            """
            metadata = state['metadata']
            session_id = state['session_id']

            if len(metadata) > 1:
                return {'route': 'multi_doc_prompt', 'metadata': metadata}
            
            if len(metadata) == 1:
                return {"route": 'single_doc_prompt', 'metadata': metadata}
        
            human_prompt = state['messages'][-1].content

            relevant_docs_with_score = self.vector_store.similarity_search_with_score(
                query=human_prompt,
                k=50,
                filter=Tag(self.config.message_history.session_id_key) == str(session_id),
            )
            file_to_best_chunk = defaultdict(lambda: (None, float('inf')))
            for doc, dist in relevant_docs_with_score:
                fname = doc.metadata.get('source', '')                
                if dist < file_to_best_chunk:
                    file_to_best_chunk[fname] = (doc, dist)

            best_metadata = [doc.metadata for (doc, _) in file_to_best_chunk.values() if doc is not None]

            num_files = len(best_metadata)
            if num_files > 1:
                route = 'multi_doc_prompt'
            elif num_files == 1:
                route = 'single_doc_prompt'
            else:
                route = 'pretrained_corpus_prompt'
            
            return {
                'route': route,
                'messages': state['messages'],
                'best_docs': best_metadata,
            }  

        def route_query_condition(state: State) -> str:
            return state['route']
        
        def single_doc_prompt(state: State) -> dict[str, Any]:
            """
            Generate prompt for single document
            """
            return self.create_context_aware_chain(state)
        
        def multi_doc_prompt(state: State) -> dict[str, Any]:
            """
            Generate prompt for multiple documents
            """
            ...

        def pretrained_corpus_prompt(state: State) -> dict[str, Any]:
            """
            Generate prompt for pretrained corpus
            """
            ...

        def generate(state: State) -> dict[str, Any]:
            """
            Generate single and multiple candidate completions
            by invoking chat model
            """
            ...
        
        graph.add_node('guardrails', guardrails)
        graph.add_edge(START, 'guardrails')
        graph.add_conditional_edges(
            'guardrails',
            guardrails_condition,
            {
                'route_query': 'route_query',
                'not_safe': 'not_safe'
            }
        )
        graph.add_node('not_safe', not_safe)
        graph.add_edge('not_safe', END)
        graph.add_node('route_query', route_query)
        graph.add_conditional_edges(
            'route_query',
            route_query_condition,
            {
                #'vectorstore': 'vectorstore',
                #'dataframe_tool': 'dataframe_tool',
                'single_doc_prompt': 'single_doc_prompt',
                'multi_doc_prompt': 'multi_doc_prompt',
                'pretrained_corpus_prompt': 'pretrained_corpus_prompt',
            }
        )
        #graph.add_edge('vectorstore', 'generate')
        #graph.add_edge('dataframe_tool', 'generate')
        graph.add_edge('single_doc_prompt', 'generate')
        graph.add_edge('multi_doc_prompt', 'generate')
        graph.add_edge('pretrained_corpus_prompt', 'generate')
        return graph.compile()


        """
        Determine if multiple candidate completions (
        use both semantic similarity and max marginal
        relevance
        )
        Go to vector store find vectors based on the single
        doc
        AND ALSO find vectors based on the query for the given
        conversation_id (just in case they ask to compare with
        a previous doc)
        """