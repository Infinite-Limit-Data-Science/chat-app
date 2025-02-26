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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompt_values import PromptValue
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
from .language_models.huggingface import HuggingFaceInference
from .chat_bot_config import ChatBotConfig
from .local_tools.route_query_tool import RouteQueryTool

ChatGenerationLike: TypeAlias = ChatGeneration | Iterator[ChatGeneration] | AsyncIterator[ChatGenerationChunk]

I = TypeVar('I', bound=Union[PromptValue, str, Sequence[MessageLikeRepresentation]])
O = TypeVar('O', bound=ChatGenerationLike)
C = TypeVar('C', bound=BaseChatModel)
S = TypeVar('S', bound=BaseChatModel)

class State(TypedDict):
    route: str
    docs: List[str]
    session_id: ObjectId
    messages: Annotated[list, add_messages]

class ChatBot(RunnableSerializable[I, O]):
    config: ChatBotConfig
    graph: Optional[CompiledStateGraph] = None

    chat_model: BaseChatModel = Field(default=None, exclude=True)
    retry_model: BaseChatModel = Field(default=None, exclude=True)
    safety_model: BaseChatModel = Field(default=None, exclude=True)
    embeddings: Embeddings = Field(default=None, exclude=True)
    vectorstore: Embeddings = Field(default=None, exclude=True)
    message_history: Embeddings = Field(default=None, exclude=True)

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

        # while mongodb module can remain in gwblue_chat_bot since it is tied to chat bot
        # and nothing else, the redis module should be extracted out since it is also being
        # used in other modules like document ingestion.
        self.vectorstore = RedisVectorStore(self.config.vectorstore)

        self.configurable = configurable
        message_schema = MongoMessageHistorySchema(
            session_id=self.configurable['session_id'], 
            **history_config)
        self.message_schema = message_schema
        self.message_history = MongoMessageHistory(self.message_schema)
        self.message_history = MongoMessageHistory(self.config.message_history)

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
            'session_id': self.config.user_config.session_id  
        })

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
            docs = state['docs']
            session_id = state['session_id']

            if len(docs) > 1:
                return {'route': 'multi_doc_prompt', 'messages': state['messages']}
            
            if len(docs) == 1:
                return {"route": 'single_doc_prompt', 'messages': state['messages']}
        
            human_prompt = state['messages'][-1].content

            relevant_docs = self.vectorstore.similarity_search(
                human_prompt, 
                k=10,
                metadata_filter={
                    'conversation_id': str(self.config.user_config.session_id)
                }  
            )
            # if

            # if len(state['docs']):
            #     if len(state['docs']) > 1:
            #         ...# direct_doc_review
            #     else:
            #         ...# multidoc_compare
            # else:              

        def route_query_condition(state: State) -> str:
            return state['route']
        
        def single_doc_prompt(state: State) -> dict[str, Any]:
            """
            Generate prompt for single document
            """
            ...
        
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