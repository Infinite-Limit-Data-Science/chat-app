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
    Self
)
import json
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompt_values import PromptValue
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, MessageLikeRepresentation
from langchain_core.messages.utils import AnyMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import (
    ChatGeneration, 
    ChatGenerationChunk
)
from pydantic import (
    BaseModel, 
    Field, 
    field_validator, 
    model_validator
)
from .language_models.huggingface import HuggingFaceInference
from .chat_bot_config import ChatBotConfig

ChatGenerationLike: TypeAlias = ChatGeneration | Iterator[ChatGeneration] | AsyncIterator[ChatGenerationChunk]

I = TypeVar(bound=Union[PromptValue, str, Sequence[MessageLikeRepresentation]])
O = TypeVar(bound=ChatGenerationLike)
C = TypeVar(bound=BaseChatModel)
S = TypeVar(bound=BaseChatModel)

class State(TypedDict):
    conversation_id: str
    datasource: Optional[str]
    vectorstore_docs: Optional[list[Document]]
    messages: Annotated[list, add_messages]

class RouteQuery(BaseModel):
    """Route a user query to most relevant datasource"""
    datasource: Literal[
        'vectorstore', 
        'multidoc_compare',
        'chat_model',
        'dataframe_analysis'
    ] = Field(..., description='Given a user question choose to route it to datasource')

class ChatBot(RunnableSerializable[I, O]):
    config: ChatBotConfig
    initial_state: State = Field(
        description='Initial graph state', 
        default_factory={}
    )
    chat_model: C = Field(
        description='Chat model for most operations', 
        default=None
    )
    safe_model: S = Field(
        description='Model to use for content safety',
        default=None
    )

    @field_validator('initial_state', mode='before')
    @classmethod
    def validate_initial_state(cls, value: State) -> State:
        # instance of State(messages=..., datasource=...)
        return value

    @model_validator(mode='after')
    def load_environment(self) -> Self:
        self.graph = StateGraph(State)
        inference_engine = HuggingFaceInference()
        self.chat_model = inference_engine.chat_model
        self.safety_model = inference_engine.safety_model
        self._compile()

        return Self
    
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
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatGenerationLike:
        ...

    async def ainvoke(self):
        pass

    def stream(self):
        pass

    async def astream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for event in self.astream(self.initial_state):
            for value in event.values():
                assert value['messages'].strip('\n') == ...
    
    def _compile(self):
        def guardrails(state: State) -> Dict[str, any]:
            ai_message = self.safety_model.invoke(state['messages'])
            return { 'messages': [ai_message] }
        self.graph.add_node('guardrails', guardrails)
        self.graph.add_edge(START, 'guardrails')

        def guardrails_condition(state: State) -> str:
            last_msg: AIMessage = state["messages"][-1]
            text = last_msg.content.lower()
            if 'safe' in text:
                return 'route_query'
            else:
                return 'not_safe'

        def not_safe_node(state: State):
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": "Your request cannot be processed. (Content flagged as not safe.)"
                    }
                ]
            }
        self.graph.add_conditional_edges(
            'guardrails',
            guardrails_condition,
            {
                "route_query": "route_query",
                "not_safe": "not_safe"
            }
        )
        self.graph.add_node('not_safe', not_safe_node)
        self.graph.add_edge('not_safe', END)

        def route_query_node(state: State):
            if state.datasource.multidoc_compare:
                chosen = 'multidoc_compare'
            
            if not chosen:
                system = """You are an expert at routing a user question to a vectorstore or wikipedia.
                    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
                    Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
                route_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        ("human", "{question}"),
                    ]
                )
                RouteQuery
                ai_response = self.chat_model.invoke(route_prompt)

                try:
                    parsed = json.loads(ai_response.content)
                    chosen = parsed["datasource"]
                except:
                    chosen = "chat_model"  # fallback if JSON parse fails
            
            return {
                "messages": [ai_response],
                "datasource": chosen
            }
        self.graph.add_node('route_query', route_query_node)

        def route_query_condition(state: State) -> str:
            datasource = state.get("datasource", "chat_model")
            return datasource
        self.graph.add_conditional_edges(
            'route_query',
            route_query_condition,
            {
                "vectorstore": "vectorstore",
                "multidoc_compare": "multidoc_compare",
                "chat_model": "chat_model",
                "dataframe_analysis": "dataframe_analysis"
            }
        )

        self.graph.add_node('vectorstore', vectorstore_node)
        self.graph.add_node('multidoc_compare', multidoc_compare_node)
        self.graph.add_node('chat_model', chat_model_node)
        self.graph.add_node('dataframe_analysis', dataframe_analysis_node)

        self.graph.add_edge('vectorstore', END)
        self.graph.add_edge('multidoc_compare', END)
        self.graph.add_edge('chat_model', END)
        # actually i don't want the dataframe_analysis to directly end
        # i want it to validate the result is good
        self.graph.add_edge('dataframe_analysis', evalaute_tool_response)

        return self.graph.compile()