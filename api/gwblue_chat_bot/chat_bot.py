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
    AsyncGenerator,
    Sequence,
    TypeAlias,
    override,
    Self,
    Any,
    List,
    Tuple,
)
import os
import json
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
    RunnableBinding,
)
from langchain_core.tracers.schemas import Run
from langchain_core.prompts import (
    BasePromptTemplate, 
    PromptTemplate, 
    ChatPromptTemplate,
)
from langchain_core.prompt_values import PromptValue, ChatPromptValue
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_SEPARATOR,
    DEFAULT_DOCUMENT_PROMPT,
)
from langchain_core.prompts import format_document
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    AIMessageChunk,
)
from langchain_core.messages import AIMessage, MessageLikeRepresentation
from langchain_core.messages.utils import AnyMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import RetrieverLike
from langchain_redis import RedisConfig
from redisvl.query.filter import Tag, FilterExpression
from pydantic import BaseModel, Field, model_validator, ConfigDict
from pymongo import DESCENDING

from .graph_state import State
from .language_models.huggingface_hub import HuggingFaceHub
from .chat_bot_config import ChatBotConfig
from ..gwblue_huggingface.huggingface_transformer_tokenizers import (
    BaseLocalTokenizer,
    BaseChatTokenizer,
)
# from .local_tools.route_query_tool import RouteQueryTool
from ..gwblue_vectorstores.redis.config import VectorStoreSchema
from ..gwblue_vectorstores.redis.multimodal_vectorstore import MultiModalVectorStore
from .prompts import registry
from .message_history import (
    MongoMessageHistorySchema,
    MongoMessageHistory,
)
from ..gwblue_vectorstores.redis.docstore import RedisDocStore
from ..gwblue_retrievers.streaming_parent_document_retriever import StreamingParentDocumentRetriever
from .semantic_scaler import Sigmoid

ChatGenerationLike: TypeAlias = (
    ChatGeneration | Iterator[ChatGeneration] | AsyncIterator[ChatGenerationChunk]
)

I = TypeVar("I", bound=Union[PromptValue, str, Sequence[MessageLikeRepresentation]])
O = TypeVar("O", bound=ChatGenerationLike)
C = TypeVar("C", bound=BaseChatModel)
S = TypeVar("S", bound=BaseChatModel)


def _clamp_temperature(temp: float) -> float:
    if temp < 1.0:
        return 1.0
    elif temp >= 5.0:
        return 0.0
    else:
        return max(0.0, min(1.0, temp))


def _textualize_model_input(content: Any) -> str:
    if isinstance(content, str):
        return content

    elif isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                text_chunks.append(text_value)
        return " ".join(text_chunks)

    else:
        return ""


def _chunk_pairs(pairs: List[Dict[str, Any]]):
    if len(pairs) % 2 != 0:
        raise ValueError("`pairs` must contain an even number of elements.")
    for i in range(0, len(pairs), 2):
        yield pairs[i : i + 2]


class StreamingResponse(BaseModel):
    type: str
    content: str
    token_usage: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    vector_metadata: List[Dict[str, Any]]
    session_id: str
    message_id: str


class ChatBot(RunnableSerializable[I, O]):
    config: ChatBotConfig
    graph: Optional[CompiledStateGraph] = None
    chat_model: BaseChatModel = Field(default=None, exclude=True)
    retry_model: BaseChatModel = Field(default=None, exclude=True)
    safety_model: BaseChatModel = Field(default=None, exclude=True)
    embeddings: Embeddings = Field(default=None, exclude=True)
    vector_store: MultiModalVectorStore = Field(default=None, exclude=True)
    message_history: MongoMessageHistory = Field(default=None, exclude=True)

    alt: Optional[bool] = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def load_environment(self) -> Self:
        graph = StateGraph(State)

        hf_hub = HuggingFaceHub(config=self.config, model_types={})
        platform = {
            "hf-inference": hf_hub,
            "vllm": hf_hub,
        }

        self.chat_model = platform[self.config.llm.provider]("chat_model")
        self.safety_model = platform[self.config.guardrails.provider]("guardrails")
        self.embeddings = platform[self.config.embeddings.provider]("embeddings")

        if self.alt:
            self.chat_model.llm = self.chat_model.llm.bind(
                temperature=_clamp_temperature(
                    self.config.llm.parameters["temperature"]
                ),
                seed=42,
            )

        config = RedisConfig(
            **{
                "redis_client": self.config.vectorstore.client,
                "metadata_schema": self.config.vectorstore.metadata_schema,
                "embedding_dimensions": self.embeddings.tokenizer.vector_dimension_length,
                **VectorStoreSchema().model_dump(),
            }
        )

        self.vector_store = MultiModalVectorStore(self.embeddings, config=config)

        message_schema = MongoMessageHistorySchema(
            session_id=self.config.message_history.session_id,
            connection_string=self.config.message_history.url,
            database_name=self.config.message_history.name,
            collection_name=self.config.message_history.collection_name,
            session_id_key=self.config.message_history.session_id_key,
        )
        self.message_history = MongoMessageHistory(message_schema)

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

    async def invoke(
        self,
        input: I,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> O:
        """Implementation soon"""
        ...

    async def ainvoke(
        self,
        input: I,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> O:
        """Implementation soon"""
        ...

    async def stream(
        self,
        input: I,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> O:
        """Implementation soon"""
        ...

    async def astream(
        self,
        input: I,
        config: Optional[Dict[str, Any]] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingResponse, None]:
        if isinstance(input, ChatPromptValue):
            input = input.to_messages()

        state = {
            "messages": input,
            "metadata": config["metadata"].get("vector_metadata", []),
            "retrieval_mode": config["configurable"].get(
                "retrieval_mode", "similarity"
            ),
        }

        async for event in self.graph.astream(state, config, stream_mode="messages"):
            ai_message_chunk, state_args = event
            if not isinstance(ai_message_chunk, AIMessageChunk):
                continue

            yield StreamingResponse(
                **{
                    "type": ai_message_chunk.__class__.__name__,
                    "content": ai_message_chunk.content,
                    "token_usage": ai_message_chunk.additional_kwargs.get(
                        "token_usage", {}
                    ),
                    "tool_calls": ai_message_chunk.additional_kwargs.get(
                        "tool_calls", []
                    ),
                    "vector_metadata": state_args.get("vector_metadata", []),
                    "session_id": str(self.config.message_history.session_id),
                    "message_id": ai_message_chunk.additional_kwargs.get("uuid", ""),
                }
            )

    @staticmethod
    def preprompt_filter(state: State, metadata: Dict[str, Any]) -> RunnableLambda:
        """
        The prompt templates in registry specifies a system prompt, but _aenter_chat_chain
        also adds system prompt to mongo directly and stuffs it in chat history. So
        we remove the duplicate system prompt
        """
        def create_preprompt_filter(input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **input_data,
                "chat_history": [
                    message
                    for message in input_data.get("chat_history", [])
                    if not isinstance(message, SystemMessage)
                    or not message.additional_kwargs.get("preprompt", False)
                ],
            }

        return RunnableLambda(create_preprompt_filter).with_config(
            run_name=f"filter_preprompt_chain_{state['route']}", metadata=metadata
        )

    @staticmethod
    def create_filter_expression(metadata: Dict[str, Any]) -> FilterExpression:
        from functools import reduce
        import operator

        tag_expressions = [Tag(key) == str(value) for key, value in metadata.items()]
        filter_expression = reduce(operator.and_, tag_expressions)
        return filter_expression

    def token_trimmer(
        self,
        max_total_tokens: Optional[int] = None,
        *,
        buffer_tokens: int = 200,
        tokenizer: Optional[BaseChatTokenizer] = None,
    ) -> RunnableLambda:
        def trim_tokens(chat_prompt: ChatPromptValue) -> ChatPromptValue:
            """
            Note depending on the chat template, the chat template may
            dynamically add its own system prompt to the request sent
            to model, such as llama guard, which injects a system prompt
            with your message and, thus, increases token count.
            """
            local_tokenizer = tokenizer if tokenizer else self.chat_model.tokenizer

            chosen_max = max_total_tokens
            if chosen_max is None:
                chosen_max = (
                    local_tokenizer.max_batch_tokens_forward_pass
                    - (local_tokenizer.max_new_tokens or 0)
                )

            messages: list[BaseMessage] = chat_prompt.to_messages()
            if any(msg.content.startswith("data:image/") for msg in messages):
                raise ValueError("Apply trimmer after image-to-text translation")
            
            budget = chosen_max - buffer_tokens

            token_ids = local_tokenizer.to_chat_template_ids(messages)
            total_tokens = len(token_ids)

            if total_tokens <= budget:
                return chat_prompt

            while len(messages) > 1:
                token_ids = local_tokenizer.to_chat_template_ids(messages)
                total_tokens = len(token_ids)
                if total_tokens <= budget:
                    break

                messages.pop(0)

            token_ids = local_tokenizer.to_chat_template_ids(messages)
            total_tokens = len(token_ids)

            if total_tokens > budget and len(messages) == 1:
                return AIMessage(
                    content=(
                        f"Your message is too long ({total_tokens} tokens)! "
                        f"Max allowed is {budget}."
                    ),
                    additional_kwargs={"exceeded_token_budget": True},
                )

            return ChatPromptValue(messages=messages)

        return RunnableLambda(trim_tokens)

    def create_generic_chain(self, state: State, system_prompt: str) -> Runnable:
        """
        Conform to `answer` key produced by `create_retrieval_chain`
        using a custom output parser

        Important note: passing only content in output parser instead of entire
        `AIMessage` means that any metadata associated with that AIMessage is lost,
        scuh as token_usage or logprobs
        """
        preprompt = self.preprompt_filter(state, {})
        
        answer_parser = RunnableLambda(
            lambda ai_message: {"answer": ai_message.content}
        )
        
        trimmer = self.token_trimmer(
            self.chat_model.tokenizer.max_batch_tokens_forward_pass - self.chat_model.tokenizer.max_new_tokens, 
            buffer_tokens=0
        )
        chain = (
            preprompt
            | registry["chat_prompt_with_history"](system_prompt)
            | trimmer
            | self.chat_model
            | answer_parser
        )

        return chain.with_config(run_name=f"generic_chat_model_chain_{state['route']}")

    def create_history_aware_retriever(
        self,
        retriever: RetrieverLike,
        prompt: BasePromptTemplate,
        preprompt_filter: Optional[Runnable] = None,
    ) -> Runnable:
        """Custom implementation to handle preprompt messages"""
        model_binding = self.chat_model.bind(stream=False)

        def validate_history(input_data: Dict[str, Any]) -> bool:
            return not input_data.get("chat_history")

        # trimmer = self.token_trimmer(
        #     self.chat_model.tokenizer.max_batch_tokens_forward_pass - self.chat_model.tokenizer.max_new_tokens, 
        #     buffer_tokens=0, 
        # )

        retrieve_documents = (
            preprompt_filter or RunnablePassthrough()
        ) | RunnableBranch(
            (
                validate_history,
                (lambda input_data: input_data["input"]) | retriever,
            ),
            # prompt | trimmer | model_binding | StrOutputParser() | retriever,
            prompt | model_binding | StrOutputParser() | retriever,
        ).with_config(
            run_name="history_aware_retriever_chain"
        )

        return retrieve_documents

    def create_stuff_documents_chain(
        self,
        prompt: BasePromptTemplate,
        preprompt_filter: Optional[Runnable] = None,
    ) -> Runnable[Dict[str, Any], Any]:
        """Custom implementation to handle preprompt messages"""
        def format_multimodal_chunks(content: str) -> str:
            multimodal_template = self.chat_model.tokenizer.multimodal_template(content)
            model_binding = self.chat_model.bind(stream=False)            
            ai_message = model_binding.invoke(multimodal_template)
            return ai_message.content
        
        def format_doc(doc: Document, doc_prompt: str) -> str:
            content = doc.page_content
            if content.startswith("data:image/"):
                page_content = format_multimodal_chunks(content)
                doc = Document(page_content=page_content)
            return format_document(doc, doc_prompt)

        def format_docs(inputs: dict) -> str:
            """
            Page number may already be included in vectorized text
            but may not be included in vectorized images. So we add
            if it does not already exist
            """
            doc_prompts = []
            for doc in inputs["context"]:
                if "page_number" in doc.metadata:
                    page_label = f"Page {doc.metadata['page_number']}"

                    if not doc.page_content.startswith(page_label):
                        doc_prompt = PromptTemplate(
                            input_variables=["page_number", "page_content"],
                            template="Page {page_number}: {page_content}"
                        )
                    doc_prompts.append(format_doc(doc, doc_prompt))
                else:
                    doc_prompts.append(format_doc(doc, DEFAULT_DOCUMENT_PROMPT))

            return DEFAULT_DOCUMENT_SEPARATOR.join(doc_prompts)

        # trimmer = self.token_trimmer(
        #     self.chat_model.tokenizer.max_batch_tokens_forward_pass - self.chat_model.tokenizer.max_new_tokens, 
        #     buffer_tokens=0
        # )

        return (
            (preprompt_filter or RunnablePassthrough())
            | RunnablePassthrough.assign(context=format_docs).with_config(
                run_name="format_inputs"
            )
            | prompt
            # | trimmer
            | self.chat_model
            | StrOutputParser()
        ).with_config(run_name="stuff_documents_chain")

    def create_context_aware_chain(self, state: State) -> Runnable:
        metadata = state["metadata"][0]
        system_prompt = state["messages"][0].content

        filter_expression = self.create_filter_expression(metadata)
        search_kwargs = {
            "k": int(round(Sigmoid.logistic(
                x=self.chat_model.tokenizer.sequence_length_forward_pass, 
                L=120, 
                a=2e-5, 
                m=60000
            ))),
            "filter": filter_expression,
        }
        # if state["retrieval_mode"] == "similarity_score_threshold":
        #     search_kwargs["score_threshold"] = 0.8

        # TODO: abstract retriever functionality outside of ChatBot
        # retriever = self.vector_store.as_retriever(
        #     search_type=state["retrieval_mode"], search_kwargs=search_kwargs
        # ).with_config(
        #     tags=[f"create_context_aware_chain_{state['route']}"],
        #     metadata=metadata,
        # )
        retriever = StreamingParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=RedisDocStore(self.config.vectorstore.client),
            search_kwargs=search_kwargs,
        )

        history_aware_retriever = self.create_history_aware_retriever(
            retriever,
            registry["contextualized_template"](),
            preprompt_filter=self.preprompt_filter(state, metadata),
        )

        question_answer_chain = self.create_stuff_documents_chain(
            registry["qa_template"](system_prompt),
            preprompt_filter=self.preprompt_filter(state, metadata),
        )

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def create_multi_retriever_chain(
        self, 
        retrievers: Tuple[List[Runnable], Dict[str, Any]], 
        state: State
    ) -> Runnable:
        context_prompt = registry["contextualized_template"]()
        retriever_map = {
            f"Source {metadata['source']}": retriever
            for retriever, metadata in retrievers
        }
        parallel_retrieval = RunnableParallel(retriever_map)

        def combine_contexts(
            retrieved_results: dict, separator=DEFAULT_DOCUMENT_SEPARATOR
        ) -> list:
            combined_results = []
            for key, docs in retrieved_results.items():
                combined_docs = separator.join(doc.page_content for doc in docs)
                combined_results.append(
                    Document(page_content=f"Context from {key}:\n{combined_docs}")
                )

            return combined_results

        combine_contexts_runnable = RunnableLambda(combine_contexts).with_config(
            run_name="combine_context_chain"
        )

        retrieve_documents = self.create_history_aware_retriever(
            parallel_retrieval,
            context_prompt,
            preprompt_filter=self.preprompt_filter(state, {}),
        )

        return retrieve_documents | combine_contexts_runnable

    def create_multi_stuff_chain(self, state: State, system_prompt: str) -> Runnable:
        qa_template = registry["qa_template"](system_prompt)

        return self.create_stuff_documents_chain(
            qa_template, preprompt_filter=self.preprompt_filter(state, {})
        )

    def create_multicontext_aware_chain(self, state: State) -> Runnable:
        system_prompt = state["messages"][0].content
        retrievers = []

        for index, metadata in enumerate(state["metadata"]):
            filter_expression = self.create_filter_expression(metadata)
            search_kwargs = {
                "k": int(round(Sigmoid.logistic(
                    x=self.chat_model.tokenizer.sequence_length_forward_pass, 
                    L=120, 
                    a=2e-5, 
                    m=60000
                ))),
                "filter": filter_expression,
            }
            # if state["retrieval_mode"] == "similarity_score_threshold":
            #     search_kwargs["score_threshold"] = 0.8

            # retriever = self.vector_store.as_retriever(
            #     search_type=state["retrieval_mode"], search_kwargs=search_kwargs
            # ).with_config(
            #     tags=[f"create_context_aware_chain_{index}_{state['route']}"],
            #     metadata=metadata,
            # )
            retriever = StreamingParentDocumentRetriever(
                vectorstore=self.vector_store,
                docstore=RedisDocStore(self.config.vectorstore.client),
                search_kwargs=search_kwargs,
            ).with_config(
                tags=[f"create_context_aware_chain_{index}_{state['route']}"],
                metadata=metadata,
            )
            retrievers.append((retriever, metadata))

        multi_retriever_chain = self.create_multi_retriever_chain(retrievers, state)
        stuffing_chain = self.create_multi_stuff_chain(state, system_prompt)

        multicontext_aware_chain = (
            RunnablePassthrough.assign(
                context=multi_retriever_chain.with_config(run_name="retrieval_chain"),
            ).assign(answer=stuffing_chain)
        ).with_config(run_name="multicontext_aware_chain")

        return multicontext_aware_chain

    async def _aenter_chat_chain(
        self, run: Run, config: RunnableConfig, system_prompt: str
    ) -> Optional[SystemMessage]:
        """On start runnable listener"""
        collection = self.message_history.chat_message_history.collection

        document = collection.find_one(
            {
                "type": "system",
                "content": system_prompt,
                self.message_history._schema.session_id_key: self.message_history._schema.session_id,
            }
        )

        if document is None:
            await self.message_history.asystem(
                system_prompt, additional_kwargs={"preprompt": True}
            )
        else:
            history_data = json.loads(document["History"])

            additional_kwargs = history_data.get("data", {}).get(
                "additional_kwargs", {}
            )
            if not additional_kwargs.get("preprompt", False):
                await self.message_history.asystem(
                    system_prompt, additional_kwargs={"preprompt": True}
                )

    async def _aexit_chat_chain(self, run: Run, config: RunnableConfig) -> None:
        """On end runnable listener"""
        collection = self.message_history.chat_message_history.collection
        if (
            ai_message := collection.find_one(
                {
                    "type": {"$in": ["ai", "AIMessageChunk"]},
                    self.config.message_history.session_id_key: self.config.message_history.session_id,
                },
                sort=[("createdAt", DESCENDING)],
            )
        ) is not None:
            chain = registry["summarization_template"]() | self.chat_model.bind(
                stream=False
            )
            summary = await chain.ainvoke({"input": ai_message["content"]})
            self.message_history.chat_message_history.add_summary(summary.content)

    async def generate_with_history(
        self,
        state: State,
        chain: Runnable,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        system_prompt = state["messages"][0].content
        human_message = state["messages"][-2]

        if isinstance(human_message.content, str):
            input_dict = {"input": human_message.content}
        elif isinstance(human_message.content, list):
            input_dict = {"input": human_message.content[1]["text"]}

        async def on_start(run: Run, config: RunnableConfig):
            await self._aenter_chat_chain(run, config, system_prompt)

        async def on_end(run: Run, config: RunnableConfig):
            await self._aexit_chat_chain(run, config)

        chain_with_history = self.message_history.get(chain).with_alisteners(
            on_start=on_start, on_end=on_end
        )

        if (
            not config
            or "configurable" not in config.configurable
            or "session_id" not in config.configurable["configurable"]
        ):
            config = RunnableConfig(
                configurable={
                    "configurable": {
                        "session_id": self.config.message_history.session_id
                    }  # TODO: generalize to session_id
                }
            )
        chain_values = await chain_with_history.ainvoke(
            input_dict, config=config["configurable"]
        )
        return {"messages": [AIMessage(content=chain_values["answer"])]}

    def _compile(self, graph: StateGraph):
        async def guardrails(state: State) -> State:
            user_content = state["messages"][1].content
            sanitized_text = _textualize_model_input(user_content)
            
            chat_prompt = ChatPromptTemplate.from_messages([
                HumanMessage(content=sanitized_text)
            ])

            trimmer = self.token_trimmer(
                self.safety_model.tokenizer.max_batch_tokens_forward_pass - self.safety_model.tokenizer.max_new_tokens, 
                buffer_tokens=0, 
                tokenizer=self.safety_model.tokenizer,
            )
            model_binding = self.safety_model.bind(stream=False)
            
            def needs_model(input_data) -> bool:
                return isinstance(input_data, ChatPromptValue)
            
            chain = (
                chat_prompt
                | trimmer
                | RunnableBranch(
                    (needs_model, model_binding),
                    RunnablePassthrough(),
                )
            )

            ai_message = await chain.ainvoke({
                'input': sanitized_text
            })
            
            guardrails_message = AIMessage(
                content=ai_message.content,
                additional_kwargs={
                    **ai_message.additional_kwargs,
                    "guardrails": True
                },
            )
            return {**state, "messages": [guardrails_message]}

        def guardrails_condition(state: State) -> str:
            last_msg: AIMessage = state["messages"][-1]
            text = last_msg.content.lower().strip("\n")

            if last_msg.additional_kwargs.get("exceeded_token_budget"):
                return "exceeded_token_budget"
            
            if "unsafe" in text:
                return "not_safe"
            elif "safe" in text:
                return "prefill_system_prompt"
            else:
                return "not_safe"

        def not_safe(_: State):
            return {
                "messages": [
                    AIMessageChunk(
                        content="Your request cannot be processed. (Content flagged as not safe.)"
                    )
                ]
            }

        async def exceeded_token_budget(state: State) -> State:
            last_msg: AIMessage = state["messages"][-1]

            return {
                "messages": [
                    AIMessageChunk(
                        content=(
                            last_msg.content
                        )
                    )
                ]
            }

        async def prefill_system_prompt(state: State) -> State:
            import copy

            system_message = state["messages"][0]
            human_message = state["messages"][-2]

            if not isinstance(human_message.content, list):
                return state

            pairs = list(_chunk_pairs(human_message.content))
            batches = []
            for pair in pairs:
                cloned_pair = [copy.deepcopy(item) for item in pair]
                cloned_pair[1]["text"] = "Describe the image in-depth."
                messages = [
                    SystemMessage(content=system_message.content),
                    HumanMessage(content=cloned_pair),
                ]
                batches.append(messages)

            non_streaming_model = self.chat_model.bind(stream=False)
            ai_messages = await non_streaming_model.abatch(batches)

            labeled_descs = []
            for i, ai_message in enumerate(ai_messages, start=1):
                desc = ai_message.content.strip()
                labeled_descs.append(f"**Image #{i}**:\n{desc}")
            combined_desc = "\n\n".join(labeled_descs)

            old_prompt = system_message.content
            updated_prompt = (
                old_prompt
                + "\n\n"
                + "If the user asks about images, pictures, or photos, then use the following descriptions as if they were the actual images:\n"
                + combined_desc
            )

            new_system_message = system_message.copy(update={"content": updated_prompt})
            state["messages"][0] = new_system_message

            return state

        async def route_query(state: State):
            """
            Account for scenarios:
            - 'Explain this document' (where this is not specified and refers to upload)
            - 'Compare these two documents' (where these is not specified and refers to multiple uploads)
            - 'Explain why it has impact' (where it refers to previously vectorized data)
            - 'Compare this and that' (where this is current upload and that is previously vectorized data)
            - 'Explain something' (where no vector data but can tap into pretrained corpus of LLM)
            """
            metadata = state["metadata"]

            if len(metadata) > 1:
                return {"route": "multi_doc_prompt", **state}

            if len(metadata) == 1 and "source" in metadata[0]:
                return {"route": "single_doc_prompt", **state}

            human_prompt = _textualize_model_input(state["messages"][-2].content)
            vector_filter = metadata[0]

            filter_expression = (
                Tag(self.config.message_history.session_id_key)
                == str(self.config.message_history.session_id)
            ) & (Tag("uuid") == vector_filter["uuid"])
            relevant_docs_with_score = (
                await self.vector_store.asimilarity_search_with_score(
                    query=human_prompt,
                    k=12,
                    filter=filter_expression,
                )
            )
            file_to_best_chunk = defaultdict(lambda: (None, float("inf")))
            for doc, dist in relevant_docs_with_score:
                fname = doc.metadata.get("source", "")
                if dist < file_to_best_chunk[fname][1]:
                    file_to_best_chunk[fname] = (doc, dist)

            best_metadata = [
                doc.metadata
                for (doc, _) in file_to_best_chunk.values()
                if doc is not None
            ]

            num_files = len(best_metadata)
            if num_files > 1:
                route = "multi_doc_prompt"
            elif num_files == 1:
                route = "single_doc_prompt"
            else:
                route = "pretrained_corpus_prompt"

            return {
                **state,
                "route": route,
                "metadata": best_metadata,
            }

        def route_query_condition(state: State) -> str:
            return state["route"]

        async def single_doc_prompt(state: State) -> dict:
            """
            Generate prompt for single document
            """
            chain = self.create_context_aware_chain(state)
            messages = await self.generate_with_history(state, chain)
            return messages

        async def multi_doc_prompt(state: State) -> Dict[str, Any]:
            """
            Generate prompt for multiple documents
            """
            chain = self.create_multicontext_aware_chain(state)
            messages = await self.generate_with_history(state, chain)
            return messages

        async def pretrained_corpus_prompt(state: State) -> Dict[str, Any]:
            """
            Generate prompt for pretrained corpus
            """
            system_prompt = state["messages"][0].content
            chain = self.create_generic_chain(state, system_prompt)
            messages = await self.generate_with_history(state, chain)
            return messages

        graph.add_node("guardrails", guardrails)
        graph.add_node("not_safe", not_safe)
        graph.add_node("exceeded_token_budget", exceeded_token_budget)
        graph.add_node("prefill_system_prompt", prefill_system_prompt)
        graph.add_node("route_query", route_query)
        graph.add_node("single_doc_prompt", single_doc_prompt)
        graph.add_node("multi_doc_prompt", multi_doc_prompt)
        graph.add_node("pretrained_corpus_prompt", pretrained_corpus_prompt)

        graph.add_edge(START, "guardrails")
        graph.add_conditional_edges(
            "guardrails",
            guardrails_condition,
            {
                "exceeded_token_budget": "exceeded_token_budget",
                "prefill_system_prompt": "prefill_system_prompt", 
                "not_safe": "not_safe",

            },
        )
        graph.add_edge("prefill_system_prompt", "route_query")
        graph.add_edge("not_safe", END)
        graph.add_edge("exceeded_token_budget", END)
        graph.add_conditional_edges(
            "route_query",
            route_query_condition,
            {
                #'vectorstore': 'vectorstore',
                #'dataframe_tool': 'dataframe_tool',
                "single_doc_prompt": "single_doc_prompt",
                "multi_doc_prompt": "multi_doc_prompt",
                "pretrained_corpus_prompt": "pretrained_corpus_prompt",
            },
        )
        # graph.add_edge('vectorstore', END)
        # graph.add_edge('dataframe_tool', END)
        graph.add_edge("single_doc_prompt", END)
        graph.add_edge("multi_doc_prompt", END)
        graph.add_edge("pretrained_corpus_prompt", END)
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
