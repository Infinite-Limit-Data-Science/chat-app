import os
import asyncio
from typing import Any, List, Optional, Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models.huggingface import ChatHuggingFace
from transformers import AutoTokenizer
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk
from typing import Iterator, AsyncIterator

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_redis import RedisConfig as Config, RedisVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

import logging
from typing import Annotated, List, Optional, Union
from fastapi import ( 
    APIRouter, 
    status, 
    Request, 
    Query, 
    Body, 
    Form, 
    Depends, 
    File, 
    UploadFile, 
    logger
)
from auth.bearer_authentication import get_current_user
from routes.chats import chat 
from routes.configs import (
    get_current_models, 
    get_current_embedding_models, 
    get_prompt_template, 
    
)
from routes.uploads import ingest_file
from orchestrators.chat.llm_models.llm import LLM
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from repositories.conversation_mongo_repository import ConversationMongoRepository as ConversationRepo
from models.conversation import (
    ConversationSchema,
    CreateConversationSchema,
    ConversationCollectionSchema, 
    UpdateConversationSchema,
)
from models.message import MessageSchema
from fastapi.responses import StreamingResponse



from fastapi import ( 
    APIRouter, 
    status, 
    Request, 
    Query, 
    Body, 
    Form, 
    Depends, 
    File, 
    UploadFile, 
    logger
)
from fastapi.responses import StreamingResponse
from auth.bearer_authentication import get_current_user

from orchestrators.chat.llm_models.model_proxy import ModelProxy as LLMProxy
from orchestrators.doc.embedding_models.model_proxy import ModelProxy as EmbeddingProxy
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractVectorStore
from orchestrators.doc.vector_stores.factories import FACTORIES as V_FACTORIES

# ISSUE IS NOT LLM OR CHAT MODEL OR INGESTOR OR VECTOR RETRIEVER

def rag_chain2(llm, vector_store: AbstractVectorStore, uuid, conversation_id):
    CONTEXTUALIZED_TEMPLATE = """
    Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
    """
    contextualized_template = ChatPromptTemplate.from_messages([
        ('system', CONTEXTUALIZED_TEMPLATE),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])

    QA_TEMPLATE = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    {context}
    """
    qa_template = ChatPromptTemplate.from_messages([
        ('system', QA_TEMPLATE),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])
 

    vector_filter = { 
        'uuid': str(uuid), 
        'conversation_id': str(conversation_id),
    }
    retriever = vector_store.retriever()
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualized_template)
    question_answer_chain = create_stuff_documents_chain(llm, qa_template)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

router = APIRouter(
    prefix='/tests', 
    tags=['test'],
    dependencies=[Depends(get_current_user)],
)
@router.post(
    '/',
    response_description="Add new conversation",
    status_code=status.HTTP_201_CREATED,
)
async def llm_stream2(    
    request: Request,
    content: str = Form(...),
    # conversation: Annotated[ConversationSchema, Form()],
    # message: Annotated[MessageSchema, Form()],
    models: List[LLM] = Depends(get_current_models),
    embedding_models: List[BaseEmbedding]  = Depends(get_current_embedding_models),
    prompt_template: str = Depends(get_prompt_template),
    upload_file: Optional[UploadFile] = File(None)):
    conversation_schema = CreateConversationSchema(uuid=request.state.uuid)

    if (
        created_conversation_id := await ConversationRepo.create(conversation_schema=conversation_schema)
    ) is not None:
        data = { 'uuid': conversation_schema.uuid, 'conversation_id': created_conversation_id }
        if upload_file:
            await ingest_file(embedding_models, upload_file, data)
        message_schema = MessageSchema(type='human', content=content, conversation_id=created_conversation_id)
        llm = LLMProxy(models).get()
        chat_llm = llm.endpoint_object
        embeddings = EmbeddingProxy(embedding_models)
        vector_store: AbstractVectorStore = V_FACTORIES['redis'](embeddings, {
            'metadata': {
                'uuid': conversation_schema.uuid, 
                'conversation_id': created_conversation_id,
                'schema': [
                    {
                        'name': 'uuid', 
                        'type': 'tag'
                    },
                    {
                        'name': 'conversation_id', 
                        'type': 'tag'
                    },
                ]
            },
            'configurable': {
                'session_id': created_conversation_id,
            }
        })

        retriever_chain = rag_chain2(chat_llm, vector_store, conversation_schema.uuid, created_conversation_id)
        chain_with_history = RunnableWithMessageHistory(
            retriever_chain,
            get_session_history,
            input_messages_key='input', 
            history_messages_key='chat_history',
            output_messages_key= 'answer')

        async def llm_stream():
            stop_token = "<|eot_id|>"    
            async for s in chain_with_history.astream(
                            {'input': 'Two versions of word embeddings are provided, both in Word2Vec C binary format. What are they?'},
                            config={'session_id': 'chat1'}):
                if 'answer' in s:
                    s_content = s['answer']

                    if stop_token in s_content:
                        s_content = s_content.replace(stop_token, "")

                    yield s_content

    return StreamingResponse(llm_stream(), media_type="text/plain", headers={"X-Accel-Buffering": "no"})



# THIS HERE IS WORKING WITH EVERYTHING ELSE REMAINING THE SAME AS BEFORE
# async def rag_astream(self, chat_llm: BaseChatModel, message: str):
#     chain = self.create_rag_chain_advanced(chat_llm)
#     # chain_with_history = self._message_history.get(chain, True)
#     # chain_with_history = chain_with_history.with_alisteners(
#     #     on_start=self._aenter_chat_chain,
#     #     on_end=self._aexit_chat_chain)
#     # config=self.runnable_config()
#     async def llm_astream():
#         stop_token = "<|eot_id|>"
#         async for s in chain.astream(
#             {'input': message, 'chat_history': []}):
#             if 'answer' in s:
#                 s_content = s['answer']
#                 if stop_token in s_content:
#                     s_content = s_content.replace(stop_token, "")

#                 yield s_content
#     return llm_astream

# THE PROBLEM IS THE chain_with_history !!!!

# THE PROBLEM IS WHEN I START THE SERVER, AND ASK QUESTION WITH TEXT DOC FIRST TIME. IT GIVES BAD ANSWER. BUT THEN WHEN I ASK A SECOND TIME AND MANY TIMES AFTER IT GIVES RIGHT ANSWER.