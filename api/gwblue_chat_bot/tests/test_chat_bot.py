import os
import json
import pytest
import asyncio
import base64
from pathlib import Path
from typing import Dict, Any, List, Generator, Iterator
from redis.client import Redis
from redis.connection import ConnectionPool
from bson import ObjectId
from uuid import uuid4
from pymongo import MongoClient
from pymongo.database import Database
from langchain_core.prompts.chat import ChatPromptTemplate, PromptValue
from langchain_core.runnables import RunnableConfig, RunnableParallel
from ..chat_bot_config import (
    LLMConfig,
    EmbeddingsConfig,
    RedisVectorStoreConfig,
    MongoMessageHistoryConfig,
    ChatBotConfig,
)
from ..chat_bot import ChatBot
from ...gwblue_huggingface import HuggingFaceEmbeddings
from ...gwblue_ingestion_pipeline import (
    LazyPdfIngestor,
    LazyWordIngestor,
)
from ...gwblue_huggingface.huggingface_transformer_tokenizers import (
    get_tokenizer_by_prefix,
    get_chat_tokenizer_by_prefix,
    BaseLocalTokenizer,
)

def _model_config(model_type: str, model_name: str) -> Dict[str, str]:
    models = json.loads(os.environ[model_type])
    model = next((model for model in models if model["name"] == model_name), None)
    if not model:
        raise ValueError(f"Model {model_name} does not exist in {model_type}")

    return {
        "name": model["name"],
        "url": model["endpoints"][0]["url"],
        "provider": model["endpoints"][0]["provider"],
    }


@pytest.fixture
def redis_client() -> Redis:
    redis_client = Redis.from_pool(
        ConnectionPool.from_url(
            os.environ["REDIS_URL"], max_connections=50, socket_timeout=30.0
        )
    )
    return redis_client

@pytest.fixture
def vlm_tokenizer() -> BaseLocalTokenizer:
    return get_tokenizer_by_prefix("TIGER-Lab/VLM2Vec-Full")

@pytest.fixture
def llama_11B_vi_tokenizer() -> BaseLocalTokenizer:
    return get_chat_tokenizer_by_prefix("meta-llama/Llama-3.2-11B-Vision-Instruct")

@pytest.fixture
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        provider=config["provider"],
        model=config["name"],
    )

@pytest.fixture
def chat_bot_config(
    redis_client: Redis,
    vlm_tokenizer: BaseLocalTokenizer
) -> ChatBotConfig:
    config = _model_config("MODELS", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    llm_config = LLMConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "parameters": {"temperature": 0.8},
            "provider": config["provider"],
        }
    )

    config = _model_config("MODELS", "meta-llama/Llama-Guard-3-8B")
    guardrails_config = LLMConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "parameters": {"temperature": 0},
            "provider": config["provider"],
        }
    )

    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")
    embeddings_config = EmbeddingsConfig(
        **{
            "model": config["name"],
            "endpoint": config["url"],
            "token": os.environ["TEST_AUTH_TOKEN"],
            "max_batch_tokens":  vlm_tokenizer.max_batch_tokens_forward_pass,
            "provider": config["provider"],
        }
    )

    metadata_schema = json.loads(os.environ["VECTOR_STORE_SCHEMA"])
    vectorstore_config = RedisVectorStoreConfig(
        **{
            "client": redis_client,
            "metadata_schema": metadata_schema,
        }
    )

    message_history_config = MongoMessageHistoryConfig(
        name=os.environ["DATABASE_NAME"],
        url=os.environ["MONGODB_URL"],
        collection_name="messages",
        session_id_key="conversation_id",
    )

    return ChatBotConfig(
        llm=llm_config,
        embeddings=embeddings_config,
        guardrails=guardrails_config,
        vectorstore=vectorstore_config,
        message_history=message_history_config,
    )

assets_dir = Path(__file__).parent / "assets"

@pytest.fixture
def nvidiaan_pdf_path() -> Path:
    return assets_dir / "NVIDIAAn.pdf"

@pytest.fixture
def nvidia_1tri_fiscal_2025() -> Path:
    return assets_dir / "Nvidia-1tri-fiscal-2025.pdf"

@pytest.fixture
def teams_to_consider_word_path() -> Path:
    return assets_dir / "Teams to Consider.docx"

@pytest.fixture
def calculus_pdf_path() -> Path:
    return assets_dir / "Calculus.pdf"

@pytest.fixture
def developer_word_path() -> Path:
    return assets_dir / "ruby-rails-developer-v2.docx"


@pytest.fixture
def cloud_engineer_word_path() -> Path:
    return assets_dir / "senior-cloud-engineer-v1.docx"

@pytest.fixture
def devops_engineer_word_path() -> Path:
    return assets_dir / "senior-devops-engineer-v4.docx"

@pytest.fixture
def calculus_book1_path() -> Path:
    return assets_dir / "CalculusBook1.pdf"

@pytest.fixture
def calculus_book2_path() -> Path:
    return assets_dir / "CalculusBook2.pdf"

@pytest.fixture
def calculus_book3_path() -> Path:
    return assets_dir / "CalculusBook3.pdf"

@pytest.fixture
def doc_compare1_pdf_path() -> Path:
    return assets_dir / "25M06-02C.pdf"

@pytest.fixture
def doc_compare2_pdf_path() -> Path:
    return assets_dir / "2025Centene.pdf"
    
@pytest.fixture
def nvidia_1tri_fiscal_2025_path() -> Path:
    return assets_dir / "Nvidia-1tri-fiscal-2025.pdf"

@pytest.fixture
def guitar_jpg_path() -> Path:
    return assets_dir / "guitar.jpg"

@pytest.fixture
def baby_jpg_path() -> Path:
    return assets_dir / "baby.jpg"

@pytest.fixture
def jpeg_pdf_path() -> Path:
    return assets_dir / "jpeg.pdf"

@pytest.fixture
def arag_ignite_pdf_path() -> Path:
    return assets_dir / "ARAG Ignite 2025 Flier 1.pdf"

@pytest.fixture
def genesys_contract_pdf_path() -> Path:
    return assets_dir / "64654-genesys.pdf"

@pytest.fixture
def messages_db(chat_bot_config: ChatBotConfig) -> Database:
    url = chat_bot_config.message_history.url
    database = chat_bot_config.message_history.name

    client = MongoClient(url)
    db = client[database]
    return db

@pytest.fixture
def conversation_doc(messages_db: Database) -> Generator[Dict[str, Any], None, None]:
    conversations = messages_db["conversations"]

    attributes = {
        "_id": ObjectId(),
        "sessionId": str(uuid4()),
    }

    doc = conversations.insert_one(attributes)
    attributes["_id"] = doc.inserted_id

    yield attributes

    # conversations.delete_one({'_id': attributes['_id']})

@pytest.fixture
def message_metadata(conversation_doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "uuid": conversation_doc["sessionId"],
        "conversation_id": conversation_doc["_id"],
    }

# @pytest.mark.asyncio
# async def test_single_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     nvidiaan_pdf_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "NVIDIAAn.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         nvidiaan_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Summarize the document"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "nvidia" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_single_large_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     calculus_book1_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "CalculusBook1.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         calculus_book1_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Summarize the document"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "calculus" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_single_doc_prompt_with_trimming(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     nvidiaan_pdf_path: Path,
#     calculus_book1_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "NVIDIAAn.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         nvidiaan_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     from langchain_community.document_loaders import PyPDFLoader
#     loader = PyPDFLoader(
#         file_path=calculus_book1_path,
#         mode="single",
#         extraction_mode="plain",
#     )
#     docs = loader.load()
#     input = docs[0].page_content

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": f'Compare the document with the following input {input}' }, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "too long" in ai_content

# @pytest.mark.asyncio
# async def test_teams_to_consider_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     teams_to_consider_word_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "Teams to Consider.docx",
#     }

#     ingestor = LazyWordIngestor(
#         teams_to_consider_word_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         add_to_docstore=True,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "Who is Himanshu Mehta?"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "himanshu" in ai_content.lower()

#     ai_content = ""
#     streaming_resp = []
#     # THE OTHER POTENTIAL PROBLEM IS YOU MAY HAVE ONLY LIMITED 2024 TOKENS in RESPONSE
#     async for chunk in chain.astream(
#         {
#             "input": "Review the attached MS Word document throughly and list out all the teams listed under the Mandatory Teams section"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "guidewell" in ai_content.lower()

#     # add page number search, filename search by metadata
#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "how many teams are listed under Teams to Consider"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "team" in ai_content

# @pytest.mark.asyncio
# async def test_arag_ignite_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     arag_ignite_pdf_path: Path,
# ):
#     # THIS IS ERRORING OUT BECAUSE I BELIEVE THE IMAGE CHUNKS DONT HAVE DOC_KEYS AND THIS IS ENFORCING IT
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "ARAG Ignite 2025 Flier 1.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         arag_ignite_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         add_to_docstore=True,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "Summarize the document"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "arag" in ai_content.lower() 

@pytest.mark.asyncio
async def test_genesys_contract_doc_prompt(
    embeddings: HuggingFaceEmbeddings,
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
    genesys_contract_pdf_path: Path,
):
    chat_bot_config.message_history.session_id = conversation_doc["_id"]

    metadata = {
        **message_metadata,
        "conversation_id": str(message_metadata["conversation_id"]),
        "source": "64654-genesys.pdf",
    }

    ingestor = LazyPdfIngestor(
        genesys_contract_pdf_path,
        embeddings=embeddings,
        metadata=metadata,
        vector_config=chat_bot_config.vectorstore,
        add_to_docstore=True,
    )
    ids = await ingestor.ingest()
    print(ids)

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", "You're a helpful assistant"), ("human", "{input}")]
    )
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot    

    config = RunnableConfig(
        tags=[
            "chat_bot_run_test",
            f"uuid_${message_metadata['uuid']}",
            f"conversation_id_${message_metadata['uuid']}",
        ],
        metadata={"vector_metadata": [metadata]},
    )

    ai_content = ""
    streaming_resp = []
    async for chunk in chain.astream(
        {
            "input": "What is the quote expiration date"
        },
        config=config,
    ):
        print(f"Custom event ${chunk.content}")
        ai_content += chunk.content
        streaming_resp.append(chunk)

    assert "21" in ai_content.lower() 

    ai_content = ""
    streaming_resp = []
    async for chunk in chain.astream(
        {
            "input": "What is the ramp period?"
        },
        config=config,
    ):
        print(f"Custom event ${chunk.content}")
        ai_content += chunk.content
        streaming_resp.append(chunk)

    assert "ramp" in ai_content.lower() 

    ai_content = ""
    streaming_resp = []
    async for chunk in chain.astream(
        {
            "input": "What is the largest line item by dollar in the document?"
        },
        config=config,
    ):
        print(f"Custom event ${chunk.content}")
        ai_content += chunk.content
        streaming_resp.append(chunk)

    assert "dollar" in ai_content.lower() 

# @pytest.mark.asyncio
# async def test_calculus_book_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     calculus_book1_path: Path,
#     vlm_tokenizer,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "CalculusBook1.pdf",
#     }

#     from langchain_redis import RedisConfig
#     from langchain.retrievers import ParentDocumentRetriever
#     from langchain_text_splitters import RecursiveCharacterTextSplitter
#     from ...gwblue_vectorstores.redis import MultiModalVectorStore
#     from ...gwblue_vectorstores.redis.config import VectorStoreSchema
#     from ...gwblue_vectorstores.redis.docstore import RedisDocStore
#     from ...gwblue_document_loaders.loaders.extended_pypdf_loader import ExtendedPyPDFLoader
#     from ...gwblue_document_loaders.parsers.base64_blob_parser import Base64BlobParser
    
#     from langchain_text_splitters import CharacterTextSplitter 

#     loader = ExtendedPyPDFLoader(
#         calculus_book1_path,
#         extract_images=True,
#         images_parser=Base64BlobParser(),
#         images_inner_format="raw",
#         mode="page",
#     )

#     docs_stream = loader.lazy_load()
#     docs = list(docs_stream)
#     assert len(docs) > 0

#     for doc in docs:
#         doc.metadata = { **doc.metadata, **metadata }

#     config = RedisConfig(
#         **{
#             "redis_client": chat_bot_config.vectorstore.client,
#             "metadata_schema": chat_bot_config.vectorstore.metadata_schema, # now includes doc_id
#             "embedding_dimensions": 3072,
#             **VectorStoreSchema().model_dump(),
#         }
#     )

#     vectorstore = MultiModalVectorStore(embeddings, config=config)
#     docstore = RedisDocStore(chat_bot_config.vectorstore.client)
    
#     length_function=lambda text: len(vlm_tokenizer.tokenizer.encode(text))

#     # RecursiveCharacterTextSplitter repeatedly re-checks the text in smaller and smaller segments (via nested recursion) and calls your tokenizer each time to measure token length. On a very large document, that can become extremely expensive – it might encode overlapping slices of the text many times in a deep recursion loop.
#     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, length_function=length_function)
#     child_splitter = RecursiveCharacterTextSplitter(chunk_size=250, length_function=length_function)
#     retriever = ParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=docstore,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter
#     )
#     await retriever.aadd_documents(docs)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot    

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "similarity"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "What did he mean that we never have to go higher than quadratics?"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     print(f'AI CONTENT {ai_content}')
#     assert "21" in ai_content.lower() 

# @pytest.mark.asyncio
# async def test_multi_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     doc_compare1_pdf_path: Path,
#     doc_compare2_pdf_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadatas = [
#         {
#             **message_metadata,
#             "conversation_id": str(message_metadata["conversation_id"]),
#             "source": "25M06-02C.pdf",
#         },
#         {
#             **message_metadata,
#             "conversation_id": str(message_metadata["conversation_id"]),
#             "source": "2025Centene.pdf",
#         },
#     ]

#     ingestors = []
#     compare_file_paths = [
#         doc_compare1_pdf_path,
#         doc_compare2_pdf_path,
#     ]
#     for file_path, metadata in zip(compare_file_paths, metadatas):
#         ingestor = LazyPdfIngestor(
#             file_path,
#             embeddings=embeddings,
#             metadata=metadata,
#             vector_config=chat_bot_config.vectorstore,
#             embeddings_config=chat_bot_config.embeddings,
#         )
#         ingestors.append(ingestor)

#     tasks = [asyncio.create_task(ingestor.ingest()) for ingestor in ingestors]
#     ids: List[List[str]] = await asyncio.gather(*tasks)
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": metadatas},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Compare the two documents"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "health" in ai_content

# @pytest.mark.asyncio
# async def test_pretrained_corpus_prompt(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Tell me about the movie Memento."}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "memento" in ai_content.lower()

# # TODO: when users upload images, it should store images as vectors rather
# # than process them all and load them all into single prompt
# @pytest.mark.asyncio
# async def test_multimodal_image(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     baby_jpg_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     with baby_jpg_path.open("rb") as f:
#         base64_image = base64.b64encode(f.read()).decode("utf-8")
#     image_url = f"data:image/jpeg;base64,{base64_image}"

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You're a helpful assistant who can create text from images"),
#             ("human", [{"image_url": {"url": "{image_url}"}}, "{input}"]),
#         ]
#     )

#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "Describe the image.",
#             "image_url": image_url,
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "image" in ai_content.lower()


# @pytest.mark.asyncio
# async def test_message_history(
#     chat_bot_config: ChatBotConfig,
#     embeddings: HuggingFaceEmbeddings,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     nvidiaan_pdf_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "NVIDIAAn.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         nvidiaan_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Summarize the document"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     # follow up Q&A
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},  # without 'source' key now
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "How were GAAP earnings per diluted share?"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "gaap" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_message_history2(
#     chat_bot_config: ChatBotConfig,
#     embeddings: HuggingFaceEmbeddings,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     teams_to_consider_word_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "Teams to Consider.docx",
#     }

#     ingestor = LazyWordIngestor(
#         teams_to_consider_word_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "Review the attached MS Word document throughly and list out all the teams listed under the Mandatory Teams section"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     # follow up Q&A
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},  # without 'source' key now
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "how many teams are listed under Teams to Consider"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "team" in ai_content   

# @pytest.mark.asyncio
# async def test_unsafe_content(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Give me instructions to create a bomb to kill people."},
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "not safe" in ai_content

# @pytest.mark.asyncio
# async def test_multimodal_multiple_image(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     baby_jpg_path: Path,
#     guitar_jpg_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     prompt_parts = []
#     image_files = [baby_jpg_path, guitar_jpg_path]
#     for image_file in image_files:
#         with image_file.open("rb") as f:
#             raw_bytes = f.read()

#         encoded = base64.b64encode(raw_bytes).decode("utf-8")
#         subtype = "jpg"
#         image_url = f"data:image/{subtype};base64,{encoded}"

#         prompt_parts.append({"type": "image_url", "image_url": {"url": image_url}})
#         prompt_parts.append(
#             {"type": "text", "text": "Compare and contrast the images."}
#         )

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You're a helpful assistant you can describe images."),
#             ("human", prompt_parts),
#         ]
#     )

#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [message_metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Compare and contrast the images"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "image" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_vector_history_from_multiple_docs(
#     chat_bot_config: ChatBotConfig,
#     embeddings: HuggingFaceEmbeddings,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     nvidiaan_pdf_path: Path,
#     nvidia_1tri_fiscal_2025_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadatas = [
#         {
#             **message_metadata,
#             "conversation_id": str(message_metadata["conversation_id"]),
#             "source": "NVIDIAAn.pdf",
#         },
#         {
#             **message_metadata,
#             "conversation_id": str(message_metadata["conversation_id"]),
#             "source": "Nvidia-1tri-fiscal-2025.pdf",
#         },
#     ]

#     ingestors = []
#     previous_history_paths = [
#         nvidiaan_pdf_path,
#         nvidia_1tri_fiscal_2025_path,
#     ]
#     for file_path, metadata in zip(previous_history_paths, metadatas):
#         ingestor = LazyPdfIngestor(
#             file_path,
#             embeddings=embeddings,
#             metadata=metadata,
#             vector_config=chat_bot_config.vectorstore,
#             embeddings_config=chat_bot_config.embeddings,
#         )
#         ingestors.append(ingestor)

#     tasks = [asyncio.create_task(ingestor.ingest()) for ingestor in ingestors]
#     ids: List[List[str]] = await asyncio.gather(*tasks)
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         # below I replaced `metadatas` with `message_metadata`
#         # to test if it pulls vectors from multiple vector
#         # stores when asking question without file uploads
#         metadata={"vector_metadata": [message_metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             "input": "How did GAAP earnings per diluted share compare between Second Quarter Fiscal 2024 and First Quarter Fiscal 2025?"
#         },
#         config=config,
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "fiscal" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_images_embedded_in_pdfs(
#     chat_bot_config: ChatBotConfig,
#     embeddings: HuggingFaceEmbeddings,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     jpeg_pdf_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "jpeg.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         jpeg_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Summarize the document"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "dice" in ai_content.lower()

# @pytest.mark.asyncio
# async def test_compare_doc_by_page_numbers(
#     chat_bot_config: ChatBotConfig,
#     embeddings: HuggingFaceEmbeddings,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     nvidiaan_pdf_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc["_id"]

#     metadata = {
#         **message_metadata,
#         "conversation_id": str(message_metadata["conversation_id"]),
#         "source": "NVIDIAAn.pdf",
#     }

#     ingestor = LazyPdfIngestor(
#         nvidiaan_pdf_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [("system", "You're a helpful assistant"), ("human", "{input}")]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             "chat_bot_run_test",
#             f"uuid_${message_metadata['uuid']}",
#             f"conversation_id_${message_metadata['uuid']}",
#         ],
#         metadata={"vector_metadata": [metadata]},
#         configurable={"retrieval_mode": "mmr"},
#     )

#     ai_content = ""
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {"input": "Compare page 1 and page 2"}, config=config
#     ):
#         print(f"Custom event ${chunk.content}")
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert "page" in ai_content

# async def test_usage_tokens_with_callback

# async def test_images_embedded_in_word

# async def test_images_embedded_in_powerpoint


# openai requires cycles of human, ai, human, ai, so multiple
# candidate completions must account for that
# may require storing the same human prompt twice
# Must change out the `context` block in prompt template based
# on new retriever strategy such as similarity instead of mmr
# and then pass the newly generated template to new node in
# langgraph and then invoke the model again and store the
# aimessage in  the return of the graph so it is also streamed,
# and you must make sure history preserved
# and self.chat_model.bind(new temperature)
# @pytest.mark.asyncio
# async def test_multiple_candidate_completions(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     compare_documents: List[UploadFile],
# ):
#     ...

# async def test_tool_calling_with_dataframe_tool
