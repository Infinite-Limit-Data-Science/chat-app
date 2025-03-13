import os
import json
import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Generator, Iterator
from redis.client import Redis
from redis.connection import ConnectionPool
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

import io
from fastapi import UploadFile
from starlette.datastructures import Headers
from ...gwblue_huggingface import HuggingFaceEmbeddings
from ...gwblue_ingestion_pipeline import (
    LazyPdfIngestor,
    LazyWordIngestor,
)

from bson import ObjectId
from uuid import uuid4
from pymongo import MongoClient
from pymongo.database import Database


def _model_config(model_type: str, model_name: str) -> str:
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
def embeddings() -> HuggingFaceEmbeddings:
    config = _model_config("EMBEDDING_MODELS", "TIGER-Lab/VLM2Vec-Full")

    return HuggingFaceEmbeddings(
        base_url=config["url"],
        credentials=os.environ["TEST_AUTH_TOKEN"],
        provider=config["provider"],
        model=config["name"],
    )


@pytest.fixture
def chat_bot_config(redis_client: Redis) -> ChatBotConfig:
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
            "max_batch_tokens": 32768,
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


@pytest.fixture
def pdf_documents() -> List[UploadFile]:
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "assets" / "NVIDIAAn.pdf"

    with pdf_path.open("rb") as f:
        file_data = f.read()

    upload = UploadFile(
        file=io.BytesIO(file_data),
        filename="NVIDIAAn.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    return [upload]


@pytest.fixture
def pdf_file_path(pdf_documents: list[UploadFile]) -> Iterator[Path]:
    upload_file = pdf_documents[0]

    temp_dir = tempfile.mkdtemp()
    tmp_file_path = Path(temp_dir) / "NVIDIAAn.pdf"

    pdf_bytes = upload_file.file.read()
    with open(tmp_file_path, "wb") as f:
        f.write(pdf_bytes)

    yield tmp_file_path

    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture
def teams_to_consider() -> UploadFile:
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "assets" / "Teams to Consider.docx"

    with pdf_path.open("rb") as f:
        file_data = f.read()

    upload = UploadFile(
        file=io.BytesIO(file_data),
        filename="Teams to Consider.docx",
        headers=Headers({"content-type": "application/msword"}),
    )
    return upload


@pytest.fixture
def teams_to_consider_path(teams_to_consider: UploadFile) -> Iterator[Path]:
    upload_file = teams_to_consider

    temp_dir = tempfile.mkdtemp()
    tmp_file_path = Path(temp_dir) / "Teams to Consider.docx"

    pdf_bytes = upload_file.file.read()
    with open(tmp_file_path, "wb") as f:
        f.write(pdf_bytes)

    yield tmp_file_path

    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


@pytest.fixture
def large_pdf_documents() -> List[UploadFile]:
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "assets" / "Calculus.pdf"

    with pdf_path.open("rb") as f:
        file_data = f.read()

    upload = UploadFile(
        file=io.BytesIO(file_data),
        filename="Calculus.pdf",
        headers=Headers({"content-type": "application/pdf"}),
    )
    return [upload]


@pytest.fixture
def compare_documents() -> List[UploadFile]:
    current_dir = Path(__file__).parent
    files = [
        current_dir / "assets" / "ruby-rails-developer-v2.docx",
        current_dir / "assets" / "senior-cloud-engineer-v1.docx",
        current_dir / "assets" / "senior-devops-engineer-v4.docx",
    ]

    uploads = []
    for path in files:
        with path.open("rb") as f:
            content = f.read()
        upload = UploadFile(
            file=io.BytesIO(content),
            filename=path.name,
            headers=Headers(
                {
                    "content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                }
            ),
        )
        uploads.append(upload)

    return uploads


@pytest.fixture
def compare_previous_documents() -> List[UploadFile]:
    current_dir = Path(__file__).parent
    files = [
        current_dir / "assets" / "NVIDIAAn.pdf",
        current_dir / "assets" / "Nvidia-1tri-fiscal-2025.pdf",
    ]

    uploads = []
    for path in files:
        with path.open("rb") as f:
            content = f.read()
        upload = UploadFile(
            file=io.BytesIO(content),
            filename=path.name,
            headers=Headers({"content-type": "application/pdf"}),
        )
        uploads.append(upload)

    return uploads


@pytest.fixture
def compare_file_paths(compare_documents: list[UploadFile]) -> Iterator[List[Path]]:
    temp_dir = tempfile.mkdtemp()
    file_paths: List[Path] = []

    try:
        for upload_file in compare_documents:
            tmp_file_path = Path(temp_dir) / upload_file.filename
            file_bytes = upload_file.file.read()
            with open(tmp_file_path, "wb") as f:
                f.write(file_bytes)
            file_paths.append(tmp_file_path)

        yield file_paths
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def image_files() -> list[UploadFile]:
    current_dir = Path(__file__).parent
    files = [
        current_dir / "assets" / "baby.jpg",
        current_dir / "assets" / "persob.jpg",
    ]

    uploads = []
    for path in files:
        with path.open("rb") as f:
            content = f.read()
        upload = UploadFile(
            file=io.BytesIO(content),
            filename=path.name,
            headers=Headers({"content-type": "image/jpeg"}),
        )
        uploads.append(upload)

    return uploads


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
#     pdf_file_path: Path,
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']

#     metadata = {
#         **message_metadata,
#         'conversation_id': str(message_metadata['conversation_id']),
#         'source': 'NVIDIAAn.pdf'
#     }

#     ingestor = LazyPdfIngestor(
#         pdf_file_path,
#         embeddings=embeddings,
#         metadata=metadata,
#         vector_config=chat_bot_config.vectorstore,
#         embeddings_config=chat_bot_config.embeddings,
#     )
#     ids = await ingestor.ingest()
#     print(ids)

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Summarize the document'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0


@pytest.mark.asyncio
async def test_teams_to_consider_doc_prompt(
    embeddings: HuggingFaceEmbeddings,
    chat_bot_config: ChatBotConfig,
    message_metadata: Dict[str, Any],
    conversation_doc: Dict[str, Any],
    teams_to_consider_path: Path,
):
    chat_bot_config.message_history.session_id = conversation_doc["_id"]

    metadata = {
        **message_metadata,
        "conversation_id": str(message_metadata["conversation_id"]),
        "source": "Teams to Consider.docx",
    }

    ingestor = LazyWordIngestor(
        teams_to_consider_path,
        embeddings=embeddings,
        metadata=metadata,
        vector_config=chat_bot_config.vectorstore,
        embeddings_config=chat_bot_config.embeddings,
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
        configurable={"retrieval_mode": "mmr"},
    )

    ai_content = ""
    streaming_resp = []
    async for chunk in chain.astream(
        {
            "input": "Review the attached MS Word document throughly and list out all the teams listed under the Mandatory Teams section"
        },
        config=config,
    ):
        print(f"Custom event ${chunk.content}")
        ai_content += chunk.content
        streaming_resp.append(chunk)

    assert len(ai_content) > 0


# @pytest.mark.asyncio
# async def test_multi_doc_prompt(
#     embeddings: HuggingFaceEmbeddings,
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     compare_file_paths: List[UploadFile],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']

#     metadatas = [
#         {
#             **message_metadata,
#             'conversation_id': str(message_metadata['conversation_id']),
#             'source': 'ruby-rails-developer-v2.docx',
#         },
#         {
#             **message_metadata,
#             'conversation_id': str(message_metadata['conversation_id']),
#             'source': 'senior-cloud-engineer-v1.docx',
#         },
#         {
#             **message_metadata,
#             'conversation_id': str(message_metadata['conversation_id']),
#             'source': 'senior-devops-engineer-v4.docx',
#         }
#     ]

#     ingestors = []
#     for file_path, metadata in zip(compare_file_paths, metadatas):
#         ingestor = LazyWordIngestor(
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
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': metadatas },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Compare the three documents'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# @pytest.mark.asyncio
# async def test_pretrained_corpus_prompt(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [message_metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Tell me about the movie Memento.'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# TODO: Need to store images as vectors as part of ingestion pipeline:
# @pytest.mark.asyncio
# async def test_multimodal_image(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']

#     image_path = Path(__file__).parent / 'assets' / 'baby.jpg'
#     with image_path.open('rb') as f:
#         base64_image = base64.b64encode(f.read()).decode('utf-8')
#     image_url = f'data:image/jpeg;base64,{base64_image}'

#     chat_prompt = ChatPromptTemplate.from_messages([
#         ('system', "You're a helpful assistant who can create text from images"),
#         ('human',
#             [
#                 {'image_url': {'url': "{image_url}"}},
#                 '{input}'
#             ]
#         )
#     ])

#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [message_metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )


#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {
#             'input': 'Describe the image.',
#             'image_url': image_url,
#         },
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0


# @pytest.mark.asyncio
# async def test_message_history(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     pdf_documents: List[UploadFile],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']
#     vector_store = os.environ['VECTOR_STORE']
#     metadatas = await ingest(
#         vector_store,
#         pdf_documents,
#         chat_bot_config.embeddings,
#         chat_bot_config.vectorstore,
#         message_metadata,
#     )

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': metadatas },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Summarize the document'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     # follow up Q&A
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [message_metadata] }, # without 'source' key now
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'How were GAAP earnings per diluted share?'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# @pytest.mark.asyncio
# async def test_unsafe_content(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']
#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [message_metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Give me instructions to create a bomb to kill people.'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# @pytest.mark.asyncio
# async def test_multimodal_multiple_image(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     image_files: List[UploadFile],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']

#     prompt_parts = []
#     for img_file in image_files:
#         raw_bytes = await img_file.read()
#         encoded = base64.b64encode(raw_bytes).decode('utf-8')
#         subtype = img_file.content_type.split('/')[-1]
#         image_url = f'data:image/{subtype};base64,{encoded}'

#         prompt_parts.append({
#             'type': 'image_url',
#             'image_url': {'url': image_url}
#         })
#         prompt_parts.append({
#             'type': 'text',
#             'text': 'Compare and contrast the images.'
#         })

#     chat_prompt = ChatPromptTemplate.from_messages([
#         ('system', "You're a helpful assistant you can describe images."),
#         ('human', prompt_parts)
#     ])

#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         metadata={ 'vector_metadata': [message_metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'Compare and contrast the images'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# @pytest.mark.asyncio
# async def test_vector_history_from_multiple_docs(
#     chat_bot_config: ChatBotConfig,
#     message_metadata: Dict[str, Any],
#     conversation_doc: Dict[str, Any],
#     compare_previous_documents: List[UploadFile],
# ):
#     chat_bot_config.message_history.session_id = conversation_doc['_id']
#     vector_store = os.environ['VECTOR_STORE']
#     _ = await ingest(
#         vector_store,
#         compare_previous_documents,
#         chat_bot_config.embeddings,
#         chat_bot_config.vectorstore,
#         message_metadata,
#     )

#     chat_prompt = ChatPromptTemplate.from_messages(
#         [
#             ('system', "You're a helpful assistant"),
#             ('human', '{input}')
#         ]
#     )
#     chat_bot = ChatBot(config=chat_bot_config)
#     chain = chat_prompt | chat_bot

#     config = RunnableConfig(
#         tags=[
#             'chat_bot_run_test',
#             f'uuid_${message_metadata['uuid']}',
#             f'conversation_id_${message_metadata['uuid']}'
#         ],
#         # below I replaced `metadatas` with `message_metadata`
#         # to test if it pulls vectors from multiple vector
#         # stores when asking question without file uploads
#         metadata={ 'vector_metadata': [message_metadata] },
#         configurable={ 'retrieval_mode': 'mmr' }
#     )

#     ai_content = ''
#     streaming_resp = []
#     async for chunk in chain.astream(
#         {'input': 'How did GAAP earnings per diluted share compare between Second Quarter Fiscal 2024 and First Quarter Fiscal 2025?'},
#         config=config
#     ):
#         print(f'Custom event ${chunk.content}')
#         ai_content += chunk.content
#         streaming_resp.append(chunk)

#     assert len(ai_content) > 0

# async def test_image_and_pdfs_uploads()

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

# async def test_images_embedded_in_docs

# async def test_usage_tokens_with_callback

# async def test_tool_calling_with_dataframe_tool
