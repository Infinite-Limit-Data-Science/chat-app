import logging
import os
import json
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, BinaryIO, Protocol, Annotated
from typing_extensions import Doc
from redisvl.query.filter import FilterExpression
from functools import partial
import asyncio

from langchain_core.vectorstores import VectorStoreRetriever

from .vector_stores import ( 
    AbstractVectorStore, 
    create_filter_expression,
)
from .embedding_models import BaseEmbedding, ModelProxy
from .ingestors import FACTORIES as I_FACTORIES
from .vector_stores.factories import STORE_FACTORIES, RETRIEVER_FACTORIES

class FileLike(Protocol):
    @property
    def filename(self) -> Annotated[str, Doc('Name of binary object')]:
        ...
    
    @property
    def file(self) -> Annotated[BinaryIO, Doc('Binary object')]:
        ...

def generate_path(fields: Dict[str, str], filename: str) -> Path:
    path_components = [f"{key}/{value}" for key, value in fields.items()]
    return Path('files').joinpath(*path_components, filename)

def generate_retrievers(
    store: str,
    vector_store_proxy: AbstractVectorStore, 
    filters: List[FilterExpression], 
    metadatas: List[dict]
) -> List[VectorStoreRetriever]:
    retrievers = []
    for filter, metadata in zip(filters, metadatas):
        retrievers.append(RETRIEVER_FACTORIES[store](
            filter=filter,
            vector_store_proxy=vector_store_proxy,
            metadata=metadata,
        ))
    
    return retrievers

async def ingest(
    store: str,
    files: List[FileLike], 
    embedding_models: List[BaseEmbedding], 
    input_data: List[dict]
) -> Tuple[List[VectorStoreRetriever], List[str]]:
    if not (vector_store_schema_str := os.getenv('VECTOR_STORE_SCHEMA')):
        raise ValueError('Expected `VECTOR_STORE_SCHEMA` to be defined')
    
    vector_store_schema = json.loads(vector_store_schema_str)
    embeddings = ModelProxy(embedding_models).get()
    vector_store_proxy: AbstractVectorStore = STORE_FACTORIES[store](vector_store_schema, embeddings)

    filenames = []
    paths = []
    filters = []
    ingestors = []
    metadatas = []

    try:
        for file in files:
            path = generate_path(input_data, file.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as f:
                shutil.copyfileobj(file.file, f)
            paths.append(path)
            filenames.append(file.filename)

            metadata = { **input_data, 'source': file.filename }
            metadatas.append(metadata)
            filters.append(create_filter_expression(
                vector_store_schema, metadata))
            ingestor = partial(
                I_FACTORIES[os.path.splitext(file.filename)[1][1:]], 
                path, vector_store_proxy, metadata
            )
            ingestors.append(ingestor)

        tasks = [asyncio.create_task(ingestor().ingest()) for ingestor in ingestors]
        ids: List[List[str]] = await asyncio.gather(*tasks)

        if not len(ids) == len(files):
            raise AssertionError(f'Expected to ingest {len(files)} files with {files}')

        return generate_retrievers(store, vector_store_proxy, filters, metadatas), filenames
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError as e:
                logging.warning(f'Error deleting file {path}: {e}')