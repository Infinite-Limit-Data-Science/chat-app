import logging
import os
from typing import List
from langchain_core.vectorstores import VectorStoreRetriever
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from orchestrators.doc.vector_stores.abstract_vector_store import (
    AbstractVectorStore, 
    AbstractFlexiSchemaFields
)
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.embedding_models.model_proxy import ModelProxy
from orchestrators.doc.ingestors.factories import FACTORIES as I_FACTORIES
from orchestrators.doc.vector_stores.factories import FACTORIES as V_FACTORIES

async def ingest(
    file: str, 
    store: str, 
    embedding_models: List[BaseEmbedding], 
    metadata: List[AbstractFlexiSchemaFields]) -> List[VectorStoreRetriever]:
    embeddings = ModelProxy(embedding_models).get()
    vector_store: AbstractVectorStore = V_FACTORIES[store](embeddings, metadata)
    ingestor: DocumentIngestor = I_FACTORIES[os.path.splitext(file)[1][1:]](file, vector_store, metadata)
    embedded_ids = await ingestor.ingest()
    logging.warning(f'{file} embedded ids {embedded_ids}')

    return vector_store.retriever()