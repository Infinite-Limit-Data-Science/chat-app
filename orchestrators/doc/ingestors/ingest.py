import os
from typing import List
from orchestrators.doc.ingestors.document_ingestor import DocumentIngestor
from orchestrators.doc.vector_stores.abstract_vector_store import AbstractVectorStore
from orchestrators.doc.embedding_models.embedding import BaseEmbedding
from orchestrators.doc.embedding_models.model_proxy import ModelProxy
from orchestrators.doc.ingestors.factories import FACTORIES as I_FACTORIES
from orchestrators.doc.vector_stores.factories import FACTORIES as V_FACTORIES
from orchestrators.doc.runnable_extensions.wrapper_runnable_config import WrapperRunnableConfig

async def ingest(
        file: str, 
        store: str, 
        embedding_models: List[BaseEmbedding], 
        wrapper_runnable_config: WrapperRunnableConfig) -> List[str]:
    vector_store: AbstractVectorStore = V_FACTORIES[store](ModelProxy(embedding_models), wrapper_runnable_config)
    ingestor: DocumentIngestor = I_FACTORIES[os.path.splitext(file)[1][1:]](file, vector_store, wrapper_runnable_config)
    return await ingestor.ingest()