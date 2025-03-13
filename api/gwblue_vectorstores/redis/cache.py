# class RedisVectorRetriever:
#     def __init__(self, vector_store: RedisVectorStore):
#         self.vector_store = vector_store
#         self.config: RedisConfig = vector_store.config
#         self.embeddings: Embeddings = vector_store.embeddings

#     @property
#     def content_field_name(self) -> str:
#         """Name for document content"""
#         return self.config.content_field

#     @property
#     def embedding_vector_field_name(self) -> str:
#         """Name for embedding vectors"""
#         return self.config.embedding_field

#     @property
#     def embedding_dimensions(self) -> int:
#         """Embedding Dimension count"""
#         return self.config.embedding_dimensions

#     async def aadd(self, documents: Iterator[Document]) -> List[str]:
#         """Add documents to the vector store asynchronously, expecting metadata per document"""
#         return await self.vector_store.aadd_documents_with_ttl(documents, _VECTOR_TTL_30_DAYS, self.embeddings.max_batch_requests)

#     async def asimilarity_search(
#         self,
#         query: str,
#         filter: FilterExpression = None
#     ) -> List[Document]:
#         """Use Async Cosine Similarity Search to get immediate results"""
#         return await self.vector_store.asimilarity_search(query, filter=filter)

#     async def adelete(
#         self,
#         query: str = '',
#         filter: FilterExpression = None
#     ) -> bool:
#         documents = await self.vector_store.asimilarity_search(query=query, filter=filter)
#         document_ids = [doc.metadata['id'] for doc in documents]
#         if document_ids:
#             result = self.vector_store.adelete(ids=document_ids)
#             if result:
#                 return True
#         return False

#     async def inspect(
#         self,
#         query: str,
#         k: int = 4,
#         filter: FilterExpression = None,
#     ) -> str:
#         from tabulate2 import tabulate
#         query_vector = await self.embeddings.aembed_query(query)
#         results = await self.vector_store.asimilarity_search_by_vector(
#             embedding=query_vector,
#             k=k,
#             filter=filter,
#         )
#         table_data = []
#         for result in results:
#             table_data.append([result.page_content, result.metadata])
#         headers = ['Document', 'Metadata']

#         output = f"""
#             Document Size:
#             {len(results)}
#             Data:
#             {tabulate(table_data, headers=headers, tablefmt='grid')}
#             Schema:
#             {self}
#         """

#         return output

#     def __str__(self):
#         """Index Schema"""
#         return str({
#             'index': {
#                 'name': self.config.index_name,
#                 'prefix': self.config.key_prefix,
#                 'storage_type': self.config.storage_type,
#             },
#             'fields': [
#                 {
#                     'name': self.content_field_name,
#                     'type': 'text'
#                 },
#                 {
#                     'name': self.embedding_vector_field_name,
#                     'type': 'vector',
#                     'attrs': {
#                         'dims': self.config.embedding_dimensions,
#                         'distance_metric': self.config.distance_metric,
#                         'algorithm': self.config.indexing_algorithm,
#                         'datatype': self.config.vector_datatype,
#                     },
#                 },
#                 # *self._schema,
#             ],
#         })
