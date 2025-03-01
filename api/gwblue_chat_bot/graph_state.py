from typing import (
    Annotated, 
    TypedDict,
    List,
    Literal,
    Dict,
    Any,
)
from bson import ObjectId
from langgraph.graph.message import add_messages

class Metadata(TypedDict):
    uuid: str
    conversation_id: str # TODO: change to session_id to make generic
    source: str

class State(TypedDict):
    route: str
    metadata: List[Metadata]
    messages: Annotated[list, add_messages]
    retrieval_mode: Literal['similarity', 'mmr', 'similarity_score_threshold']