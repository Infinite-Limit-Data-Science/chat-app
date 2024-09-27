from typing import List, TypedDict, Dict, Any
from langchain_core.callbacks.base import Callbacks
from langchain_core.runnables.config import RunnableConfig

class WrapperRunnableConfig(TypedDict, total=False):
    runnable_config: RunnableConfig
    
    tags: List[str]

    metadata: Dict[str, Any]

    callbacks: Callbacks

    configurable: Dict[str, Any]