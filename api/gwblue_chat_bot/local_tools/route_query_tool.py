from typing import Literal
from pydantic import BaseModel, Field

class RouteQueryTool(BaseModel):
    """Route a user query to most relevant datasource"""
    datasource: Literal[
        'vectorstore', 
        'pretrained',
    ] = Field(..., description='Given a user question choose to route it to datasource')