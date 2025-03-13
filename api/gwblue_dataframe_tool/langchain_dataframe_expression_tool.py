from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool


class QueryDataFrame(BaseModel):
    """Query a Pandas dataframe using a valid Pandas query expression."""

    query: str = Field(..., description="A valid Pandas query expression.")

    def to_openai_tool(self) -> str:
        convert_to_openai_tool(self)
