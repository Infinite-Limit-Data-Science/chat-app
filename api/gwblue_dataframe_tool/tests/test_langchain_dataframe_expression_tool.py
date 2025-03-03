import pytest
import pandas as pd
from typing import List
from ..langchain_dataframe_expression_tool import LangchainDataFrameExpressionTool, QueryDataFrame
from ...langchain_chat import FACTORIES as LLM_FACTORIES
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)

@pytest.fixture
def pandas_dataframe():
    data = {
        "First Name": ["John", "Jane", "Alex", "Emily", "Michael", "Sarah", "David", "Laura", "Chris", "Anna"],
        "Last Name": ["Doe", "Smith", "Johnson", "Brown", "Wilson", "Taylor", "Anderson", "Martinez", "Lee", "Garcia"],
        "Age": [34, 29, 45, 38, 50, 27, 41, 36, 33, 28],
        "Insurance Plan": [
            "HMO", "Blue Options PPO", "HMO", "EPO", "PPO",
            "HMO", "Blue Options PPO", "EPO", "HMO", "PPO"
        ],
        "Premiums": [350.50, 420.75, 300.00, 500.25, 610.40, 320.90, 450.30, 490.50, 310.80, 400.00],
        "Annual Physical Checkup": [True, False, True, False, True, True, False, True, False, True]
    }
    return pd.DataFrame(data)

@pytest.fixture
def chat_model() -> BaseChatModel:
    model_details = {"name":"meta-llama/Meta-Llama-3.1-70B-Instruct","endpoints":[{"url":"http://ip:8080/","type":"tgi"}],"classification":"text-generation","active": True}
    tgi = LLM_FACTORIES['tgi'](**model_details)
    return tgi.endpoint_object

@pytest.fixture
def chat_history() -> List[BaseMessage]:
    return [
        HumanMessage(content=""),
        AIMessage(content=""),
        HumanMessage(content=""),
        AIMessage(content=""),
    ]

@pytest.fixture
def function_call_schema() -> QueryDataFrame:
    return QueryDataFrame(query="what is 2 + 2")

def test_few_shot_prompt(function_call_schema: QueryDataFrame, chat_model: BaseChatModel, chat_history: List[BaseMessage]):
    formatted_tool = function_call_schema.to_openai_tool()
    print(formatted_tool)