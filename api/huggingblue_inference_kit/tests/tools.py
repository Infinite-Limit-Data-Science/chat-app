import ast
import pandas as pd
from typing import Type, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun

class PandasExpressionInput(BaseModel):
    df_expr: str = Field(
        title='Pandas DataFrame Expression',
        description='A Python expression referencing a DataFrame named `df`.'
    )

    @field_validator('df_expr')
    @classmethod
    def check_syntax(cls, value):
        try:
            ast.parse(value, mode='eval')
        except SyntaxError as e:
            raise ValueError(f'Invalid Python expression: {e}')
        return value

class PandasExpressionTool(BaseTool):
    """
    Tool that evaluates a Pandas DataFrame expression referencing `df`.
    The actual DataFrame is provided at runtime (e.g., self.df or from a run_manager).
    """
    name: str = 'pandas_expression_tool'
    description: str = (
        "Use this tool to evaluate a Python expression referencing `df` "
        "as a pandas DataFrame. For example: df['A'].mean()."
    )
    df: Optional[pd.DataFrame] = Field(
        default=None,
        description='The actual user-provided DataFrame to evaluate expressions on'
    )

    args_schema: Type[BaseModel] = PandasExpressionInput

    def _run(
        self,
        df_expr: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Evaluate `df_expr` in a restricted environment with `df` referencing
        our actual user-provided DataFrame.
        """
        # If you prefer storing the DataFrame in the run_manager or callback,
        # you could do: user_df = run_manager.get("df")
        try:
            result = eval(df_expr, {'__builtins__': None}, {'df': self.df})
        except Exception as e:
            raise ValueError(f'Error evaluating expression {df_expr}: {e}')

        return str(result)

    async def _arun(
        self,
        df_expr: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Asynchronous version of the tool method.
        """
        ...