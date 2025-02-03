import sys
import json
from .langchain_dataframe_expression_tool import QueryDataFrame

function_call = sys.argv[1]

try:
    json.load(function_call)
except json.JSONDecodeError as e:
    print(f'JSONDecodeError: {e}')
else:
    query_data_frame = QueryDataFrame(query=function_call)
    print(query_data_frame.to_openai_tool())