import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from .prompt_examples import examples
from ..langchain_doc.vector_stores import STORE_FACTORIES

PREFIX_FUNCTIONS = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
"""

FUNCTIONS_WITH_DF = """
    This is the result of `print(df.head())`:
    {df_head}
"""

FUNCTIONS_WITH_MULTI_DF = """
    This is the result of `print(df.head())` for each dataframe:
    {dfs_head}
"""


def few_shot_chat_prompt(
    dfs: pd.DataFrame, prefix: str | None = None, suffix: str | None = None
) -> ChatPromptTemplate:
    # you need to check if data is already vectorized, otherwise you will vectorize the example data each time you run the tool
    # therefore this example data must contain metadata
    to_vectorize = [" ".join(example.values()) for example in examples]
    vectorstore = STORE_FACTORIES["redis"]()
    vectorstore.from_texts(to_vectorize, vectorstore.embeddings, metadatas=examples)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],
        example_selector=example_selector,
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )

    system = prefix or PREFIX_FUNCTIONS
    system += suffix or FUNCTIONS_WITH_MULTI_DF if len(dfs) > 1 else FUNCTIONS_WITH_DF

    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            few_shot_prompt,
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    print(chat_prompt_template.format_messages())

    return chat_prompt_template
