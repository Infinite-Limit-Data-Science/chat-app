from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

registry = {}

def register(name):
    def decorator(func):
        registry[name] = func
        return func
    return decorator

BASE_TEMPLATE="""
    Given the conversation history below, generate a search query that is more explicit and detailed.

    Conversation History:
    {chat_history}

    User Query:
    {input}

    Search Query:
    """
@register('base_runnable')
def base_runnable():
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(BASE_TEMPLATE)
    ])

CONTEXTUALIZED_TEMPLATE = """
    Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.
"""
@register('contextualized_template')
def contextualized_template():
    """Returns runnable"""
    return ChatPromptTemplate.from_messages([
        ('system', CONTEXTUALIZED_TEMPLATE),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])

QA_TEMPLATE = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    {context}
"""
@register('qa_template')
def qa_template():
    """Returns runnable"""
    return ChatPromptTemplate.from_messages([
        ('system', QA_TEMPLATE),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])

LLM_TEMPLATE = """
    You are a helpful assistant. Answer all the question to the best of your ability
"""
@register('llm_template')
def llm_template():
        """Returns runnable"""
        return ChatPromptTemplate.from_messages([
            ("system",LLM_TEMPLATE),
            MessagesPlaceholder('chat_history'),
            ('human', "{input}"),
        ])