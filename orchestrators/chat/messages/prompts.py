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

CHAT_HISTORY_TEMPLATE = """
    You are a helpful assistant. Answer all the questions to the best of your ability
"""
@register('chat_history_template')
def chat_history_template():
        """Returns runnable"""
        return ChatPromptTemplate.from_messages([
            ("system",CHAT_HISTORY_TEMPLATE),
            MessagesPlaceholder('chat_history'),
            ('human', "{input}"),
        ])

SUMMARIZATION_TEMPLATE = """Provide a concise summary in a few words, and prefix the summary with an emoji that reflects the tone of the summary:

Text:
{input}

Summary:"""
@register('summarization_template')
def summarization_template():
     """Returns runnable"""
     return ChatPromptTemplate.from_template(SUMMARIZATION_TEMPLATE)

"""
Note the actual "context" of this ChatPromptTemplate are dynamically generated
based on the number of documents to compare
"""
HISTORY_COMPARE_TEMPLATE = """
You are tasked with answering questions based on the context provided below.
Please compare and analyze the information from different documents.

{context}
"""
@register('history_compare_template')
def history_compare_template():
    """Returns runnable"""
    return ChatPromptTemplate.from_messages([
        ("system",HISTORY_COMPARE_TEMPLATE),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])