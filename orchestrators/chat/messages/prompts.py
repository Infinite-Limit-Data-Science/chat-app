from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

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

REPHRASED_TEMPLATE="""
    Given the conversation history below, generate a search query that is more explicit and detailed.

    Conversation History:
    {chat_history}

    User Query:
    {input}

    Rephrased Query:
    """
@register('rephrased_runnable')
def rephrased_runnable():
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(REPHRASED_TEMPLATE)
    ])