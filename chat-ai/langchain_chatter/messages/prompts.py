from typing import Optional
from langchain_core.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    PromptTemplate,
)

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

MY_CHAT_TEMPLATE = """
You are an assistant for question-answering tasks.
"""
@register('qa_template')
def qa_template(preprompt: Optional[str] = None) -> ChatPromptTemplate:
    """Returns runnable with a dynamic preprompt."""
    preprompt_text = preprompt or MY_CHAT_TEMPLATE
    
    system_template = f"""
        {preprompt_text}
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        {{context}}
    """
    
    return ChatPromptTemplate.from_messages([
        ('system', system_template),
        MessagesPlaceholder('chat_history'),
        ('human', "{input}"),
    ])

@register('chat_preprompt_template')
def chat_preprompt_template(preprompt: Optional[str] = None) -> ChatPromptTemplate:
    """Returns runnable"""
    system_template = preprompt or MY_CHAT_TEMPLATE
    return ChatPromptTemplate.from_messages([
        ("system",system_template),
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

CONTENT_SAFETY_TEMPLATE = """
<|begin_of_text|><|start_header_id|>{agent_type}<|end_header_id|>

Task: Check if there is unsafe content in the following message according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes. 
S2: Non-Violent Crimes. 
S3: Sex Crimes. 
S4: Child Exploitation. 
S5: Defamation. 
S6: Specialized Advice. 
S7: Privacy. 
S8: Intellectual Property. 
S9: Indiscriminate Weapons. 
S10: Hate. 
S11: Self-Harm. 
S12: Sexual Content. 
S13: Elections. 
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {input}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST user message in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
@register('guardrails_template')
def guardrails_template():
    """Returns runnable"""
    return PromptTemplate(
        input_variables=['input', 'agent_type'],
        template=CONTENT_SAFETY_TEMPLATE
    )