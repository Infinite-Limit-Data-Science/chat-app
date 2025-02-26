import os
import json
import pytest
from langchain_core.prompts.chat import ChatPromptTemplate
from ..chat_bot_config import (
    LLMConfig,
    EmbeddingsConfig,
    VectorStoreConfig,
    UserConfig,
    ChatBotConfig
) 
from ..chat_bot import ChatBot

@pytest.fixture
def chat_bot_config() -> ChatBotConfig:
    models = json.loads(os.environ['MODELS'])
    chat_model = models[0]
    llm_config = LLMConfig(**{
        'name': chat_model['name'],
        'endpoint': chat_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'parameters': {},
        'server': 'tgi',
    })

    embeddings = json.loads(os.environ['EMBEDDING_MODELS'])
    embeddings_model = embeddings[0]
    embeddings_config = EmbeddingsConfig(**{
        'name': embeddings_model['name'],
        'endpoint': embeddings_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'server': 'tei',        
    })

    guardrails_model = models[-1]
    guardrails_config = LLMConfig(**{
        'name': guardrails_model['name'],
        'endpoint': guardrails_model['endpoints'][0]['url'],
        'token': os.environ['TEST_AUTH_TOKEN'],
        'parameters': {},
        'server': 'tgi',
    })

    vectorstore_config = VectorStoreConfig(**{
        'name': 'redis',
        'url': os.environ['REDIS_URL'],
        'collection_name': 'messages',
        'session_id_key': 'conversation_id',
    })

    user_config = UserConfig(**{
        'uuid': '1',
        'session_id': 1,
        'session_id_str': '1',
        'vectorstore_docs': []
    })

    return ChatBotConfig(
        llm=llm_config,
        embeddings=embeddings_config,
        guardrails=guardrails_config,
        vectorstore=vectorstore_config,
        user_config=user_config,
    )

def test_guardrails_node(chat_bot_config: ChatBotConfig):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You're a helpful assistant"),
            ('human', 'Tell me about the movie {input}')
        ]
    )
    chat_bot = ChatBot(config=chat_bot_config)
    chain = chat_prompt | chat_bot
    conversation = chain.invoke({'input': 'Memento'})
    assert conversation['messages'][-1].content.strip('\n') == 'safe'

    # for event in graph.stream(initial_state):
    #     for value in event.values():
    #         assert value['messages'].strip('\n') == 'safe'

# initial_state = {
#     "conversation_id": "my-convo-123",
#     "datasource": "multidoc_compare",  # we already know the source
#     "vectorstore_docs": [],
#     "messages": [
#         {
#           "role": "user",
#           "content": "'What is Generative AI?'"
#         }
#     ],
# }