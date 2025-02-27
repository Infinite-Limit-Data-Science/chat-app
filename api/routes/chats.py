from typing import Callable, AsyncGenerator, Dict, Any
from langchain_core.prompts.chat import ChatPromptTemplate
from ..gwblue_chat_bot.chat_bot import ChatBot
from ..gwblue_chat_bot.chat_bot_config import ChatBotConfig

async def chat(
    *,
    system: str,
    input: str,
    config: ChatBotConfig,
    metadata: Dict[str, Any],
) -> Callable[[], AsyncGenerator[str, None]]:
    chat_prompt = ChatPromptTemplate.from_messsages([
        ('system', system),
        ('human', '{input}')
    ])
    chat_bot = ChatBot(config=config)

    chain = chat_prompt | chat_bot 

    # I NEED TO DO A RUNNABLE_PARALLEL AND PASS retrieval_mode mmr for one and retrieval_mode similarity_search_with_threshold for the other !!!!
    # THIS SHOULD ONLY BE DONE WITH DOCUMENT UPLOADS. IF NO DOCUMENT UPLOAD, THEN ONLY SEND BACK A SINGLE RESPONSE.
    return await chain.astream({'input': input}, metadata=metadata, retrieval_mode='mmr')


    # Chat bot should be invoked with dictionary:
    # chain = {
    #     "model": ...,
    #     "guardrails": ...,
    # } | chat_bot
    # chain.astream()
    # the dictionary will be promoted to a RunnableLambda piped to chat_bot
    # and it returns a RunnableSequence
    # for pandas expression tool, have an option max_results which prompts
    # the model to give multiple responses, so we can aggregate responses
    # and send back to user. Return responses in list similar to how 
    # TavilySearchResults tool works. 
    # Also the Tool should have an invoke method and astream method.


    # chat_bot = ChatBot()
    # builder = ChatBotBuilder(chat_bot)
    # builder.build_vector_part(
    #     vector_store,
    #     retrievers, 
    #     embedding_models,
    #     {
    #         **data,
    #         'conversation_id': str(data['conversation_id'])
    #     },
    # )
    # builder.build_llm_part(models)
    # builder.build_guardrails_part(guardrails)
    # builder.build_prompt_part(user_prompt_template)
    # builder.build_message_part({
    #     'connection_string': database_instance.connection_string,
    #     'database_name': database_instance.name,
    #     'collection_name': database_instance.message_history_collection,
    #     'session_id_key': database_instance.message_history_key,
    # },
    # {
    #     'session_id': data['conversation_id'],
    # })