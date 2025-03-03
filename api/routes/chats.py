from typing import Callable, AsyncGenerator, Dict, Any, Union, List
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from ..gwblue_chat_bot.chat_bot import ChatBot
from ..gwblue_chat_bot.chat_bot_config import ChatBotConfig

async def chat(
    *,
    system: str,
    input: List[Any],
    config: ChatBotConfig,
    vector_metadata: List[Dict[str, Any]],
) -> Callable[[], AsyncGenerator[str, None]]:
    """
    Return a streaming generator of LLM response chunks.
    The `input` can be:
      - A string (no images)
      - A list (possibly with image dicts, e.g. [{'image_url': {...}}, 'some text'])
    """
    message_metadata = {k: vector_metadata[0][k] for k in ('uuid', 'conversation_id')}

    if isinstance(input, list) and len(input) == 1:
        chat_prompt = ChatPromptTemplate.from_messages([
            ('system', system),
            ('human', '{input}')
        ])
        chain_input = {'input': input}

    else:
        chat_prompt = ChatPromptTemplate.from_messages([
            ('system', system),
            ('human', [
                {'image_url': {'url': "{image_url}"}},
                '{input}'
            ])
        ])
        chain_input = {'input': input[0], 'image_url': input[1]['image_url']['url']}

    chat_bot = ChatBot(config=config)
    chain = chat_prompt | chat_bot

    run_config = RunnableConfig(
        tags=[
            "chat_bot_run_test",
            f"uuid_{message_metadata['uuid']}",
            f"conversation_id_{message_metadata['conversation_id']}"
        ],
        metadata={
            'vector_metadata': vector_metadata
        },
        configurable={
            'retrieval_mode': 'mmr'
        }
    )

    async def stream_response():
        async for chunk in chain.astream(chain_input, config=run_config):
            data_str = chunk.model_dump_json(indent=2)
            yield data_str

    return stream_response


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