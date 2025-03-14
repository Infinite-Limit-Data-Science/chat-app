from typing import Callable, AsyncGenerator, Dict, Any, Union, List
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from ..gwblue_chat_bot.chat_bot import ChatBot
from ..gwblue_chat_bot.chat_bot_config import ChatBotConfig


async def chat(
    *,
    system: str,
    input_dict: Dict[str, Any],
    config: ChatBotConfig,
    vector_metadata: List[Dict[str, Any]],
) -> Callable[[], AsyncGenerator[str, None]]:
    """
    Return a streaming generator of LLM response chunks.
    The `input` can be:
      - A string (no images)
      - A list (possibly with image dicts, e.g. [{'image_url': {...}}, 'some text'])
    """
    message_metadata = {k: vector_metadata[0][k] for k in ("uuid", "conversation_id")}

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}")]
    )
    if input_dict.get("prompt", False):
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", input_dict["prompt"])]
        )

    input = input_dict["input"]

    chat_bot = ChatBot(config=config)
    chain = chat_prompt | chat_bot

    run_config = RunnableConfig(
        tags=[
            "chat_bot_run_test",
            f"uuid_{message_metadata['uuid']}",
            f"conversation_id_{message_metadata['conversation_id']}",
        ],
        metadata={"vector_metadata": vector_metadata},
        configurable={"retrieval_mode": "mmr"},
    )

    async def stream_response():
        async for chunk in chain.astream({"input": input}, config=run_config):
            data_str = chunk.model_dump_json(indent=2)
            yield data_str

    return stream_response
