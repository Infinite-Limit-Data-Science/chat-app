    # TODO: chat_model, chat_bot (which is an abstraction over chat_model)
    # dataframe expression tool   
    # document loader with metadata and smart vector retriever
    # langgraph

def test_chat_model_type(chat_model: HuggingFaceChatModel):
    ...

def test_identifying_params(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke_with_prompt_template(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke_with_output_parser(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke_with_few_shot_prompt(
        chat_model: HuggingFaceChatModel, 
        vectorstore: Iterator[RedisVectorStore],
        sample_population: List[str]):
    ...

def test_chat_model_invoke_with_callbacks(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke_with_callbacks(chat_model: HuggingFaceChatModel):
    ...

def test_chat_model_invoke_with_run_information(spy_chat_model: SpyHuggingFaceChatModel):
    """
    This allows us to store uuids as metadata
    and then we can use langsmith to check
    performance bottlenecks and usage for specific 
    uuids (users)
    """
    ...

def test_chat_model_invoke_with_token_usage_in_response(chat_model: HuggingFaceChatModel):
    ...




def store_state_in_case_of_fatal_error():
    """
    Since a RunnableSequence is a kind of RunnableSerializable
    it has a to_json() method to serialize the state of the composite
    parts (the Runnables)
    """
