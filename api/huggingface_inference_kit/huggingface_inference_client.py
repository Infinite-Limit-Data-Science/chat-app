from typing import (
    Protocol, 
    Annotated, 
    Self, 
    Optional, 
    Dict, 
    Union,
    List, 
    Iterable,
    Literal, 
    overload,
    runtime_checkable
)
from typing_extensions import Doc
from urllib.parse import urlparse
import numpy as np
from pydantic import model_validator, field_validator, Field, ConfigDict
from huggingface_hub.inference._generated.types import (
    ChatCompletionOutput, 
    ChatCompletionStreamOutput,
    ChatCompletionInputGrammarType, 
    ChatCompletionInputStreamOptions,
    ChatCompletionInputToolChoiceClass, 
    ChatCompletionInputToolChoiceEnum,
    ChatCompletionInputTool
)
from huggingface_hub.inference._providers import get_provider_helper
from huggingface_hub.inference._common import _import_numpy, _bytes_to_dict
from .inference_schema import HuggingFaceTEIMixin

@runtime_checkable
class HuggingFaceInferenceLike(Protocol):
    def chat_completion(
        self,
        messages: List[Dict],
        *,
        stream: Optional[bool] = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Annotated[Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]], Doc('OpenAI-compatible API for chat-based text generation')]:
        """
        Args:
            messages (List of [`ChatCompletionInputMessage`]):
                Conversation history consisting of roles and content pairs.            
            stream (`bool`, *optional*):
                Enable realtime streaming of responses. Defaults to False.
                Note both the low-level InferenceClient and AsyncInferenceClient support streaming (stream=True), 
                but they serve different purposes in terms of execution models.
                Streaming controls how the response is delivered (token-by-token instead of waiting for the full response).
                In InferenceClient, it is synchronous - the function blocks execution until it receives a response.
                In AsyncInferenceClient, async execution allows multiple tasks to run concurrently without blocking.
                Uses Python's async and await, making it non-blocking.
                Useful when you need to make multiple API calls concurrently.
            frequency_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in
                the text so far, decreasing the model's likelihood to repeat the same line verbatim.
            logprobs (`bool`, *optional*):
                Whether to return log probabilities of the output tokens or not. If true, returns the log
                probabilities of each output token returned in the content of message.
            max_tokens (`int`, *optional*):
                Maximum number of tokens allowed in the response. 
                That is, the maximum number of tokens that can be generated in the chat completion.
                Defaults to 100.
            num_generations (`int`, *optional*):
                The number of completions to generate for each prompt.
                Specifies how many responses the model should generate for a single input prompt.
                If n=1 (default), the model returns one response.
                If n=3, the model returns three different completions for the same prompt.
                Each completion is independently generated, meaning they can be different depending on settings like temperature and top_p.
            presence_penalty (`float`, *optional*):
                Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
                text so far, increasing the model's likelihood to talk about new topics.
            response_format ([`ChatCompletionInputGrammarType`], *optional*):
                Starting from version 1.4.3, Hugging Face's Text Generation Inference (TGI) supports the response_format parameter
                This feature is compatible with models like Llama 3.1 70B.
                Constrain the model's output to adhere to a defined structure or pattern, enhancing the precision and reliability of the generated responses.
                Example:
                    response_format = ChatCompletionInputGrammarType(
                        type="regex",
                        value="(([A-Z][a-z]+) ){1,3}"  # Enforce capitalized words (1-3)
                    )
                    response = client.chat_completion(
                        messages=[{"role": "user", "content": "Give me a structured response"}],
                        response_format=response_format,
                    )
                    print(response.choices[0].message.content)
                    
                    class Animals(BaseModel):
                        location: str
                        activity: str
                        animals_seen: conint(ge=1, le=5)
                        animals: list[str]
                    
                    response_format = ChatCompletionInputGrammarType(
                        type="json",
                        value=Animals.schema()  # Convert Pydantic model to JSON Schema
                    )
                    response = client.chat_completion(
                        messages=[{"role": "user", "content": "Describe my pet sightings in JSON format"}],
                        response_format=response_format,
                    )
                    print(response.choices[0].message.content)
            seed (Optional[`int`], *optional*):
                The seed parameter in chat_completion is used to control the randomness of the model's response, 
                ensuring reproducible outputs when generating text.
                When you use an AI model like Llama 3.1 70B (or any other transformer-based model), 
                the text generation process involves sampling from a probability distribution of possible next tokens. 
                This means that if you provide the same prompt multiple times, you might get different responses.
                By setting a fixed seed, the model's sampling process is deterministic.
                This means you will get the same output every time for the same input.
                Without setting a seed, responses may vary slightly even for identical prompts.
                When text is generated, models use random number generators (RNGs) to sample words. 
                The seed value initializes this RNG, making the process predictable and repeatable.
                If seed is not set (None by default), the model will behave stochastically, 
                meaning different runs can produce different outputs even if all other parameters remain the same.
                This option defaults to None
            stop (`List[str]`, *optional*):
                The stop parameter is a list of up to four strings that instruct the model to stop generating tokens 
                once it encounters one of the specified stop sequences.
                When generating text, the model continues token-by-token.
                If a stop sequence is encountered in the response, the model stops generating immediately.
                This is useful for controlling output length.
                This option defaults to None
            stream_options ([`ChatCompletionInputStreamOptions`], *optional*):
                The stream_options parameter in chat_completion is used to configure how streaming responses are returned. 
                Specifically, it allows you to include additional usage statistics in the streamed output.
                If include_usage=True, an additional chunk is streamed before the final [DONE] message, containing:
                    - Total token usage (how many tokens were used in prompt & response).
                    - All other chunks will include a usage field, but set to null.
                Example:
                response = client.chat_completion(
                    messages,
                    stream=True,
                    stream_options=ChatCompletionInputStreamOptions(include_usage=True),
                )
                for chunk in response:
                    print(chunk)
                ChatCompletionStreamOutput(choices=[], usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})  # ✅ First chunk contains usage
                ChatCompletionStreamOutput(choices=[...], usage=None)  # First token
                ChatCompletionStreamOutput(choices=[...], usage=None)  # Next token
                ...
                [Done]  # End of response               
            temperature (`float`, *optional*):
                Controls randomness of the generations. Lower values ensure
                less random completions. Range: [0, 2]. Defaults to 1.0.
            tools (List of [`ChatCompletionInputTool`], *optional*):
                A list of tools the model may "call". Currently, only functions are supported as a tool. Use this to
                provide a list of functions the model may generate JSON inputs for. 
                Example:
                messages = [{"role": "user", "content": "What's the weather like in New York?"}]
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Fetches weather data for a given city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "City name"},
                                },
                                "required": ["city"]
                            }
                        }
                    }
                ]
                response = client.chat_completion(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",  # Let the model decide if a tool is needed
                    tool_prompt="Use the most recent weather data before answering.",
                )
            tool_choice ([`ChatCompletionInputToolChoiceClass`] or [`ChatCompletionInputToolChoiceEnum`], *optional*):
                This parameter determines how the model selects a tool from the available tools list.
                It accepts either:
                - A specific function name using ChatCompletionInputToolChoiceClass
                - A predefined behavior setting using ChatCompletionInputToolChoiceEnum
                Example:
                from text_generation.types import ChatCompletionInputToolChoiceClass, ChatCompletionInputFunctionName
                tool_choice = ChatCompletionInputToolChoiceClass(
                    function=ChatCompletionInputFunctionName(name="get_weather")
                )
            tool_prompt (`str`, *optional*):
                The tool_prompt parameter allows you to append extra instructions before the tool is executed.
                Example: 
                tool_prompt = "Use the most recent weather data before answering."
            top_logprobs (`int`, *optional*):
                An integer between 0 and 5 specifying the number of most likely tokens to return at each token
                position, each with an associated log probability. logprobs must be set to true if this parameter is
                used.
            top_p (`float`, *optional*):
                Fraction of the most likely next words to sample from.
                Must be between 0 and 1. Defaults to 1.0.
        """
        ...

    def feature_extraction(
        self,
        texts: str | List[str],
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
    ) -> Annotated["np.ndarray", Doc('The returned embedding represents the input text as a float32 numpy array.')]:
        """
        Args:
            texts (`str`, `List[str]`):
                The text to embed.
            normalize (`bool`, *optional*):
                Whether to normalize the embeddings or not.
                Only available on server powered by Text-Embedding-Inference.
                Determines whether the returned embedding vectors should be normalized (i.e., converted to unit length).
                Normalization ensures that the output embedding vectors have a unit norm (i.e., their magnitude is 1). 
                This is done using L2 normalization
            prompt_name (`str`, *optional*):
                The name of the prompt that should be used by for encoding. If not set, no prompt will be applied.
                Used to prepend a specific prompt template before encoding the input text. 
                This is mainly used with models that support customized prompt formats, such as those trained with Sentence Transformers.
                The prompt_name tells the model to apply a predefined template before encoding the text.
                For example, if the model is trained with prompts for queries and documents, 
                you might want different representations of the same text depending on whether it's a query (short search phrase) or a document (longer context).
                Example:
                embedding_query = client.feature_extraction("What is AI?", prompt_name="query")
                The input "What is AI?" will be transformed to "query: What is AI?" before embedding
                embedding_document = client.feature_extraction("What is AI?", prompt_name="document")
                The input "What is AI?" will be transformed to "document: What is AI?" before embedding.
            truncate
                Whether to truncate the embeddings or not.
                Whether to truncate (cut off) input text if it exceeds the model's maximum token limit.
                Most text embedding models (such as BAAI/bge-large-en-v1.5) have a fixed token limit (e.g., 512 tokens).
                If an input text exceeds this limit, the model cannot process the extra tokens.
                If truncate=True, the model automatically cuts the text down to the allowed length before embedding.
                Only available on server powered by Text-Embedding-Inference.
            truncation_direction (`Literal["Left", "Right"]`, *optional*):
                Which side of the input should be truncated when `truncate=True` is passed.        
        """
        ...

class HuggingFaceBaseInferenceClient(HuggingFaceTEIMixin):
    client: Optional[HuggingFaceInferenceLike] = Field(description='A low-level Inference Client that implements the HuggingFaceInferenceLike protocol', default=None)
    async_client: Optional[HuggingFaceInferenceLike] = Field(description='A low-level Async Inference Client that implements the HuggingFaceInferenceLike protocol', default=None)

    model_config = ConfigDict(
        extra='forbid',
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed_url = urlparse(value)
        if parsed_url.path not in ('', '/'):
            raise ValueError(f'Invalid base_url: {value}. Must not contain extra path segments.')
        return value
    
class HuggingFaceInferenceClient(HuggingFaceBaseInferenceClient):
    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        try:
            from huggingface_hub import (
                InferenceClient, 
                AsyncInferenceClient
            )

            client = InferenceClient(
                model=self.base_url,
                api_key=self.credentials,
                timeout=self.timeout,
                headers=self.headers
            )

            self.client = client

            async_client = AsyncInferenceClient(
                base_url=self.base_url,
                api_key=self.credentials,
                timeout=self.timeout,
                headers=self.headers
            )

            self.async_client = async_client
        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        
        return self

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[False] = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> ChatCompletionOutput: 
        ...

    @overload
    def chat_completion(  # type: ignore
        self,
        messages: List[Dict],
        *,
        stream: Literal[True] = True,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Iterable[ChatCompletionStreamOutput]: 
        ...

    @overload
    def chat_completion(
        self,
        messages: List[Dict],
        *,
        stream: bool = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]: 
        ...

    def chat_completion(
        self,
        messages: List[Dict],
        *,
        stream: Optional[bool] = False,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        num_generations: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatCompletionInputGrammarType] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        stream_options: Optional[ChatCompletionInputStreamOptions] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ChatCompletionInputTool]] = None,
        tool_choice: Optional[Union[ChatCompletionInputToolChoiceClass, "ChatCompletionInputToolChoiceEnum"]] = None,
        tool_prompt: Optional[str] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletionOutput, Iterable[ChatCompletionStreamOutput]]:
        self.client.chat_completion(
            messages=messages,
            stream=stream,
            frequency_penalty=frequency_penalty,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=num_generations,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options, # returns tokens used if set
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt=tool_prompt,
            top_logprobs=top_logprobs,
            top_p=top_p
        )

    def feature_extraction(
        self,
        texts: str | List[str],
        *,
        normalize: Optional[bool] = None,
        prompt_name: Optional[str] = None,
        truncate: Optional[bool] = None,
        truncation_direction: Optional[Literal["Left", "Right"]] = None,
    ) -> "np.ndarray":
        provider_helper = get_provider_helper('hf-inference', task='feature-extraction')
        request_parameters = provider_helper.prepare_request(
            inputs=texts,
            parameters={
                "normalize": normalize,
                "prompt_name": prompt_name,
                "truncate": truncate,
                "truncation_direction": truncation_direction,
            },
            headers=self.headers or {},
            model=self.base_url,
            api_key=self.credentials,
        )

        response = self.client._inner_post(request_parameters)
        np = _import_numpy()
        return np.array(_bytes_to_dict(response), dtype="float32")