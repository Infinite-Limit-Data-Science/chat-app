### Consuming

**There are many ways to consume Text Generation Inference (TGI) server in your applications. After launching the server, you can use the Messages API /v1/chat/completions route and make a POST request to get results from the server. You can also pass "stream": true to the call if you want TGI to return a stream of tokens**.

You can make the requests using any tool of your preference, such as curl, Python, or TypeScript.

1) curl

After a successful server launch, you can query the model using the v1/chat/completions route, to get responses that are compliant to the OpenAI Chat Completion spec:

```shell
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is deep learning?"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
    -H 'Content-Type: application/json'
```

For non-chat use-cases, you can also use the /generate and /generate_stream routes.

```shell
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{
  "inputs":"What is Deep Learning?",
  "parameters":{
    "max_new_tokens":20
  }
}' \
    -H 'Content-Type: application/json'
```

2) Python

huggingface_hub is a Python library to interact with the Hugging Face Hub, including its endpoints. It provides a high-level class, **huggingface_hub.InferenceClient, which makes it easy to make calls to TGI’s Messages API. InferenceClient also takes care of parameter validation and provides a simple-to-use interface**.

Install huggingface_hub package via pip.

```shell
pip install huggingface_hub
```

You can now use InferenceClient the exact same way you would use OpenAI client in Python

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="http://localhost:8080/v1/",
)

output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count to 10"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```

There is also an async version of the client, AsyncInferenceClient, based on asyncio and aiohttp.

3) OpenAI Client

You can directly use the OpenAI Python or JS clients to interact with TGI.

Install the OpenAI Python package via pip.

```shell
pip install openai
```

```python
from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://localhost:8080/v1/",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant." },
        {"role": "user", "content": "What is deep learning?"}
    ],
    stream=True
)

# iterate and print stream
for message in chat_completion:
    print(message)
```