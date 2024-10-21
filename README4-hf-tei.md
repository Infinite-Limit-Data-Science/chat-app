### HuggingFace Text Embedding Inference

https://huggingface.co/docs/text-embeddings-inference/index

Text Embeddings Inference (TEI) is a comprehensive toolkit designed for efficient deployment and serving of open source text embeddings models. It enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE, and E5.

Key Features:

- Streamlined Deployment: TEI eliminates the need for a model graph compilation step for an easier deployment process.

- Efficient Resource Utilization: Benefit from small Docker images and rapid boot times, allowing for true serverless capabilities.

- Dynamic Batching: TEI incorporates token-based dynamic batching thus optimizing resource utilization during inference.

- Optimized Inference: TEI leverages Flash Attention, Candle, and cuBLASLt by using optimized transformers code for inference.

- Safetensors weight loading: TEI loads Safetensors weights for faster boot times.

- Production-Ready: TEI supports distributed tracing through Open Telemetry and exports Prometheus metrics.

The easiest way to get started with TEI is to use one of the official Docker containers. After making sure that your hardware is supported, install the NVIDIA Container Toolkit if you plan on utilizing GPUs. NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher. Finally, deploy your model. Let’s say you want to use BAAI/bge-large-en-v1.5. Here’s how you can do this:

```shell
token=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
model=BAAI/bge-large-en-v1.5
volume=$PWD/teidata

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8070:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-client-batch-size 65 --auto-truncate
```

Once you have deployed a model, you can use the embed endpoint by sending requests:

```shell
curl 100.28.34.190:8070/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'

curl 100.28.34.190:8070/embed -X POST -d '{"inputs":"What is Deep Learning?"}' -H 'Content-Type: application/json'
[[0.018113142,0.00302585,-0.049911194,-0.03506436,0.0142142335,-0.023612928,-0.01585384,-0.021689294,-0.005009194,0.063208796,0.0146531835,0.028402645,0.023948595,-0.034057356,-0.019468723,-0.0037794884,-0.016912485,-0.0066746217,-0.04616721,-0.0037213922,-0.027137438,0.05071163,-0.027782952,-0.0048316773,-0.037000902,0.011477251,0.071368106,0.029125623,0.047742266,0.034986895,-0.019042684,-0.025898049,0.0071845786,-0.03245648,-0.02070811,-0.012503619,0.009986112,-0.024891047,-0.06362193,-0.051537894,-0.015195415,0.0324823,0.023173977,-0.094038576,-0.051150583,0.011231955,0.038446855,0.0021301985,-0.016331522,-0.003314718,-0.008236768,0.030055163,-0.009024295,-0.014420798,0.016822113,-0.021418177,0.014988851, ...
```

Look at all those beautiful vectors above!

### Inputs

An input represents a sentence in a corpus. Therefore, an input may be refered to as a Document in NLP. In the context of a corpus in Natural Language Processing (NLP), a document refers to a single unit of text within the corpus. The exact definition of what constitutes a "document" can vary based on the use case and the structure of the corpus, but generally, it is a discrete and self-contained piece of text that can be processed, analyzed, or understood on its own. A document is a segment of text that contains relevant content for the specific NLP task. It could be an article, a web page, a book, a paragraph, or even a sentence, depending on the size and scope of the corpus. The level of granularity for what constitutes a "document" depends on the specific NLP task:

- Sentence-Level Tasks: In some cases, like sentiment analysis or machine translation, each sentence may be treated as a document.
- Paragraph or Full-Text-Level Tasks: In other cases, such as topic modeling or document classification, a document could refer to a full article, essay, or book.

Below we have two Requests (we will talk about next). With the /embed endpoint below, we have two inputs. Each represents a sentence (or Document). Note Input length is not fixed. Each Input has a different token (word, partial word, or punctuation mark) length:

```shell
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["Today is a nice day", "I like you"]}' \
    -H 'Content-Type: application/json'
And for Sequence Classification:


curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":[["I like you."], ["I hate pineapples"]]}' \
    -H 'Content-Type: application/json'
```

It's very important to remember each Input can have a different token length!!!! **Using the /embed endpoint on the vm you deployed the TEI is a great way to test the model directly with inputs, batching, context lengths!!!!!!!**

Relationship to a Corpus:

- A corpus is a collection of documents, and each document represents one unit of analysis within that corpus.
- The corpus can be used for various NLP tasks such as training machine learning models, performing text analytics, or extracting knowledge.

In a sentiment analysis task, a "document" might be a product review (several sentences) from an online store. For topic modeling, a document could be an entire blog post or article that the model will categorize into a particular topic.

### Requests

In the context of HuggingFace's Text Embeddings Inference (TEI), the term requests refers to individual HTTP requests sent to the server's /embed endpoint. Each request can contain one or more text inputs that the client wants to embed. These requests are typically sent from different clients or even the same client but at different times.

A request is a single HTTP POST operation made by a client to the server (for example, to the /embed endpoint) to perform a specific task, such as embedding a set of text inputs. Each request can contain one or more text inputs that need to be processed by the model. The below example is one request made by a client containing two text inputs.

```shell
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["Today is a nice day", "I like you"]}' \
    -H 'Content-Type: application/json'
```

### Batches and Dynamic Batching

How is a Request different from a Batch. A batch refers to a group of requests (or parts of requests) that are combined together for processing by the model in one go. Batching is typically done by the server for efficiency purposes, especially in scenarios like dynamic batching, where multiple requests are handled at the same time.

Example of a Batch: If two different users send requests to the server at nearly the same time, and the total token count across both requests is within the server's limit, the server may group both requests into a single batch and process them together.

Dynamic Batching in the context of machine learning inference, particularly in systems like Hugging Face's Text Embeddings Inference (TEI), refers to the process of dynamically grouping or batching multiple inference requests together based on their token size or computational requirements. This technique is used to improve resource efficiency and throughput during the inference process by combining smaller or individual requests into a single, larger batch that can be processed more efficiently by the model.

How Dynamic Batching Works:

- Token-Based Batching: Dynamic batching often involves organizing incoming text inputs into groups based on the number of tokens they contain. Requests that arrive at roughly the same time and have similar token lengths can be processed together as a batch.
- Dynamic vs. Static Batching: Traditional static batching requires the system to wait for a fixed-size batch before processing, which can lead to delays or inefficiencies if requests don't perfectly align. In contrast, dynamic batching adjusts the batch size based on real-time traffic, reducing idle time and making better use of available hardware.

Benefits of Dynamic Batching:

- Improved Resource Utilization: By dynamically adjusting the batch size, TEI ensures that the GPU or other processing units are fully utilized, even when handling varying input sizes or unpredictable request patterns.
- Lower Latency: Dynamic batching can minimize waiting times by immediately grouping requests that arrive within a short window, leading to faster overall response times compared to waiting for a full static batch.
- Optimized for Real-Time: Since real-time applications might see varying request loads, dynamic batching allows for flexible and efficient processing without compromising on performance.

In the TEI system, dynamic batching helps ensure that the text embeddings models—such as FlagEmbedding, GTE, and E5—are able to process multiple requests at once while minimizing latency. This is especially important in production environments where there are many concurrent user requests that can vary significantly in size.

Dynamic batching is critical for optimizing performance in cloud environments where the system must balance between high throughput (handling many requests at once) and low latency (responding quickly to individual users).

In summary, dynamic batching in TEI ensures that text embedding models can efficiently handle multiple concurrent requests by grouping them based on token size, optimizing the use of hardware resources, and reducing inference latency.

Suppose we have two separate curl requests that send text inputs to the /embed endpoint:

```shell
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["Today is a nice day", "I like you"]}' \
    -H 'Content-Type: application/json'

curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["The weather is beautiful today", "Let’s go for a walk"]}' \
    -H 'Content-Type: application/json'
```

The first one sends a request with two inputs: "Today is a nice day" and "I like you". The second sends another request with two inputs: "The weather is beautiful today" and "Let’s go for a walk". Both of these curl requests are received by the TEI server in a short time window. Instead of handling each request individually, the server dynamically batches them together because:

- The total number of tokens (from both requests) is within the limit of the context window (e.g., 8k tokens).
- The TEI system groups these inputs into a single batch for more efficient processing.

TEI processes the combined batch in one go, allowing the system to make better use of hardware (e.g., GPU) resources. Once processed, each request is sent back to the corresponding client as if they were handled individually, but the server benefits from processing them together.

Internal Process (Under the Hood):

- Token-based grouping: The system checks the token count for all input texts and dynamically creates a batch based on the total number of tokens.
- Efficient GPU utilization: By batching multiple requests, the GPU is able to handle larger computations more efficiently, reducing the overhead from processing small requests one by one.
- Reduced Latency: Even though the server groups requests, it doesn’t significantly increase the latency for individual clients. Instead, it optimizes overall throughput by making efficient use of available resources.

Both curl invocations will receive a JSON response containing the embeddings for each input text, but behind the scenes, they were dynamically batched together to improve performance.

There are three crucial parameters for batching the HF TEI supports: --max-batch-tokens, --max-batch-requests, and --max-client-batch-size. Here’s a breakdown of how --max-batch-requests works alongside the other parameters like --max-batch-tokens and --max-client-batch-size:

- --max-batch-tokens: This parameter controls the total number of tokens that can be processed in a single batch. For example, if it's set to 1000, the system can process up to 1000 tokens in a single batch, even if the tokens come from multiple requests.
- --max-client-batch-size: This parameter limits the maximum number of inputs a single client can send in one request. For example, if this is set to 32, a single client can submit up to 32 text inputs in one request.
- --max-batch-requests: This parameter limits the maximum number of individual requests that can be combined into a single batch for processing. This is particularly useful in environments where there may be many smaller requests arriving at nearly the same time. TEI can combine multiple requests into one batch up to the limit specified by this parameter. Example: If --max-batch-requests=5, then up to 5 separate requests can be grouped into a single batch, assuming they fit within the limits defined by --max-batch-tokens and don’t exceed hardware limits.

**How These Parameters Work Together. Scenario: You have --max-batch-tokens=1000, --max-client-batch-size=32, and --max-batch-requests=10. If two users send requests at the same time, and each request has 32 inputs that sum up to a total of 800 tokens across both requests, TEI could batch both requests together as they fit within the 1000 token limit. If a third request comes in within the same time window, and the total token count across all three requests remains under 1000, TEI can batch all three. However, if the total number of requests reaches the --max-batch-requests limit (e.g., 10), TEI will stop adding new requests to the batch, even if there is room within the token limit.**

### The Inputs, Tokens, Batch Calculation

To estimate how many inputs 𝑛 will get you 𝑥 tokens in HuggingFace’s Text Embeddings Inference (TEI), you can follow a general process based on the characteristics of your text inputs and the tokenizer used by the model. 

When determining chunk size, should I consider the context length of the embedding model BAAI/bge-large-en-v1.5 which is 512 or the LLM which is Llama 3.1 70B Instruct, which has a context length of 128k? When determining the chunk size for your use case, you should prioritize the context length of the embedding model (in this case, BAAI/bge-large-en-v1.5 with a context length of 512 tokens), rather than the LLM (Llama 3.1 70B Instruct) with a larger context length of 128k tokens.

Why Prioritize the Embedding Model's Context Length?

- Embedding Model Limitation: The embedding model (BAAI/bge-large-en-v1.5) can only process a maximum of 512 tokens at a time. This is a hard limit, meaning that if any input chunk exceeds 512 tokens, the model will not be able to process it fully. Hence, your chunk size should be tailored to fit within this limit to ensure the embedding model works as intended.
- LLM Context Length: The Llama 3.1 70B Instruct model, which has a much larger context window of 128k tokens, is used later in the pipeline (presumably for generating responses or handling tasks that require more extensive context). However, by the time you're working with the LLM, the text has already been embedded, meaning it's the output from the embedding model (chunks of embeddings) that will be passed to the LLM, not the raw text itself. The LLM's context length comes into play when the embeddings or responses from the embedding model are combined for further processing, but at this point, the embedding model has already handled the chunking.

Practical Approach:

- As the name implies, RecursiveCharacterTextSplitter in Langchain works with character units when determining the chunk_size. This means that the size of the chunks it creates is based on the number of characters in the text, rather than tokens or words. Set your chunk size based on the 512-token limit of the embedding model. You can use the RecursiveCharacterTextSplitter to generate chunks of approximately 512 tokens (e.g., setting a chunk size of around 2000 characters, based on the rough estimate that 1 token ≈ 4 characters).
















### Context Length

In the context of Large Language Models (LLMs), an 8k context length refers to the model's ability to process and retain a sequence of up to 8,000 tokens (or pieces of text) in a single input or output session. This "context window" defines how much text the model can consider at once when generating responses or embeddings.

Tokens: Tokens are smaller units of text that can be words, parts of words, or punctuation marks. In many models, one token is roughly equivalent to 4 characters of English text. So, a context length of 8,000 tokens corresponds to approximately 6,000-8,000 words (depending on the language and specific text).

Embedding Models: In LLMs used for embeddings, the context length determines the maximum size of text that can be processed at once for generating embeddings. If the text exceeds the context limit (8k tokens), it must be split into chunks that fit within the model’s context window. For example, the max context length for BAAI/bge-large-en is limited to 512. For sentences whose length is larger than 512, it will process only the first 512 tokens. If you have auto truncate flag on when launching the HF TEI container, inputs exceeding 512 will be truncated. We suggest splitting the long document into several passages.

Use Cases: This is especially relevant for tasks involving long documents (e.g., scientific papers, legal texts) or conversations where the model needs to remember and analyze longer spans of text. For instance, if you're feeding a long document into an embedding model to extract vector representations, an 8k context length ensures that large chunks of the text can be processed at once without losing important context.

Performance: Models with larger context windows (e.g., 8k, 16k, or even 100k tokens) are capable of retaining more information, making them better suited for tasks requiring understanding of larger sequences of text, such as summarization, question answering over long documents, or complex multi-step reasoning.

MAX_BATCH_TOKENS is chosen based on our number of workers and the context window of our embedding model.
NUMBER OF WORKERS!!!!

MAX_WORKERS argument

--tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation.
          Default to the number of CPU cores on the machine

You should consider further tuning MAX_BATCH_TOKENS and MAX_CONCURRENT_REQUESTS if you have high workloads

You may have inputs that exceed the context. In such scenarios, it’s up to you to handle them. In my case, I’d like to truncate rather than have an error. Let’s test that it works.

embedding_input = "This input will get multiplied" * 10000
print(f"The length of the embedding_input is: {len(embedding_input)}")
response = endpoint.client.post(json={"inputs": embedding_input, "truncate": True}, task="feature-extraction")
response = np.array(json.loads(response.decode()))
response[0][:20]

https://huggingface.co/learn/cookbook/automatic_embedding_tei_inference_endpoints

### WORKERS, Max Concurrent Requests

```shell
token=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
model=BAAI/bge-large-en-v1.5
volume=$PWD/teidata

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8070:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-client-batch-size 128 --auto-truncate
8 workers (4096 )
max batch tokens 4096

```

512 * 8 = 4096

input 

--max-batch-tokens (4096)
--max-batch-requests (8)
--max-client-batch-size (128)


Split by 2000 characters (to have faster processing)
Let's say each input is 2000 characters (which is 500 tokens)
inputs per request (4096 / 500 = 8)



The input to be less than 512 tokens

--max-batch-requests: This parameter limits the maximum number of individual requests that can be combined into a single batch for processing.






ubuntu@ip-172-31-41-115:~$ nvidia-smi
Sun Oct 20 20:57:41 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1E.0 Off |                    0 |
|  0%   33C    P0    58W / 300W |    940MiB / 23028MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+



token=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
model=BAAI/bge-large-en-v1.5
volume=$PWD/teidata

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8070:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 4096 --max-batch-requests 8 --max-client-batch-size 128 --tokenization-workers 8

--auto-truncate