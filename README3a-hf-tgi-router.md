### Text Generation Inference Architecture

These are the separate components:

- Router: also named webserver, that receives the client requests, buffers them, creates some batches, and prepares gRPC calls to a model server. In this context, gRPC (gRPC Remote Procedure Call) is a protocol used to enable efficient, high-performance communication between different components of a system, especially useful for distributed architectures. 
- Model Server: responsible of receiving the gRPC requests and to process the inference on the model. If the model is sharded across multiple accelerators (e.g.: multiple GPUs), the model server shards might be synchronized via NCCL or equivalent. NVIDIA Collective Communications Library (NCCL) is a library developed by NVIDIA to provide high-performance communication primitives optimized for multi-GPU and multi-node environments. It is commonly used to synchronize data across multiple GPUs, enabling efficient distributed training and inference of large models.
- Launcher: helper that will be able to launch one or several model servers (if model is sharded), and it launches the router with the compatible arguments.

The router and the model server can be two different machines, they do not need to be deployed together.

### VERY IMPORTANT: Text Generation Inference Architecture: The Router
This component is a rust web server binary that accepts HTTP requests using the **custom HTTP API, as well as OpenAI’s Messages API**. The **router receives the API calls and handles the “baches” logic**. It uses different strategies to reduce latency between requests and responses, especially oriented to decoding latency. It will use queues, schedulers, and block allocators to achieve that and produce batched requests that it will then be sent to the model server.

1) MAX_CONCURRENT_REQUESTS

```shell
--max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
    [env: MAX_CONCURRENT_REQUESTS=] [default: 128]
```

In the Hugging Face Text Generation Inference (TGI) CLI, the --max-concurrent-requests argument controls the maximum number of concurrent requests the server can handle for text generation. This setting helps manage server load by limiting the number of simultaneous generation requests that can be processed at any given time.

Concurrency Control: By setting a limit on concurrent requests, --max-concurrent-requests helps prevent the server from becoming overloaded with too many requests at once, which could degrade performance or exhaust system resources.
Resource Management: Each request requires CPU, memory, and potentially GPU resources. Limiting concurrent requests ensures that there are enough resources available for each request to be processed efficiently.

**The --max-concurrent-requests parameter in Hugging Face Text Generation Inference (TGI) controls the maximum number of concurrent requests that the server can handle across all batches, not per batch. This setting defines the overall limit on the number of simultaneous requests that can be processed by the server at any given time, regardless of how they are batched.**

**In Hugging Face Text Generation Inference (TGI), the --max-input-tokens and --max-total-tokens parameters apply to each request individually**. Here’s how these limits work on a per-request basis:

- max_input_tokens: This setting defines the maximum number of input tokens allowed for each individual request. If a request’s input exceeds this token limit, it will be rejected or truncated, depending on the server’s configuration. Need to check source code of hf tgi to see how it handles it.
- max_total_tokens: This setting defines the maximum total number of tokens allowed for each request, which includes both the input tokens and any generated tokens in the response. For example, if max_total_tokens is set to 1024, and a request has 400 input tokens, the model will be able to generate up to 624 tokens for the response, bringing the total to 1024.

These limits apply to each request separately, meaning each concurrent request handled by TGI will be subject to its own individual token restrictions as defined by max_input_tokens and max_total_tokens. This ensures that the server manages memory and compute resources effectively across multiple requests without any single request exceeding the specified token limits.

2) MAX_INPUT_TOKENS

```shell
--max-input-tokens <MAX_INPUT_TOKENS>
    [env: MAX_INPUT_TOKENS=] [default: 1024]
```

- Purpose: Sets the maximum number of tokens allowed in the input prompt for a request.
- Usage: This ensures that a request with too many input tokens is restricted, preventing overly large input sequences from causing delays or overloading the server.
- Example: If MAX_INPUT_TOKENS=512, the server will reject requests with an input length greater than 512 tokens.

3) MAX_TOTAL_TOKENS

- Purpose: Sets the maximum total number of tokens (input tokens + generated tokens) allowed for a request.
- Usage: This limit ensures that the combined size of the input and generated output does not exceed a defined threshold, helping to manage memory and computational load.
- Example: If MAX_TOTAL_TOKENS=1024 and a request has 500 input tokens, it can generate up to 524 output tokens before hitting this limit.

3) MAX_BATCH_PREFILL_TOKENS

```shell
--max-batch-prefill-tokens <MAX_BATCH_PREFILL_TOKENS>
    [env: MAX_BATCH_PREFILL_TOKENS=] [default: 4096]
```

Limits the number of tokens for the prefill operation. Since **this operation take the most memory and is compute bound**, it is interesting to limit the number of requests that can be sent. Default to `max_input_tokens + 50` to give a bit of room

The MAX_BATCH_PREFILL_TOKENS setting in Hugging Face TGI is designed to control the maximum number of tokens processed during the prefill operation for a batch of requests. The prefill phase is a crucial part of text generation in which the model processes the initial tokens before producing new tokens for each request.

What is the Prefill Operation?

- The prefill operation is the first phase in text generation, where the model reads the input prompt and prepares the context for generating new tokens.
- This phase involves loading the input tokens into the model’s memory and processing them before it can start generating (or "sampling") new tokens for the response.
- **The prefill operation tends to be memory-intensive and compute-bound because it involves processing the input tokens through the model’s layers. What Are Model Layers? A layer in a neural network is a collection of neurons or units that perform a specific transformation on the input data**. In the context of transformer models (e.g., GPT, BERT), these layers consist of several components designed to learn and extract patterns and dependencies from the input tokens. **The Self-Attention Layer computes the attention scores between every token in the input sequence, determining which tokens should be more focused on based on their relationships with others in the sequence**. It uses queries (Q), keys (K), and values (V) derived from the input tokens to calculate these relationships. **The result of the self-attention operation is a weighted representation of the input tokens that reflects the context and relationships among them.** Feed-Forward Layer: After the self-attention layer, a feed-forward layer processes the output further. It consists of one or more dense (fully connected) layers that transform the input using activation functions (e.g., ReLU). Normalization Layers and Residual Connections are sometimes additional layers.

The prefill operation is the initial phase where the model processes the input tokens through these layers to build the context needed for generating further tokens. Here’s how the layers are involved:

1) Input Embedding: The input tokens are first converted into embeddings, which are dense vector representations that capture the meaning and characteristics of each token. 
2) Passing Through Transformer Layers: The embeddings are passed through multiple transformer blocks (a stack of self-attention, feed-forward, normalization, and residual connections). Each layer processes the embeddings, gradually building up a contextualized representation of the input sequence. This involves computing self-attention scores and transforming the input using the feed-forward layers.
3) Building Context: The output from each transformer layer provides a refined and contextualized representation of the input tokens. By the end of the prefill operation, the model has processed the entire sequence through all the layers, creating an internal state or context that encodes the relationships and patterns present in the input.
4) Ready for Generation: Once the input has passed through all the layers, the model is ready to start generating new tokens based on this context. The prefill operation essentially "primes" the model with the input data, setting up the necessary information to produce output tokens in the next phase.

Why Is This Memory-Intensive and Compute-Bound?

- Memory-Intensive: As the input tokens pass through multiple layers, the model stores intermediate activations and states for each layer. This is particularly demanding for models with many layers (e.g., dozens or even hundreds of transformer blocks) and large input sequences, leading to significant memory consumption.
- Compute-Bound: Each layer involves complex matrix multiplications, especially in the self-attention mechanism, where the model computes relationships between every token pair in the sequence. This requires substantial computational resources, especially as the number of tokens and layers increase.

How is --max-batch-prefill-tokens any different from max_input_tokens? max_input_tokens is a configuration limit that defines the maximum number of input tokens allowed for any given request. It acts as a hard limit to ensure that no individual request has an input that exceeds a certain length, helping to control resource usage and prevent extremely large requests from overloading the server.

**--max-batch-prefill-tokens is indeed a limit in Hugging Face Text Generation Inference (TGI), but it applies specifically to batch processing during the prefill operation rather than to individual requests.**

- Scope: **--max-batch-prefill-tokens limits the total number of tokens across all requests in a batch during the prefill operation, rather than just for a single request.**
- Purpose: This setting controls the maximum token load that TGI will process at once when multiple requests are batched together. It helps manage memory usage and computational load when handling a group of requests concurrently.
- Example: If --max-batch-prefill-tokens is set to 4096, then during batch processing, the combined input tokens of all requests in that batch cannot exceed 4096 tokens. If the batch’s total input token count is above this threshold, TGI will either split the batch or delay some requests to stay within the limit.

4) MAX_INPUT_TOKENS, MAX_TOTAL_TOKENS, and MAX_BATCH_PREFILL_TOKENS together

Suppose we want to set up TGI to:

- Limit each request’s input tokens to 512.
- Set a cap on the total tokens per request (input + generated tokens) to 1024.
- Restrict the prefill phase for batches to 4096 tokens total, to avoid excessive memory usage when batching requests.

```shell
  --model-id my-large-model \
  --max-input-tokens 512 \
  --max-total-tokens 1024 \
  --max-batch-prefill-tokens 4096
```

--max-input-tokens 512:
- This setting restricts each individual request’s input length to 512 tokens.
- If a request tries to send more than 512 tokens as input, it will be rejected or truncated, depending on TGI’s handling strategy.
- This limit helps control resource usage by preventing requests with very large input prompts from consuming too much memory.

- --max-total-tokens 1024:
- This setting limits each individual request’s combined total tokens (input tokens + output tokens) to 1024.
- For example, if a request uses 512 input tokens, the model will generate up to 512 output tokens, making the total 1024.
- This ensures that no single request can demand excessive output tokens, which could slow down the server or exceed available resources.

- --max-batch-prefill-tokens 4096:
- **This setting limits the total tokens across all input prompts in a batch to 4096 tokens during the prefill phase.**
- **For example, if the TGI server receives a batch of 8 requests, each with 512 input tokens, the total input tokens for that batch would be exactly 4096, so the batch is processed as-is.**
- **If the batch’s total input tokens exceed 4096 (e.g., if there are 10 requests with 512 tokens each), TGI will either split the batch or queue some requests to stay within the 4096 token limit.** TODO: look into hf tgi source code to see exactly what it does.
- This prevents memory and compute overload by capping the prefill operation’s token load, optimizing batching efficiency without risking resource saturation.

5) MAX_BATCH_TOTAL_TOKENS

```shell
--max-batch-total-tokens <MAX_BATCH_TOTAL_TOKENS>
    [env: MAX_BATCH_TOTAL_TOKENS=]
```

The --max-batch-total-tokens parameter in Hugging Face TGI sets a **hard limit on the total number of tokens (input + generated tokens) across all requests within a batch**. It’s a critical control for managing memory usage effectively, especially when multiple requests are batched together.  It ensures that the cumulative memory and compute load from multiple requests doesn’t exceed the hardware's capacity. This parameter is essential when running large models because TGI needs to manage memory carefully to avoid overloading the hardware (e.g., GPUs or TPUs) where the model is deployed.

Example Scenarios:

If --max-batch-total-tokens=1000, then:

- You could have 10 requests each with 100 tokens (10 * 100 = 1000).
- Alternatively, a single request with 1000 tokens could also fit within this limit.
- However, 20 requests with 100 tokens each (20 * 100 = 2000) would exceed this limit, so the batch would either be split or some requests would be queued.

6) MAX_BATCH_TOTAL_TOKENS and FLASH ATTENTION

**Flash Attention allows TGI to avoid padding by only calculating attention on actual tokens (instead of adding "padding tokens" to match batch sizes)**. This leads to more efficient memory use. For instance, without Flash Attention, all requests in a batch would need to be padded to the same length, meaning unused tokens might consume memory. With Flash Attention, max_batch_total_tokens can be allocated more flexibly, allowing for a finer distribution of tokens across requests.

**The ideal max_batch_total_tokens value depends on available hardware resources. Set it as high as your hardware’s memory allows, considering factors like:**

- Model Size: Larger models consume more memory.
- Quantization: Quantized models use less memory, allowing a higher max_batch_total_tokens.
- Other Optimizations: Techniques like Flash Attention can allow for a higher setting compared to traditional padded attention.

Flash Attention is an optimized implementation of the attention mechanism that significantly improves memory efficiency and computation speed during the inference and training phases of transformer models. It achieves this by reorganizing the way attention is computed, allowing for efficient computation without padding.

Traditional Attention vs. Flash Attention

In a typical transformer model:

- The attention mechanism computes the relationship between tokens using query (Q), key (K), and value (V) matrices.
- **When processing batches of sequences (sequences referring to the tokenized text sent as input tokens to the model's layers) with different lengths, models typically pad shorter sequences with padding tokens ([PAD]) to make all sequences the same length. This padding helps align dimensions for matrix operations, but it also wastes memory and computation on these irrelevant tokens.** When we talk about sequences of different lengths in the context of transformer models and attention mechanisms, we're referring to text sequences (like sentences, paragraphs, or any input text) that have varying numbers of tokens. In natural language processing (NLP) and text generation tasks, the input to the model is a sequence of tokens (words, subwords, or characters). These tokens are derived from splitting and encoding the text using a tokenizer. Different sequences (e.g., sentences or paragraphs) naturally vary in length because they contain different numbers of words and characters. For example:

Sequence 1: "Hello, how are you?" might have 5 tokens.
Sequence 2: "This is an example of a longer input sequence, and it has more tokens." might have 15 tokens.
Sequence 3: "Short!" might have just 1 or 2 tokens.

When processing these sequences in a batch, the model needs to handle these different lengths efficiently. To manage these differences in length when processing multiple sequences in a batch, traditional transformer models typically use padding.  Shorter sequences are padded with special tokens (e.g., [PAD] token) to match the length of the longest sequence in the batch. This ensures that all sequences have the same length, which allows for efficient matrix operations (required for attention calculations).

So what is the secret of Flash Attention? Flash Attention eliminates the need for padding by reorganizing the computation of attention using several techniques:

- Fused Kernels for Memory Efficiency: Flash Attention uses fused kernels—specialized GPU operations that perform multiple steps (e.g., computing attention scores and applying softmax) within a single kernel call. This avoids intermediate memory allocations that traditional implementations need, reducing overall memory usage.
- Block-Sparse Computation: Instead of padding sequences to the same length, Flash Attention processes tokens in a block-sparse manner. It computes attention only for the actual tokens present in the sequence, skipping over any padding tokens entirely. By handling variable-length sequences dynamically, Flash Attention focuses on the relevant parts of the input data, enabling it to calculate attention scores only where needed, without padding.
- Custom Softmax Operation: Flash Attention uses a numerically stable, memory-efficient softmax implementation. It computes attention scores directly on smaller, non-padded segments of the sequences, rather than applying softmax over the entire padded length. This method significantly reduces both memory consumption and computational overhead, as the model only works with the actual data, not padding.
- Efficient Memory Usage through Streaming: Flash Attention performs attention computation in a streaming fashion, meaning it processes chunks of data directly on the GPU without having to store the full attention matrix in memory. This technique is key in managing long sequences, as it allows attention to be calculated incrementally, only on the needed tokens. By processing smaller chunks dynamically, Flash Attention is able to maintain high throughput while avoiding the overhead of managing large matrices filled with padding tokens.

In the context of TGI, when multiple requests are batched together, Flash Attention efficiently processes each request’s tokens individually, without padding them to match the longest sequence in the batch. This allows TGI to fit more sequences into a batch (up to the limit set by --max-batch-total-tokens) without wasting memory on padding. As a result, TGI achieves higher throughput and better memory utilization.

7) MAX_WAITING_TOKENS

```shell
--max-waiting-tokens <MAX_WAITING_TOKENS>
    [env: MAX_WAITING_TOKENS=] [default: 20]
```

MAX_WAITING_TOKENS is a parameter in Hugging Face Text Generation Inference (TGI) that controls how many tokens can accumulate before the server forces waiting queries to join the running batch. It is an important setting that helps balance the processing of new queries (requests) with the ongoing token generation in existing queries. 

This parameter helps manage the scheduling and batching of requests efficiently by determining when the server should pause ongoing tasks (like decoding tokens for the current batch) to process and integrate the new waiting queries.

**Two Phases in Text Generation**:

- **Prefill Phase: The initial phase where the model processes the input tokens to build the context. New queries require this phase**.
- **Decode Phase: The phase where the model generates tokens based on the context established during the prefill phase**.

Interaction Between New and Running Queries:

- New queries that arrive while the model is decoding (generating tokens) require a prefill operation to set up their context. This prefill operation is different from the ongoing decode operations.
- To process these new queries, the server needs to pause the running batch (which is currently decoding) and run the prefill operation for the waiting queries.

Effect of MAX_WAITING_TOKENS:

- When the number of tokens in the new waiting queries reaches the MAX_WAITING_TOKENS threshold, the server pauses the ongoing decoding and processes the prefill for these new queries, adding them to the running batch.
- If the threshold is too small: The server frequently pauses the decoding phase to accommodate new queries, leading to delays for existing requests as they get repeatedly interrupted.
- If the threshold is too large: New queries may have to wait too long before being processed and added to the batch. This results in higher latency for users submitting new queries, as they must wait until enough tokens accumulate to meet the threshold.

Why Use a Token-Based Threshold?

- The MAX_WAITING_TOKENS value is expressed in tokens rather than a time-based metric because the length of tokens is a more model-agnostic measure that reflects the workload's complexity.
- Using a token count helps balance the server's processing efficiently, regardless of the model's specific architecture or speed. However, the ideal value should be tuned based on the latency requirements for end users.

Examples of How MAX_WAITING_TOKENS Impacts Performance:

- Small Value (e.g., 5 tokens): The server frequently pauses the decoding to accommodate new queries, leading to constant interruptions. This could cause delays for the current batch, as it keeps getting interrupted to process new queries.
- Large Value (e.g., 50 tokens): New queries might wait too long before they are processed and added to the batch. This increases the latency for users submitting new requests because their queries sit idle until the server processes enough tokens to meet the threshold.
- Default Value (20 tokens): **The default value is a balanced setting that ensures new queries are not delayed excessively while avoiding frequent interruptions to ongoing token generation. However, this can be fine-tuned based on the server’s load and the desired latency**. By increasing this slightly, you can ensure existing prompts complete before a specified threshold (e.g. 40 seconds)!

8) MAX_BATCH_SIZE

```shell
--max-batch-size <MAX_BATCH_SIZE>
    Enforce a maximum number of requests per batch Specific flag for hardware targets that do not support unpadded inference
    
    [env: MAX_BATCH_SIZE=]
```

**MAX_BATCH_SIZE is not used for NVidia targets!!!!!!!!** Visit this ticket where the repo's owner says it: https://github.com/huggingface/text-generation-inference/issues/2615. What does this mean? When the repo owner, Narsil, refers to "Nvidia targets," he means deployments that run on Nvidia GPUs, as opposed to non-GPU (e.g., CPU) or other types of hardware accelerators. Also note in the argument reference: "Specific flag for hardware targets that do not support unpadded inference". Of course, NVidia CUDA GPUs support unpadded inference. That is the power of them. Narsil also notes that the MAX_BATCH_SIZE parameter is not used or considered when running on Nvidia targets, because TGI prioritizes the number of tokens in a batch rather than the number of users/requests. The server will automatically adjust VRAM usage based on available memory and the number of tokens.

In TGI, memory usage is primarily driven by the number of tokens being processed in a batch. This includes both the input tokens (the text provided by users) and the output tokens (the text generated by the model). For example, if multiple requests are grouped into a batch and the combined number of tokens in that batch increases, the VRAM usage will also increase accordingly. The server allocates memory to handle these tokens effectively. **TGI automatically adjusts how it uses VRAM based on the available GPU memory and the total number of tokens in a batch. This means that the server will maximize the number of tokens it can handle while staying within the constraints of the GPU's VRAM capacity.**

On Nvidia GPUs, TGI takes full advantage of VRAM to optimize the performance and handle as many tokens as possible within the memory limit. Unlike MAX_BATCH_SIZE, which focuses on the number of user requests, the token-based approach allows TGI to manage VRAM more flexibly by focusing on how many tokens are being processed at once, which directly impacts memory usage.

When Narsil says, "TGI will always use all the allowed memory for KV-cache, to allow MANY users on the same machine," he is referring to how Hugging Face's Text Generation Inference (TGI) server manages GPU memory to support high concurrency and efficient processing.

KV-cache stands for "Key-Value cache." In transformer models, during the generation process (e.g., for autoregressive tasks), intermediate representations of the input sequence (keys and values) are cached. This allows the model to efficiently generate new tokens without recalculating previous tokens' representations from scratch. What do you mean by generation process, are yuo talking about prefill operation or the decoding phase (where new tokens generated)? When discussing the KV-cache in the context of transformer models, the term generation process refers primarily to the decoding phase, where new tokens are generated.

Prefill Operation: This is the initial phase where the model processes the input sequence (e.g., a prompt). During this stage, the model reads and processes the input tokens to establish the context for generation. The KV-cache is not actively used during the prefill operation, as this stage involves creating the initial context by processing the input tokens through all layers of the model. **The outcome of this phase is the establishment of the key-value pairs (context) that will be used for subsequent generation.** 

The decoding phase is where the model generates new tokens based on the processed input and previously generated tokens. KV-cache plays a critical role here. The model stores the intermediate representations of the input and previously generated tokens in the KV-cache. This allows the model to reference these cached values when computing attention for new tokens, enabling efficient incremental generation without reprocessing the entire input sequence. During each step of the decoding phase, the model only processes the new token while reusing the stored key-value pairs from prior steps, which significantly speeds up generation.

**TGI is designed to maximize the utilization of GPU memory to handle as many concurrent generation requests as possible. By using all the allowed memory (as specified or available) for the KV-cache, TGI can store more key-value pairs, enabling the model to serve more users simultaneously.**

If you want to control how much VRAM you use, you need to set --cuda-memory-fraction 0.3 for instance to use 30% of available VRAM.

For GPUs that only support padded inference, MAX_BATCH_SIZE in Hugging Face Text Generation Inference (TGI) specifies the maximum number of requests that can be processed together in a single batch. Again, this setting is specific to unpadded inference (i.e., when all sequences in a batch must have the same length).

9) TOKENIZER_NAME

```shell
--tokenizer-name <TOKENIZER_NAME>
    [env: TOKENIZER_NAME=] [default: bigscience/bloom]
```

**The --tokenizer-name argument in Hugging Face Text Generation Inference (TGI) specifies the tokenizer that the server should use for processing text input. The tokenizer is a crucial component of any language model setup, as it converts text into tokens (subwords, words, or characters) that the model can understand and process**.

What is --tokenizer-name?

- This argument tells TGI which tokenizer to use for tokenizing the input text before it is fed into the model.
- **The value passed to --tokenizer-name should be the name of a pretrained tokenizer from Hugging Face's model hub (e.g., bigscience/bloom, gpt2, facebook/opt-66b), or it could point to a local directory where a tokenizer configuration file is stored**.
- **The tokenizer you specify should match the model you are using because different models have different tokenization strategies (e.g., byte-pair encoding (BPE), sentencepiece, etc.)**.

Why is --tokenizer-name Important?

- Ensures Compatibility: The tokenizer must be compatible with the model you are running. The model's vocabulary and tokenization strategy must align with the tokenizer, as the model was trained on inputs processed in a specific way. Using an incompatible tokenizer may produce unexpected results or cause errors.
- Handles Preprocessing: The tokenizer processes and encodes the input text into token IDs that the model can use. It also decodes model outputs back into human-readable text.
- Model-Specific Settings: Some tokenizers are highly specialized for their models, using customized vocabulary sizes, token types (e.g., special tokens like [CLS] or [SEP]), and tokenization rules. By specifying the tokenizer name, you ensure these model-specific settings are correctly applied.
- Below, the server is instructed to use the Bloom model (bigscience/bloom), and the --tokenizer-name specifies the corresponding tokenizer (bigscience/bloom). This ensures that the inputs and outputs are tokenized and detokenized in a manner consistent with how the model expects them.

```shell
text-generation-launcher --model-id bigscience/bloom --tokenizer-name bigscience/bloom
```

How the Tokenizer Works

- Tokenization: The tokenizer splits the input text into tokens (subwords, words, or characters) and converts them into token IDs that represent each token. These IDs are then passed to the model for processing.
- Decoding: After the model generates token IDs as output, the tokenizer decodes these IDs back into human-readable text.

You might need to change this argument if:

- You are using a custom tokenizer that has been specifically trained for a fine-tuned model or a new language model.
- The tokenizer is stored locally rather than available on Hugging Face’s hub. In such cases, you can point to the local directory containing the tokenizer files.

**Doesn't TGI load the model itself into memory when you run it? Would it load the tokenizer files too? Yes, when Hugging Face TGI loads a model, it automatically loads the tokenizer files along with it if they are available in the model’s repository**. Here’s how it works:

- **Model Repository: When you specify --model-id (e.g., meta-llama/Llama-3.1-70B-Instruct), TGI accesses the model’s repository on the Hugging Face Hub or from a local path, depending on where the model is stored**.
- Loading the Model: TGI loads the model weights and configurations (e.g., config.json and model checkpoint files) into memory. This step sets up the model for inference.
- Loading the Tokenizer: TGI looks for tokenizer files in the same repository as the model. These files typically include: tokenizer_config.json, tokenizer.json, Additional files like vocab.json, merges.txt (for models using byte-pair encoding, like GPT models).

What if you use the original tokenizer.json format instead of the serialized one generated from the PrTrainedTokenizerFast?

If you use the original tokenizer.json format (the one before it is re-serialized by PreTrainedTokenizerFast) instead of the re-serialized format, the tokenizer should generally still work as long as it is compatible with the model and the original tokenizer.json contains all the required information. However, there are important considerations regarding performance, compatibility, and behavior when using the original versus the re-serialized format generated by PreTrainedTokenizerFast.

The original format is usually tailored to the tokenizer's original training environment and may not be optimized for modern use cases that involve fast tokenization (e.g., batch processing, GPU acceleration).

10) TOKENIZER_CONFIG_PATH

```shell
--tokenizer-config-path <TOKENIZER_CONFIG_PATH>
    [env: TOKENIZER_CONFIG_PATH=]
```

The --tokenizer-config-path argument in Hugging Face Text Generation Inference (TGI) specifies the path to the tokenizer configuration file (tokenizer_config.json) that TGI should use when setting up the tokenizer for the model. This file contains important metadata and settings about how the tokenizer should function and how it interacts with the model during inference.

Purpose of --tokenizer-config-path

- The --tokenizer-config-path argument allows you to explicitly specify the location of the tokenizer_config.json file, which may be either a local path or a remote path (if stored in the model repository).
- It ensures that TGI uses the correct tokenizer settings that match the model’s requirements, even if the tokenizer configuration file is not stored directly alongside the model weights or if you want to override the default path.

What Is tokenizer_config.json? The tokenizer_config.json file contains information about the tokenizer and its settings, such as:

- Vocabulary Path: Location of the vocabulary files (e.g., vocab.json, merges.txt) that the tokenizer uses.
- Tokenization Strategy: Information on how the tokenizer processes the text, including settings for pre-tokenization, post-processing, normalization, or special token handling.
- Type of Tokenizer: The specific tokenizer class (e.g., ByteLevelBPETokenizer, SentencePieceTokenizer) and any relevant configurations associated with it.
- Special Tokens: Definitions for special tokens like [CLS], [SEP], [PAD], [MASK], and any other tokens used by the model during inference or training.   

11) VALIDATION_WORKERS

```shell
--validation-workers <VALIDATION_WORKERS>
    [env: VALIDATION_WORKERS=] [default: 2]
```

The --validation-workers argument in Hugging Face Text Generation Inference (TGI) specifies the number of worker processes used for validating and preprocessing model inputs (like the tokenizer and model configurations) when TGI is initializing or loading the model. These workers help in parallelizing and speeding up the validation tasks to ensure that the model and its associated components (e.g., tokenizer) are correctly configured before serving inference requests.

12) JSON_OUTPUT

```shell
--json-output
    [env: JSON_OUTPUT=]
```

The --json-output argument in Hugging Face Text Generation Inference (TGI) specifies that the output format of the server responses should be in JSON. This setting is useful when you want the inference results (e.g., generated text or model predictions) and related metadata to be returned in a structured, machine-readable JSON format.

**JSON output can include not only the generated text but also additional information like token probabilities, sequence lengths, latency, or any other metadata associated with the inference request.**

Enhanced Debugging and Analysis: The JSON output can include additional fields that help developers understand the model’s behavior and performance. This might include details such as:

- Generated tokens: The tokens generated by the model, including their respective probabilities or scores.
- Sequence lengths: The lengths of the generated sequences or inputs.
- Timing and latency: Information on how long each step of the process took, which can be helpful for performance monitoring.

Example Usage:

```shell
text-generation-launcher --model-id bigscience/bloom --json-output
```

In this example, the TGI server is instructed to return all inference results in JSON format for the model bigscience/bloom. When a client makes a request, the response will be structured in a JSON object containing all relevant information. When --json-output is enabled, a typical response might look like:

```shell
{
    "generated_text": "The quick brown fox jumps over the lazy dog.",
    "metadata": {
        "tokens": [
            {"token": "The", "id": 1001, "probability": 0.95},
            {"token": "quick", "id": 1002, "probability": 0.89},
            ...
        ],
        "sequence_length": 9,
        "latency_ms": 120,
        "model": "bigscience/bloom"
    }
}
```

13) MAX_CLIENT_BATCH_SIZE

```shell
--max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
    [env: MAX_CLIENT_BATCH_SIZE=] [default: 4]    
```

**Do not conflate MAX_CLIENT_BATCH_SIZE with MAX_BATCH_SIZE. MAX_BATCH_SIZE in Hugging Face Text Generation Inference (TGI) specifies the maximum number of requests that can be processed together in a single batch. MAX_CLIENT_BATCH_SIZE has to do with input queries**.

Difference Between Requests and Queries:

- **Request: A request is an API call made by a client to the TGI server. It can contain one or multiple queries (input texts) bundled together. Each request can be thought of as a “container” that holds one or more input queries**.
- **Query: A query is an individual input text or a single inference task that the model processes. For example, a query could be a single sentence or a paragraph that needs to be processed by the model to generate a response**.

What Does --max-client-batch-size Control?

**The --max-client-batch-size argument specifically controls the maximum number of queries that a single client can include within a single request to the TGI server. It does not refer to the number of requests a client can send, but rather the number of queries (input texts) in each request**.

How --max-client-batch-size Works

- When a client sends a request to the TGI server, they can bundle multiple queries together. For instance, instead of sending each piece of text individually (as separate requests), they might package them together in a single API call for efficiency.
- The --max-client-batch-size parameter sets a limit on how many of these queries can be combined within a single API call. If the number of queries in a request exceeds this limit, the server will either reject the request or only process up to the maximum allowed number of queries.

Example:

```shell
text-generation-launcher --model-id bigscience/bloom --max-client-batch-size 4
```

This means that each API call made by a client can include up to 4 queries (input texts). If a client sends a request with 6 queries, the server will enforce the limit of 4 and may respond with an error or process only the first 4 queries.

**How does the TGI distinguish one query from another if a query can be one sentence or multiple paragraphs**?

In Hugging Face Text Generation Inference (TGI), **the distinction between multiple queries in a single request is based on how the client structures and formats the request**. TGI expects the input to follow a specific structure (often JSON) that clearly separates one query from another. Here's how TGI distinguishes queries:

- TGI uses structured input formats, such as JSON arrays, to distinguish multiple queries in a single API call. The client sends the queries as individual elements in a JSON array, with each element representing a distinct query.
- This structured format ensures that TGI can parse and identify each query separately, regardless of whether the query is a single sentence or multiple paragraphs.

A typical request containing multiple queries might look like this:

```json
{
    "inputs": [
        "What is the capital of France?",
        "Translate the following text into Spanish: The quick brown fox jumps over the lazy dog.",
        "Write a summary of the first two chapters of 'Moby Dick'.",
        "Generate a paragraph describing the weather in New York City today."
    ]
}
```

In this example, the "inputs" key contains an array where each element is a separate query. The structure allows TGI to recognize and process each entry as an independent query, even if some are short questions and others are long paragraphs. The client’s responsibility is to format the request correctly, ensuring that each query is clearly separated as an individual element in the array.

Tokenization and Processing:

- Once TGI receives the structured input, it tokenizes each query separately based on the tokenizer associated with the model.
- It processes each query independently, ensuring that the generated outputs correspond directly to each query in the input array.

Response Structure:

If the input was structured as an array of four queries, the output would also be an array of four generated responses:

```json
{
    "outputs": [
        "The capital of France is Paris.",
        "El zorro marrón rápido salta sobre el perro perezoso.",
        "In the first two chapters of 'Moby Dick', Ishmael, the narrator, introduces himself...",
        "The weather in New York City today is sunny with a few clouds..."
    ]
}
```

In the event streaming is enabled, how is the client supposed to distinguish multiple streams which contain common phrases like "the" "person" etc? When streaming multiple responses for queries in a batch in Hugging Face Text Generation Inference (TGI), the server typically includes metadata or tags alongside the streamed content so that the client can correctly distinguish which tokens belong to which query. This way, the client can identify and separate the output streams for each query even when they contain common phrases or similar tokens. Here's how this process works in practice:

The server usually attaches metadata to each segment of the stream to indicate which query the tokens belong to. This metadata may include:

- Query ID or Sequence ID: A unique identifier for each query in the batch, allowing the client to match the tokens to the correct query.
- Token Position: Information about the position of the token within the sequence for easier reconstruction of the complete response.

The server sends this metadata alongside the generated token in each update, ensuring that the client knows exactly which query each token is associated with.

The streaming data typically comes in a structured format like JSON, where each update contains both the token and the associated metadata. An example might look like:

```json
{
    "query_id": 1,
    "token": "The",
    "position": 0
}

{
    "query_id": 2,
    "token": "El",
    "position": 0
}
```

14) MODEL_ID

```shell
--model-id <MODEL_ID>  

    [env: MODEL_ID=]
    [default: bigscience/bloom-560m]
```

The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `gpt2` or `OpenAssistant/oasst-sft-1-pythia-12b`. Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of transformers

15) REVISION

```shell
      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`
          
          [env: REVISION=]
```

**This parameter can actually be important. For example, in meta-llama/Llama-3.2-90B-Vision-Instruct, on commit 13: https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct/commit/cd1f492d2960d1f4172b589e11a0d212d04c02c5, the author updated the tokenizer_config.json file. Imagine deploying an older version and not having that update. It could impact the quality of results when performing conversational AI**.

15) SHARDED

```shell
      --sharded <SHARDED>
          Whether to shard the model across multiple GPUs By default text-generation-inference will use all available GPUs to run the model. Setting it to `false` deactivates `num_shard`
          
          [env: SHARDED=]
          [possible values: true, false]
```

16) NUM_SHARD

```shell
      --num-shard <NUM_SHARD>
          The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance
          
          [env: NUM_SHARD=]
```

The --sharded parameter specifies whether to shard the model across multiple GPUs. By default text-generation-inference will use all available GPUs to run the model.  Setting it to `false` deactivates `num_shard`. So you should keep this to true, which is the default. 

The --num-shard parameter allows you to specify the number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 text-generation-launcher... --num_shard 2` and `CUDA_VISIBLE_DEVICES=2,3 text-generation-launcher... --num_shard 2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.

17) QUANTIZE

This option specifies the quantization method to use for the model. It is not necessary to specify this option for pre-quantized models, since the quantization method is read from the model configuration. Marlin kernels will be used automatically for GPTQ/AWQ models. Possible values:

- awq:  4 bit quantization. Requires a specific AWQ quantized model: <https://hf.co/models?search=awq>. In here, you will find the popular hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 !
- eetq: 8 bit quantization, doesn't require specific model. Should be a drop-in replacement to bitsandbytes with much better performance. Kernels are from <https://github.com/NetEase-FuXi/EETQ.git>
- gptq: 4 bit quantization. Requires a specific GTPQ quantized model: <https://hf.co/models?search=gptq>. text-generation-inference will use exllama (faster) kernels wherever possible, and use triton kernel (wider support) when it's not. AWQ has faster kernels
- marlin: 4 bit quantization. Requires a specific Marlin quantized model: <https://hf.co/models?search=marlin>
- bitsandbytes: Bitsandbytes 8bit. Can be applied on any model, will cut the memory requirement in half, but it is known that the model will be much slower to run than the native f16
- bitsandbytes-nf4: Bitsandbytes 4bit. Can be applied on any model, will cut the memory requirement by 4x, but it is known that the model will be much slower to run than the native f16

Given that you're running Meta Llama 3.1 70B Instruct on a system with 4 A10G GPUs, each with 24 GiB of memory, and need to optimize memory usage while maintaining performance, here’s how the available quantization options fit into your scenario:

- Total GPU memory: 96 GiB when sharding across 4 GPUs.
- Model size: 70 billion parameters, which is very large and requires significant memory.
- Performance: You want good inference speed but also need to manage memory efficiently.

GPTQ (4-bit):

- Advantages: 4-bit quantization that significantly reduces memory usage, but only works with models already quantized for GPTQ.
- Why it fits: GPTQ is known for its balance between memory savings and performance. It uses exllama kernels for faster inference wherever possible. For a large model like Llama 3.1 70B, GPTQ’s 4-bit quantization can help fit the model within your 96 GiB memory limit, while maintaining relatively fast performance.
- Drawback: Requires a GPTQ-quantized model. You need to ensure the model is available in GPTQ format from Hugging Face.

AWQ (4-bit):

- Advantages: AWQ also offers 4-bit quantization but tends to have faster kernels compared to GPTQ. It provides good memory savings and can be a strong choice for multi-GPU setups like yours.
- Why it fits: AWQ quantization is an excellent choice if you're looking for a balance between memory efficiency and speed, especially with the faster kernels. With 4 GPUs and a 4-bit model, you should be able to load the 70B model comfortably.
- Drawback: Like GPTQ, AWQ requires a specific AWQ-quantized model.

Bitsandbytes (8-bit or 4-bit):

- Advantages: Can be applied to any model, offering memory reduction by half (8-bit) or four times (4-bit).
- Why it fits: While Bitsandbytes is more flexible and can be applied to any model, it is slower than native FP16. If memory is a critical concern, but speed is less important, you could use the 8-bit or 4-bit options.
- Drawback: Significantly slower than AWQ or GPTQ. It might not be ideal if you're looking for optimal inference speed.

Best Option: AWQ (4-bit) if the model is available in AWQ format. It offers faster kernels and the memory reduction you need to fit the model across your GPUs.

Alternative: GPTQ (4-bit) if you prefer faster inference but still want to reduce memory.

Fallback: Bitsandbytes (8-bit or 4-bit) if you're unable to find pre-quantized models, though it will be slower.

Using a prebuilt quantized model:

```shell
model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
token=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
volume=$PWD/data
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $model --max-batch-prefill-tokens 4800 --max-input-length 4768 --max-total-tokens 10960 --num-shard 4
```
