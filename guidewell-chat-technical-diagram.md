# GuideWell Chat: The Definitive Guide: Part I

Guiding Principal:
Making GuideWell Chat as capable and productive as ChatGPT 4o, using techniques that ChatGPT cannot use:
- applying classical and regression statistics on internal Florida Blue data
- using machine learning math, testing it against Florida Blue user prompts and fine-tuning with a full automated suite of unit tests
- proprietary data lake (not yet implemented)
- Pulling together the above in building algorithms that can be generically fed to third-party Large Language Models, like Meta Llama, to generate more probable tokens that fit the expected value of florida blue consumers

GuideWell Chat Core packages written for Langchain framework:
1) GuideWellChat: Chat Bot
    - multi-agent chat bot
    - message history
    - context aware Q&A
    - RAG
    - tool calling (supporting multiple model validation)
    - image and text inputs
    - token information
    - human in the loop (coming soon)
2) GuidWellChat: HuggingFace
    - HuggingFace Inference Client
    - HuggingFace Embeddings
    - HuggingFace LLM
    - HuggingFace Chat Model
    - Transformer Tokenizer Proxy
3) GuideWellChat: Document Loader
    - Extended PyPDF Loader
    - PowerPoint Loader
    - Word Loader
    - Extended PDF Parser
    - Base64 Blob Parser
4) GuideWellChat: Ingestion Pipeline
    - Lazy PDF Ingestor
    - Lazy Word Ingestor
    - Lazy PowerPoint Ingestor
    - Lazy Text Ingestor
5) GuideWellChat: Retrievers
    - Streaming Parent Document Retriever
6) GuideWellChat: Text Splitters
    - Mixed Content Text Splitter (multi-modal text splitter)
    - Streaming Text Splitter (streams chunks)
7) GuideWellChat: Vector Store
    - redis
        - MultiModal Vector Store (multi-modal support)
        - Docstore (storage for document inheritance)
8) GuideWellChat API Server (fastapi)
10) GuideWell UI (react)
11) GuideWell Continuous Integration (gitlab cicd)

GuideWell Chat packages will be migrated to Gen AI Libs
Release date is Friday 18, 2025

# The NEW GuideWell Chat Document Lookup Algorithm
Released into OpenShift test environment Monday April 14, 2025

## Embedding Strategy

### 1. Input Assets

Supported formats:
- pdf files
- word files
- powerpoint files
- plain text files
- image files (multiple formats supported)

Each asset is composed of text, images, and metadata parsed by the Document Loader and Document Parser. Metadata is hierarchical. Note image extraction support exists for PDF documents. Support for Word and Powerpoint coming soon.

Inheritable Metadata:
- uuid (racf)
- conversation id
- source
- page number (not always applicable to word)
- total pages
- chunk index (ordinal weight for embedding)
- cosine distance score (additional similarity weight for embedding)
- reranking score (arithmetic sum of cosine distance scores for child documents per parent)
- multimodal type (e.g. text chunk, image, etc)
- n-gram weights (part of embedding/partially implemented)
    - calculates probability weights for the population mean of all fonts in document using a scaling factor
    - biword and triword weights using xml xpath parsing and NLTK Natural Langauge Processing package
    - for n-grams marked as important, embeds hint injections as part of the overal embeddings.
        - hint injections improve multi-head self‑attention scores, where attention heads focus on specified tokens during the forward pass of neural network
        - hint injections include "importance" symbols, <importance> xml tags, becoming a strong hint via learned weights
        - repeated n-gram phrases for emphasis, where tokens in embedding vectors appear multiple times in a sequence
        - Actual Test Results: In semantic retrieval of 170 documents, without the hint injections some chunks ranked 10th or 12th in the result. With hint injections, they ranking moved all the way up to 1st. 
- semantic size threshold (e.g. image dimensions, color channels such as RGB/grayscale) (unpersisted/images only)

Diagram of the efficacy of n-gram weights part of the text splitting phase of the ingestion process:

````
| **ASCII Diagram** |
|:-----------------:|
| ```
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Embedding + Multi-Head Self-Attention in Each Transformer Layer                             │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
``` |
| ```
       ↓                 ↓             ↓             ↓                         ↓       ↓
       ↓                 ↓             ↓             ↓                         ↓       ↓
``` |
| ```
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │<important>  │   │ Mandatory   │   │  Teams      │   │</important> │   │   Repeated   │
   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
``` |
| ```
          ↕                ↕                 ↕               ↕                 ↕
          ↕                ↕    (Heads can learn to          ↕                 ↕
          ↕                ↕    attend strongly to            ↕                 ↕
          ↕                ↕    repeated & special tokens)    ↕                 ↕
          ↕                ↕               ↕                  ↕                 ↕
``` |
| ```
   ┌───────────────────────────────────────────────────────────────────────────────────────────┐
   │ Multi-Head Attention Heads: each token queries/keys/values across entire sequence       │
   └───────────────────────────────────────────────────────────────────────────────────────────┘
``` |

````

PDF Metadata:
- uses inheritable metadata

Word Metadata:
- uses inheritable metadata

PowerPoint Metadata:
- uses inheritable metadata

Plain Text Metadata
- uses some of inheritable metadata

### 2. Chunking Strategy

All documents by default use the Parent-Child Chunking strategy. Parent chunks and child chunks are stored in the same data store (Redis) for low latency. Child chunks embody the actual embedding vectors along with other metadata. Child chunks are stored in a RediSearch index. 

Index Information: 
| Field              | Value              |
|--------------------|--------------------|
| **index_name**     | user_conversations |
| **distance_metric**| COSINE             |
| **indexing_algorithm** | FLAT           |
| **vector_datatype**| FLOAT32            |
| **storage_type**   | hash               |


Child Index Fields:
| Field              | Value              |
|--------------------|--------------------|
| **text**           | text               |
| **embedding**      | vector             |
| **uuid**           | tag                |
| **conversation_id** | tag               |
| **source**         | tag                |
| **doc_id**         | tag                |

Key points:
- Note the `text` fields are used for full‑text indexing and the `tag` fields are used for exact matches. The text chunk represents a text chunk or a path to image data, which eventually will be stored in a data lake (although as of now, it is stored in Redis as base64 encoded image).
- Parent chunks are not stored in a RediSearch index. They are key-value pairs comprising of a key and the parent text chunk.
- Parent and child chunks have a TTL of 30 days before they are purged from the in-memory data store. It is not using RDB (Redis Database backup) snapshots or AOF (Append Only File).
- child chunks are a fraction of the size of parent chunks (e.g. text corresponding to 150 tokens vs text corresponding to 2000 tokens). Tokens are defined by the tokenizer used by the underlying embedding model.

#### Chunking Strategy: Hugging Face Transformers and the Adaptive Token Boundary strategy

The text splitter utilizes the tokenizer capability provided by the Transformers package directly when performing chunking. It loads the Tokenizer linked to the specific model card of the specified LLM. There is a python and rust tokenizer version available. It almost always uses the faster rust version called `PreTrainedTokenizerFast`. Importantly, each tokenizer has its own `morphological variations`:

| Word          | Subword Tokens                   | Token IDs       | Embeddings                        |
|---------------|----------------------------------|-----------------|-----------------------------------|
| unbelievable  | ["un", "##believ", "##able"]     | [1024, 5821, 9082] | Each ID -> a learned vector, e.g. 768D |
| impossible    | ["im", "##possibl", "##e"]       | [2171, 7762, 4021] | Each ID -> a learned vector, e.g. 768D |
| reading       | ["read", "##ing"]                | [1379, 5432]       | Each ID -> a learned vector, e.g. 768D |
| cats          | ["cat", "##s"]                   | [3741, 215]        | Each ID -> a learned vector, e.g. 768D |
| astonishingly | ["a", "##stonish", "##ing", "##ly"] | [31, 6821, 5432, 9990] | Each ID -> a learned vector, e.g. 768D |

In Transformer-based language models, each token ID in the vocabulary maps to a high-dimensional embedding vector. We are currently using a transformer embedding model called vlm2vec-full produced by TIGER Labs, which has a 3072 dimensions as expressed in the config file of their model card (e.g. "hidden_size": 3072 ). These vectors are stored in the token embedding table, an internal parameter matrix that the model learns during training.  

The transformers package exposes `offset mappings`. The "offset mappings" refer to the start and end character positions in the original text for each token produced by a Fast tokenizer. So for each token id, there is a 2‑tuple (start_char, end_char) indicating which substring of the original text corresponds to that token.

Our `Adaptive Token Boundary` Text Splitter strategy leverages the offset mappings to implement a customized token chunking strategy that beats langchain token text splitting in benchmarks:
- rather than tokenizing per chunk, we tokenize once during initialization
- we chunk tokens rather than text using transformer tokenizer offset mappings as a reference to discover word boundaries
- we store pending chunks to ensure overlap when chunks are too small
- tokens are streamed, as opposed to fully loaded in memory

When loading the tokenizer, we also cache certain information that corresponds to the way the model was loaded on the inference platform, whether it be TGI or vLLM (e.g. max batch tokens, max model length) 

### 3. Embeddings

Parent and Child text and image chunks are embedded into Redis vector store. In a previous section, we outlined the metadata and schema that is associated with documents stored in the vector store. For efficiency, and depending on the batch parameters configured on the embeddings model, we send multiple input sequences at once to be embedded in a single forward pass of the neural network.

## Retrieval Strategy

## 1. Chat Model

Our Chat Bot is built on top of our Chat Model component which implements the Langchain Runnable interface. It supports the following:
- extends a light weight wrapper around the HuggingFace-based Inference Client, with support for HuggingFace TGI and vLLM, namely chat completion endpoint
- handles conversation-style messages (system/user/assistant/etc.) and returns a chat-style response back from the inference client
- supports synchronous and asynchronous outputs and streaming outputs, as well as batching
- handles stop sequences and post-processing, including prompt token counts, completion token counts, total token counts, log probabilities both in streaming and non-streaming scenarios
- supports tool calling
- allows for multimodal inputs

| **Step**                                                     | **What Actually Happens**                                                                                                                                                                                                                                                                                                                          | **Input → Output**                                                                                                                                                                                                                                                                                                                                                      |
|:------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. User Prompt**<br/>(HumanMessage)                        | A **HumanMessage** arrives (e.g., “What is the capital of France?”). The `HuggingFaceChatModel` also includes a **SystemMessage** (e.g., “You are a helpful assistant”), and possibly other `BaseMessages` from chat history or relevant context.                                                           | **Input:** 1 or more LangChain `BaseMessage`s (system, user, historical, etc.)<br/><br/>**Output:** A combined list of messages representing the entire conversation context (system + user + others).                                                                                                                           |
| **2. Structured Chat Format**                                | The `HuggingFaceChatModel` passes these messages to the **HuggingFace Inference Client** in a “structured” format (similar to OpenAI style). For instance, `{"role": "system", "content": ...}`, `{"role": "user", "content": ...}`, etc. This is accepted by TGI, TEI, or vLLM endpoints.                                                     | **Input:** Merged `BaseMessage`s from step 1.<br/><br/>**Output:** A structured list of chat dicts: e.g., `[{"role": "system", "content": ...}, {"role": "user","content": ...}, ...]` that the inference client can send to TGI/TEI/vLLM.                                                                                         |
| **3. Model Inference**<br/>(TGI / TEI / vLLM)                | The underlying **HuggingFaceLLM** calls the huggingface hub client. That client uses TGI, TEI, or vLLM to run the forward pass. The model returns either a single final text or a stream of partial text chunks. Synchronous or asynchronous modes are supported.                                                                                 | **Input:** The chat messages + model parameters (e.g. `max_tokens`, `temperature`, etc.)<br/><br/>**Output:** Raw model-generated text (or partial chunks for streaming).                                                                                                                                                                                              |
| **4. Post-processing**                                       | Once the raw or partial text arrives, the chat model can: <br/>• **Strip** anything after stop sequences <br/>• Merge partial chunks if streaming <br/>• Summarize **token usage** (counts, etc.) so you know how many tokens were used in generation.                                                                                          | **Input:** Raw text or chunked text from TGI/TEI/vLLM<br/><br/>**Output:** Cleaned text, plus usage stats (token count, model name, etc.)                                                                                                                                                                                                                               |
| **5. Build `ChatResult`**                                    | Lastly, the `HuggingFaceChatModel` packages the final assistant text into one or more **`ChatGeneration`** objects (which each hold text + `generation_info`), and returns a **`ChatResult`**. The user’s application can display or further process the assistant’s message.                                                                    | **Input:** Cleaned text + usage info<br/><br/>**Output:** A `ChatResult` containing `ChatGeneration(s)` (the assistant text) plus a `generation_info` dictionary with usage details. This is what the user’s system sees as the final chat completion.                                                                                                                     |

## 2. Chat Bot

The Chat Bot is a multi‑agent reasoning flow that combines moderation (e.g., Llama Guard) and quality loops to keep agents aligned and safe. It supports long‑term conversation via persisted context (Mongo or Redis for storing chat history and vector data respectively), so each session can build on past interactions. With a pool of LLMs available, the system can reason about the best model or approach to use—be it single‑doc retrieval, multi‑doc context merging, or purely pretrained knowledge. In addition, there will be future support to handle human‑in‑the‑loop oversight, allowing a person to approve or steer actions. Overall, it empowers flexible and reliable conversation flows where doc retrieval, tool calling, and dynamic routing coalesce in one robust pipeline.

| **Node**                          | **Description / Purpose**                                                                                                                                                                                                                                                                                                                                                   | **Transitions**                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **START**                         | The entry point into this multi-agent flow.                                                                                                                                                                                                                                                                                                                                 | → **guardrails** (unconditional).                                                                                                                                                                                                                                                                                                                                                |
| **guardrails**                    | Invokes a safety model (e.g. Llama Guard) to check the user’s input. If it’s over the token limit or unsafe content, it branches accordingly.                                                                                                                                                                                                                                                                          | • If user input is too large: → **exceeded_token_budget** → END <br/> • If user input is unsafe: → **not_safe** → END <br/> • Otherwise: → **prefill_system_prompt**                                                                                                                                                                                                                                                     |
| **exceeded_token_budget**         | Final node if the user’s request can’t be processed due to token budget issues. Produces a short message about it, then ends.                                                                                                                                                                                                                                                                                        | → **END** (No further transitions.)                                                                                                                                                                                                                                                                                                                                              |
| **not_safe**                      | Final node if the content is flagged as unsafe by moderation. The system returns a rejection message.                                                                                                                                                                                                                                                                                                               | → **END**                                                                                                                                                                                                                                                                                                                                                                        |
| **prefill_system_prompt**         | Takes the existing conversation context (including any images) and updates the system prompt with additional instructions, appending to the system prompt if necessary.                                                                                                                                                                                                                                              | → **route_query** (No branching logic here – it always moves forward.)                                                                                                                                                                                                                                                                                                           |
| **route_query**                   | A reasoning step that looks at user’s prompt + any context stored in vector store. If context found in vector store for user prompt, it decides: <br/>- If **1 doc** is relevant (or user uploaded a single file), → **single_doc_prompt** <br/>- If **multiple** docs or multiple user-uploaded files are relevant, → **multi_doc_prompt** <br/>- If no doc context is found or needed, → **pretrained_corpus_prompt**.                              | → **single_doc_prompt**, **multi_doc_prompt**, or **pretrained_corpus_prompt** depending on logic.                                                                                                                                                                                                                                                                              |
| **single_doc_prompt**             | Builds a **single doc context** chain: <br/>1) merges chat history from Mongo (via `generate_with_history`). <br/>2) calls `create_context_aware_chain` to retrieve & combine relevant text from that **one** doc. <br/>3) The chain is invoked, returning final answers.                                                                                                                                              | → **END** (completes after returning a final assistant message). Potentially could branch to a “retry model” if the response is invalid.                                                                                                                                                                                                                                                                               |
| **multi_doc_prompt**              | Builds a **multi-doc chain**: <br/>1) merges chat history from Mongo (via `generate_with_history`). <br/>2) calls `create_multicontext_aware_chain` – each doc’s retrieval is done in parallel, results get merged into a single answer. <br/>3) Returns final assistant text.                                                                                                                                           | → **END** (completes with final multi-doc answer). Potentially might call a “retry model.”                                                                                                                                                                                                                                                                                                                             |
| **pretrained_corpus_prompt**      | If user’s question is purely general knowledge (no docs or user files relevant), it uses a **pretrained corpus** approach: <br/>1) merges chat history from Mongo (via `generate_with_history`). <br/>2) calls `create_generic_chain`, which just uses the base LLM. <br/>3) Yields final output.                                                                                                                    | → **END** (completes with a final message). Could also do a fallback “retry model” if the answer is inadequate.                                                                                                                                                                                                                                                                                                        |
| **Chat History Integration**      | In each doc-prompt node (`single_doc_prompt`, `multi_doc_prompt`, `pretrained_corpus_prompt`), chat history is accounted for, hooking into Mongo. The entire prior conversation influences the final answer.                                                                                                                                                                                                         | *(Not a separate node; it’s a mechanism that wraps each chain node.)*                                                                                                                                                                                                                                                                                                                                                  |
| **Retriever Capability**          | - `create_context_aware_chain` (used by **single_doc_prompt**) performs retrieval from **one** relevant doc. <br/>- `create_multicontext_aware_chain` (used by **multi_doc_prompt**) does parallel retrieval from **multiple** docs, merging them. <br/>- For **pretrained_corpus_prompt**, retrieval is bypassed.                                                                                             | *(Not a separate node; it’s the retrieval logic inside each chain node.)*                                                                                                                                                                                                                                                                                                                                               |
| **END**                           | Flow terminates. The final assistant message is returned to the user, or an error if tokens are exceeded or content is not safe.                                                                                                                                                                                                                                                                                      | *(No transitions.)*                                                                                                                                                                                                                                                                                                                                                             |


## 3. Retriever

LLM models can vary in context windows, but over time, the context windows tend to grow at exponential rates. The solution is to provide a method to organically adjust to context growth for a whole field of LLMs, with different context, without manual intervention. We don't want to go in and modify some k parameters each time a new model appears or to handle different parameters for different models. At the same time, we want a greedy k, where we maximize token availability by retrieving as much context as possible without hitting limits. This is the governing principle for some of the subsequent mathematical decisions.

Multiple formulas used in machine learning were entertained. The decision is to assume a Sigmoid approach, specifically logistic. Compared to functions like a logarithm or square root, the sigmoid provides a more tunable mid-region: we pick the slope "a" to control how sharply we move from low retrieval to high retrieval, and choose a midpoint "m" where we cross half the maximum. We define a final cap $k_{\text{max}}$  to prevent fetching an absurd number of documents. This mirrors how sigmoids are historically used in logistic regression or as an activation function in neural networks, where it smoothly transitions from near 0 to near 1. Here, we're leveraging exactly that property to ensure you have a smooth growth in the number of documents to fetch—enough so that you maximize context usage without drastically overloading the model at any point.

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

Specifically, we use the logistic function (in the family of Sigmoids):

\[
k(x) \;=\; \frac{L}{\,1 \;+\; e^{-\,a\,(\,x - m\,)}\,}
\]
- 𝑘(𝑥) = number of documents to return in a single pass as a function of 𝑥
- 𝑥 = context window size (e.g. 30k, 130k).
- $L$ = upper bound or saturation (e.g. 120).
    - 𝐿 is the maximum value of 𝑘 our system is willing to handle in a pass
    - For example, 1000 for k will cause system disruption
- 𝑎 = slope > 0. Larger 𝑎 → sharper jump.
- 𝑚 = midpoint in 𝑥. At 𝑥 = 𝑚, $f(x) = \frac{L}{2}$ 

Intuition:
- $f(30,000) ≈ 50$
- $f(120,000) ≈ 110$
- $f(125,000) ≈ 115$
- $f(130,000) ≈ 120$

Note the values represent the returned child documents. The actual parent documents that are used in a forward pass to the LLM are a fraction of the child documents.

### Retriever: The Multiple Retriever Solution

The user has the option to upload as many documents as they want, let's call $n$. The expectation is the system should not become overloaded and break. 

The solution is to penalize all retrievers as fairly as possible but of course to get as close to $x$. In effect, some retrievers will pull more documents than others. The approach taken is the hamilton method:

\[ 
k_i = \mathrm{round}\Bigl(\mathrm{Sigmoid.logistic}(x_i, L, a, m)\Bigr)
\]
$k_i =$ original value
$k_i' =$ updated value

\[
k_{\text{sum}} = \sum_{i=1}^{N} k_i
\]

\[
\text{If } \sum k_i \,\le\, K_{\max}, 
\quad \text{then define } k_i' = k_i.
\]

\[
\text{If } k_{\text{sum}} > K_{\max}, 
\quad \text{then let } \alpha = \frac{K_{\max}}{\sum k_i}
\quad 
\]

\[
k_i' = \alpha \times k_i
\]

\[
\text{base\_sum} 
= \sum_{i=1}^{N} \lfloor k_i' \rfloor.
\]

