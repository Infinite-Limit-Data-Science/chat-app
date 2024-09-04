### HuggingFace Text Generation Inference: Main Features

Source: https://github.com/huggingface/text-generation-inference

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). You have the option to use Rust or Python as the codebase. TGI enables high-performance text generation for the most popular open-source LLMs, including Llama. Key features include Tensor Parallelism. Tensor Parallelism is a technique used in deep learning to parallelize the computation of large tensors across multiple GPUs.

The TGI also features Token Streaming, a technique used in NLP to process large amounts of text data in a streaming fashion. In traditional NLP models, the input text is typically processed in batches, where the entire batch is processed at once. However, this can be inefficient for large amounts of text data, as it requires a significant amount of memory and computation. Token Streaming addresses this issue by processing the input text in a streaming fashion, where the text is processed one token at a time. This allows for more efficient use of memory and computation, and enables the processing of large amounts of text data in real-time.

The TGI also performs continuous batching of incoming requests to improve the throughput by grouping multiple incoming requests together and processing them as a single batch. In traditional systems, each incoming request is processed individually.

Processing NLP tasks, such as text generation, is computationally expensive and memory-intensive. The TGI uses Flash Attention, a technique that reduces the computational cost of attention mechanisms in transformers by using a more efficient algorithm. It also leverages Paged Attention, a technique that reduces the memory requirements of transformers by processing the input data in smaller chunks, rather than all at once.

The TGI uses various techniques for quantization. This reduces the precision of the model's weights and activations from high-precision values to lower-precision ones, significantly reducing the memory requirements and computational cost of the model.

The TGI also supports Safetensors weight loading, which provides a safe and efficient way to load and store model weights, which can be especially important for large models.

Importantly, in the context of Text Generation Inference (TGI), the TGI supports a Stop Sequence, which is a special sequence of tokens that can be used to indicate the end of a generated text sequence. In other words, a Stop Sequence is a way to tell the model to stop generating text, usually because the desired output has been reached or because the model has reached a certain threshold of confidence. The TGI supports this as well.

The HuggingFace TGI has many other features, but the top ones are highlighted above.

### Setup Infrastructure

The easiest way of deploying the HuggingFace TGI is using the official Docker image. We will use the Deep Learning AMI GPU TensorFlow 2.12.0 (Ubuntu 20.04) 20230324 AMI as the base. This AMI supports the following EC2 instances: G3, P3, P3dn, P4d, G5, G4dn. Note the g4dn.2xlarge EC2 Instance is built on the NVIDIA Tesla T4 GPU, which was released in 2018 and build on the Turing architecture. Pricing for the g4dn.2xlarge Instance Type is available here: https://aws-pricing.com/g4dn.2xlarge.html. However, recent versions of the HuggingFace TGI will not run on this older GPI. You will be greeted with an error like this:

```bash
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
2024-09-03T00:45:16.041862Z ERROR warmup{max_input_length=4095 max_prefill_tokens=4145 max_total_tokens=4096 max_batch_size=None}:warmup: text_generation_client: router/client/src/lib.rs:46: Server error: CANCELLED
Error: WebServer(Warmup(Generation("CANCELLED")))
```

The best bet is to follow the suggestion of the error message and use a more recent architecture, such as the Ampere architecture. Hence, we will launch  Meta LLama 3.1 7B LLM on a g5.2xlarge, which uses the NVIDIA A10G Tensor Core GPU powered by the Ampere architecture. Follow the link to get a description of this Instance Type: https://aws.amazon.com/ec2/instance-types/g5/.

Furthermore, the Deep Learning AMI GPU TensorFlow 2.12.0 (Ubuntu 20.04) 20230324 AMI includes TensorFlow 2.12.0, which is compatible with the Meta Llama 3.1 7B model, for example. Here are more specs:

- TensorFlow Version: 2.12.0
- CUDA Version: 11.2
- cuDNN Version: 8.1.0
- Ubuntu Version: 20.04

In the EC2 Console, once selecting the AMI, choose the g5.2xlarge Instance Type. Configure VPC settings as appropriate. Use 512GB GP3 EBS Volume. For Purchasing Option, select "Spot Instances" to potentially save a lot in cost, sometimes 90 percent savings.

### Select the Model

HuggingFace has a Model Hub, boasting pre-trained models for NLP tasks, such as language translation, sentiment analysis, and text clessification. The models on HuggingFace Model Hub are open-source models and, thus, available for download. Visit the HuggingFace landing page at https://huggingface.co and click on "Models". Notice the huge selection of open-source models. On the left side are categories to filter your search. The first category is the Task category. The Task category contains NLP tasks (e.g. Q&A and Text Classification), Image Classification for Computer Vision, Automatic Speech Recognition for Speech. The second category is the Libraries category. The three most popular categories are PyTorch, TensorFlow, and JAX. Finally, the Libraries category is used to specify which high-level framework to specify for models, such as Transformers. Hugging Face provides a popular open-source library called Transformers, which allows users to easily implement and fine-tune transformer-based models, such as BERT, RoBERTa, and XLNet, for a wide range of NLP tasks. The Datasets category allows you to filter models that are trained on specific datasets. The next category is the Languages category, which allows you to specify models trained on certain languages, such as English. The last category allows you to choose the License the model has shared.

In the Models search bar, search for "meta-llama-3.1-8B-Instruct". Select the result to view the Model Card for meta-llama/Meta-Llama-3.1-8B-Instruct. Notice the list of tags for the model:

- Text Generation: This tag indicates that the model is designed for generating human-like text, either from scratch or by completing a given prompt.

- Transformers: This tag signifies that the model architecture is based on the Transformer model, which is a type of neural network particularly well-suited for natural language processing tasks. Transformers are known for their ability to handle sequential input data, like text, more efficiently and effectively than traditional recurrent neural networks.

- Safetensors: This is a format for storing and loading model weights and other data in a way that's safer and more secure than traditional methods. Safetensors are designed to prevent common issues like data corruption and to ensure the integrity of the model's weights.

- PyTorch: This tag indicates that the model is implemented using the PyTorch deep learning framework. PyTorch is a popular open-source machine learning library for Python, known for its dynamic computation graph and ease of use.

- 8 languages

- llama: This is a reference to the model's name or series, indicating it's part of the LLaMA (Large Language Model Meta AI) family of models developed by Meta AI.

- facebook: This tag is likely a reference to the model's origin or the company that developed it, Meta AI, which is part of Meta Platforms, Inc., the same company that owns and operates Facebook.

- meta: Similar to the "facebook" tag, this likely refers to the model's origin, specifically that it was developed by Meta AI.

- llama-3: This further specifies the model's version or series within the LLaMA family, indicating it's part of the third generation or iteration of LLaMA models.

- conversational: This tag suggests the model is designed or suitable for conversational AI tasks, meaning it can engage in dialogue, respond to questions, or participate in discussions in a way that simulates human conversation.

- text-generation-inference: This tag emphasizes the model's capability for text generation tasks during inference (i.e., when the model is used to make predictions or generate text after it has been trained).

- Inference Endpoints: This tag likely indicates that the model is optimized or intended for use in inference endpoints, which are interfaces through which a trained model can receive input and generate output in a production or deployment setting.

- arxiv: 2204.05149: This is a reference to a specific paper on arXiv, a repository of electronic preprints (known as e-prints) in physics, mathematics, computer science, and related disciplines. The number "2204.05149" is the identifier for the paper that describes the model or its underlying technology.

### Launching TGI as a Docker Container

At this point, you have the infrastructure, with the necessary NVidia GPUs, and you have the model name. You may think you are ready to run the LLM. But you require two more prerequisites. First, you need to generate and use a HuggingFace Token from the HuggingFace Settings > Access Tokens panel. Second, you will need to seek approval before being able to use the Meta Llama family of models. Hence, go to the Model's Model Card, click on "Files and versions", and then request approval. It can take up to a day to be granted approval. Once you have these prerequisites satisfied, connect to the EC2 Instance as the Ubuntu user. The actual parameters to pass to the Docker container depends a lot on your hardware. The `nvidia-smi` command tells you the name of your GPU and how much Cuda memory in use and how much Cuda memory the system has:

```bash
nvidia-smi
Mon Sep  2 23:37:41 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   28C    P8    14W /  70W |      2MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|=============================================================================|
|  No running processes found                                                 |
```

As shown, we have a Tesla T4 GPU with 16 GB of CUDA memory. But as mentioned already, it uses the Turling architecture and when you try running the LLM:

```bash
model=meta-llama/Meta-Llama-3.1-8B-Instruct
token=token-from-huggingface
volume=$PWD/data

docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $model
```

You may run into an ubiquitous error: 

```bash
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU

 rank=0
2024-09-02T23:29:59.823812Z ERROR text_generation_launcher: Shard 0 failed to start
2024-09-02T23:29:59.823837Z  INFO text_generation_launcher: Shutting down shards
Error: ShardCannotStart
```

We clearly knew we were going to run into this issue with the Tesla T4 GPU. However, I wanted to show this error, in the event you get the same error for a newer GPU. You have a few options to resolve this error; each has tradeoffs:

- Adjust Batch Size: adjusting the batch size involves changing the number of input samples that are processed together as a single unit during training or inference. Decreasing the batch size reduces memory requirements, which is can be suitable for LLMs with limited GPU. The negative impact includes decreasing performance by processing fewer samples in parallel and increasing the number of iterations required to process a large dataset

- Quantize: quantizing a model involves reducing the precision of the model's weights and activations from 32-bit floating-point numbers to lower-precision numbers. Quantization can significantly reduce the memory requirements of the model, making it possible to run on devices with limited memory. Quantization can also improve the performance of the model by reducing the number of calculations required. Quantization can lead to faster inference times, making it suitable for real-time applications. However, quantization can lead to a loss of accuracy, as the reduced precision of the weights and activations can affect the model's ability to make predictions.

Here are the options in practice:

```shell
# decrease batch size
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $model --max-batch-size 16

# quantize
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $model --quantize bitsandbytes-nf4
```

Note do not confuse max_batch_size with max_client_batch_size. max_batch_size refers to the maximum number of input samples that can be processed together in a single batch by the model. This is a property of the model itself and determines how many inputs can be processed in parallel. max_client_batch_size, on the other hand, refers to the maximum number of input samples that can be sent by a client in a single request to the model server. This is a property of the client and determines how many inputs can be sent in a single request.

After battling the error above, you may get this error:

```shell
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
2024-09-03T00:45:16.041862Z ERROR warmup{max_input_length=4095 max_prefill_tokens=4145 max_total_tokens=4096 max_batch_size=None}:warmup: text_generation_client: router/client/src/lib.rs:46: Server error: CANCELLED
```

This is because you are using an older NVidia architecture, such as Turing. Turing is an older architecture from 2018. The Ampere architecture was released in 2020 and the newer Hopper architecture was released in 2022. As suggested in this documentation, use a GPU built on the Ampere architecture at least.

A successful launch of the Docker container will yield log outputs as shown below:

```shell
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $model
Unable to find image 'ghcr.io/huggingface/text-generation-inference:2.2.0' locally
2.2.0: Pulling from huggingface/text-generation-inference
aece8493d397: Pull complete
45f7ea5367fe: Pull complete
3d97a47c3c73: Pull complete
12cd4d19752f: Pull complete                                                                                           da5a484f9d74: Pull complete                                                                                           4f4fb700ef54: Pull complete                                                                                           43566b48e5d6: Pull complete
f165933352a8: Pull complete
f166ffc7c7b4: Pull complete
58165ae83a0e: Pull complete
074d930e1b90: Pull complete                                                                                           1033b2636622: Pull complete                                                                                           e0aa534acffe: Pull complete                                                                                           130989d28b48: Pull complete                                                                                           a65ea9ebfaba: Pull complete
7225b2c46f88: Pull complete
43154e73908f: Pull complete                                                                                           8f400e318724: Pull complete
f694acf6c40f: Pull complete
44fc79164bc4: Pull complete
8bc7c142e917: Pull complete
021f7d48bdcb: Pull complete
c9d01f7d10cc: Pull complete
400740bc31be: Pull complete                                                                                           bd4b49ea4512: Pull complete
141228b9bdde: Pull complete
34d4a7457184: Pull complete
66e724dff43a: Pull complete                                                                                           25c75c242d08: Pull complete
6a4be63c7e70: Pull complete
b2d83f4bca52: Pull complete
373c47aa4b50: Pull complete
Digest: sha256:d39d513f13727ffa9b6a4d0e949f36413b944aabc9a236c0aa2986c929906769
Status: Downloaded newer image for ghcr.io/huggingface/text-generation-inference:2.2.0
2024-09-03T04:36:57.491823Z  INFO text_generation_launcher: Args {
    model_id: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    revision: None,
    validation_workers: 2,
    sharded: None,
    num_shard: None,
    quantize: None,
    speculate: None,
    dtype: None,
    trust_remote_code: false,
    max_concurrent_requests: 128,
    max_best_of: 2,
    max_stop_sequences: 4,
    max_top_n_tokens: 5,
    max_input_tokens: None,
    max_input_length: None,
    max_total_tokens: None,
    waiting_served_ratio: 0.3,
    max_batch_prefill_tokens: None,
    max_batch_total_tokens: None,
    max_waiting_tokens: 20,
    max_batch_size: None,
    cuda_graphs: None,
    hostname: "518fcee57dd5",
    port: 80,
    shard_uds_path: "/tmp/text-generation-server",
    master_addr: "localhost",
    master_port: 29500,
    huggingface_hub_cache: Some(
        "/data",
    ),
    weights_cache_override: None,
    disable_custom_kernels: false,
    cuda_memory_fraction: 1.0,
    rope_scaling: None,
    rope_factor: None,
    json_output: false,
    otlp_endpoint: None,
    otlp_service_name: "text-generation-inference.router",
    cors_allow_origin: [],
    watermark_gamma: None,
    watermark_delta: None,
    ngrok: false,
    ngrok_authtoken: None,
    ngrok_edge: None,
    tokenizer_config_path: None,
    disable_grammar_support: false,
    env: false,
    max_client_batch_size: 4,
    lora_adapters: None,
    disable_usage_stats: false,
    disable_crash_reports: false,
}
2024-09-03T04:36:57.491979Z  INFO hf_hub: Token file not found "/root/.cache/huggingface/token"
2024-09-03T04:36:57.507079Z  INFO text_generation_launcher: Default `max_input_tokens` to 4095
2024-09-03T04:36:57.507101Z  INFO text_generation_launcher: Default `max_total_tokens` to 4096
2024-09-03T04:36:57.507104Z  INFO text_generation_launcher: Default `max_batch_prefill_tokens` to 4145
2024-09-03T04:36:57.507106Z  INFO text_generation_launcher: Using default cuda graphs [1, 2, 4, 8, 16, 32]
2024-09-03T04:36:57.507198Z  INFO download: text_generation_launcher: Starting check and download process for meta-llama/Meta-Llama-3.1-8B-Instruct
2024-09-03T04:37:01.632012Z  INFO text_generation_launcher: Download file: model-00001-of-00004.safetensors
2024-09-03T04:37:08.927851Z  INFO text_generation_launcher: Downloaded /data/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/model-00001-of-00004.safetensors in 0:00:07.
2024-09-03T04:37:08.927963Z  INFO text_generation_launcher: Download: [1/4] -- ETA: 0:00:21
2024-09-03T04:37:08.928312Z  INFO text_generation_launcher: Download file: model-00002-of-00004.safetensors
2024-09-03T04:38:31.672929Z  INFO text_generation_launcher: Downloaded /data/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/model-00002-of-00004.safetensors in 0:01:22.
2024-09-03T04:38:31.672967Z  INFO text_generation_launcher: Download: [2/4] -- ETA: 0:01:30
2024-09-03T04:38:31.673299Z  INFO text_generation_launcher: Download file: model-00003-of-00004.safetensors
2024-09-03T04:39:15.116042Z  INFO text_generation_launcher: Downloaded /data/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/model-00003-of-00004.safetensors in 0:00:43.
2024-09-03T04:39:15.116123Z  INFO text_generation_launcher: Download: [3/4] -- ETA: 0:00:44.333333
2024-09-03T04:39:15.116431Z  INFO text_generation_launcher: Download file: model-00004-of-00004.safetensors
2024-09-03T04:39:22.450825Z  INFO text_generation_launcher: Downloaded /data/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/model-00004-of-00004.safetensors in 0:00:07.
2024-09-03T04:39:22.450918Z  INFO text_generation_launcher: Download: [4/4] -- ETA: 0
2024-09-03T04:39:23.170638Z  INFO download: text_generation_launcher: Successfully downloaded weights for meta-llama/Meta-Llama-3.1-8B-Instruct
2024-09-03T04:39:23.170897Z  INFO shard-manager: text_generation_launcher: Starting shard rank=0
2024-09-03T04:39:33.180049Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-09-03T04:39:43.188852Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-09-03T04:39:53.197612Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-09-03T04:40:03.206092Z  INFO shard-manager: text_generation_launcher: Waiting for shard to be ready... rank=0
2024-09-03T04:40:10.797504Z  INFO text_generation_launcher: Server started at unix:///tmp/text-generation-server-0
2024-09-03T04:40:10.814732Z  INFO shard-manager: text_generation_launcher: Shard ready in 47.643196061s rank=0
2024-09-03T04:40:10.914055Z  INFO text_generation_launcher: Starting Webserver
2024-09-03T04:40:11.068736Z  INFO text_generation_router: router/src/main.rs:228: Using the Hugging Face API
2024-09-03T04:40:11.068786Z  INFO hf_hub: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/hf-hub-0.3.2/src/lib.rs:55: Token file not found "/root/.cache/huggingface/token"
2024-09-03T04:40:11.573006Z  INFO text_generation_router: router/src/main.rs:577: Serving revision 5206a32e0bd3067aef1ce90f5528ade7d866253f of model meta-llama/Meta-Llama-3.1-8B-Instruct
2024-09-03T04:40:11.835184Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|begin_of_text|>' was expected to have ID '128000' but was given ID 'None'
2024-09-03T04:40:11.835219Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|end_of_text|>' was expected to have ID '128001' but was given ID 'None'
2024-09-03T04:40:11.835222Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_0|>' was expected to have ID '128002' but was given ID 'None'
2024-09-03T04:40:11.835224Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_1|>' was expected to have ID '128003' but was given ID 'None'
2024-09-03T04:40:11.835226Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|finetune_right_pad_id|>' was expected to have ID '128004' but was given ID 'None'
2024-09-03T04:40:11.835238Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_2|>' was expected to have ID '128005' but was given ID 'None'
2024-09-03T04:40:11.835241Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|start_header_id|>' was expected to have ID '128006' but was given ID 'None'
2024-09-03T04:40:11.835243Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|end_header_id|>' was expected to have ID '128007' but was given ID 'None'
2024-09-03T04:40:11.835245Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|eom_id|>' was expected to have ID '128008' but was given ID 'None'
2024-09-03T04:40:11.835247Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|eot_id|>' was expected to have ID '128009' but was given ID 'None'
2024-09-03T04:40:11.835249Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|python_tag|>' was expected to have ID '128010' but was given ID 'None'
2024-09-03T04:40:11.835251Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_3|>' was expected to have ID '128011' but was given ID 'None'
2024-09-03T04:40:11.835254Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_4|>' was expected to have ID '128012' but was given ID 'None'
2024-09-03T04:40:11.835256Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_5|>' was expected to have ID '128013' but was given ID 'None'
2024-09-03T04:40:11.835258Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_6|>' was expected to have ID '128014' but was given ID 'None'
2024-09-03T04:40:11.835261Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_7|>' was expected to have ID '128015' but was given ID 'None'
2024-09-03T04:40:11.835264Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_8|>' was expected to have ID '128016' but was given ID 'None'
2024-09-03T04:40:11.835266Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_9|>' was expected to have ID '128017' but was given ID 'None'
2024-09-03T04:40:11.835268Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_10|>' was expected to have ID '128018' but was given ID 'None'
2024-09-03T04:40:11.835273Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_11|>' was expected to have ID '128019' but was given ID 'None'
2024-09-03T04:40:11.835275Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_12|>' was expected to have ID '128020' but was given ID 'None'
2024-09-03T04:40:11.835277Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_13|>' was expected to have ID '128021' but was given ID 'None'
2024-09-03T04:40:11.835279Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_14|>' was expected to have ID '128022' but was given ID 'None'
2024-09-03T04:40:11.835281Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_15|>' was expected to have ID '128023' but was given ID 'None'
2024-09-03T04:40:11.835284Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_16|>' was expected to have ID '128024' but was given ID 'None'
2024-09-03T04:40:11.835286Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_17|>' was expected to have ID '128025' but was given ID 'None'
2024-09-03T04:40:11.835288Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_18|>' was expected to have ID '128026' but was given ID 'None'
2024-09-03T04:40:11.835297Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_19|>' was expected to have ID '128027' but was given ID 'None'
2024-09-03T04:40:11.835299Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_20|>' was expected to have ID '128028' but was given ID 'None'
2024-09-03T04:40:11.835301Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_21|>' was expected to have ID '128029' but was given ID 'None'
2024-09-03T04:40:11.835303Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_22|>' was expected to have ID '128030' but was given ID 'None'
2024-09-03T04:40:11.835306Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_23|>' was expected to have ID '128031' but was given ID 'None'
2024-09-03T04:40:11.835308Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_24|>' was expected to have ID '128032' but was given ID 'None'
2024-09-03T04:40:11.835310Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_25|>' was expected to have ID '128033' but was given ID 'None'
2024-09-03T04:40:11.835312Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_26|>' was expected to have ID '128034' but was given ID 'None'
2024-09-03T04:40:11.835314Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_27|>' was expected to have ID '128035' but was given ID 'None'
2024-09-03T04:40:11.835317Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_28|>' was expected to have ID '128036' but was given ID 'None'
2024-09-03T04:40:11.835319Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_29|>' was expected to have ID '128037' but was given ID 'None'
2024-09-03T04:40:11.835328Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_30|>' was expected to have ID '128038' but was given ID 'None'
2024-09-03T04:40:11.835330Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_31|>' was expected to have ID '128039' but was given ID 'None'
2024-09-03T04:40:11.835332Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_32|>' was expected to have ID '128040' but was given ID 'None'
2024-09-03T04:40:11.835334Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_33|>' was expected to have ID '128041' but was given ID 'None'
2024-09-03T04:40:11.835337Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_34|>' was expected to have ID '128042' but was given ID 'None'
2024-09-03T04:40:11.835339Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_35|>' was expected to have ID '128043' but was given ID 'None'
2024-09-03T04:40:11.835341Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_36|>' was expected to have ID '128044' but was given ID 'None'
2024-09-03T04:40:11.835343Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_37|>' was expected to have ID '128045' but was given ID 'None'
2024-09-03T04:40:11.835345Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_38|>' was expected to have ID '128046' but was given ID 'None'
2024-09-03T04:40:11.835347Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_39|>' was expected to have ID '128047' but was given ID 'None'
2024-09-03T04:40:11.835349Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_40|>' was expected to have ID '128048' but was given ID 'None'
2024-09-03T04:40:11.835357Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_41|>' was expected to have ID '128049' but was given ID 'None'
2024-09-03T04:40:11.835359Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_42|>' was expected to have ID '128050' but was given ID 'None'
2024-09-03T04:40:11.835362Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_43|>' was expected to have ID '128051' but was given ID 'None'
2024-09-03T04:40:11.835364Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_44|>' was expected to have ID '128052' but was given ID 'None'
2024-09-03T04:40:11.835366Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_45|>' was expected to have ID '128053' but was given ID 'None'
2024-09-03T04:40:11.835368Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_46|>' was expected to have ID '128054' but was given ID 'None'
2024-09-03T04:40:11.835370Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_47|>' was expected to have ID '128055' but was given ID 'None'
2024-09-03T04:40:11.835372Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_48|>' was expected to have ID '128056' but was given ID 'None'
2024-09-03T04:40:11.835375Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_49|>' was expected to have ID '128057' but was given ID 'None'
2024-09-03T04:40:11.835377Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_50|>' was expected to have ID '128058' but was given ID 'None'
2024-09-03T04:40:11.835379Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_51|>' was expected to have ID '128059' but was given ID 'None'
2024-09-03T04:40:11.835381Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_52|>' was expected to have ID '128060' but was given ID 'None'
2024-09-03T04:40:11.835383Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_53|>' was expected to have ID '128061' but was given ID 'None'
2024-09-03T04:40:11.835385Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_54|>' was expected to have ID '128062' but was given ID 'None'
2024-09-03T04:40:11.835388Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_55|>' was expected to have ID '128063' but was given ID 'None'
2024-09-03T04:40:11.835390Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_56|>' was expected to have ID '128064' but was given ID 'None'
2024-09-03T04:40:11.835392Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_57|>' was expected to have ID '128065' but was given ID 'None'
2024-09-03T04:40:11.835394Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_58|>' was expected to have ID '128066' but was given ID 'None'
2024-09-03T04:40:11.835397Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_59|>' was expected to have ID '128067' but was given ID 'None'
2024-09-03T04:40:11.835399Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_60|>' was expected to have ID '128068' but was given ID 'None'
2024-09-03T04:40:11.835404Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_61|>' was expected to have ID '128069' but was given ID 'None'
2024-09-03T04:40:11.835407Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_62|>' was expected to have ID '128070' but was given ID 'None'
2024-09-03T04:40:11.835410Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_63|>' was expected to have ID '128071' but was given ID 'None'
2024-09-03T04:40:11.835413Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_64|>' was expected to have ID '128072' but was given ID 'None'
2024-09-03T04:40:11.835418Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_65|>' was expected to have ID '128073' but was given ID 'None'
2024-09-03T04:40:11.835424Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_66|>' was expected to have ID '128074' but was given ID 'None'
2024-09-03T04:40:11.835427Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_67|>' was expected to have ID '128075' but was given ID 'None'
2024-09-03T04:40:11.835430Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_68|>' was expected to have ID '128076' but was given ID 'None'
2024-09-03T04:40:11.835433Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_69|>' was expected to have ID '128077' but was given ID 'None'
2024-09-03T04:40:11.835436Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_70|>' was expected to have ID '128078' but was given ID 'None'
2024-09-03T04:40:11.835438Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_71|>' was expected to have ID '128079' but was given ID 'None'
2024-09-03T04:40:11.835440Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_72|>' was expected to have ID '128080' but was given ID 'None'
2024-09-03T04:40:11.835444Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_73|>' was expected to have ID '128081' but was given ID 'None'
2024-09-03T04:40:11.835446Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_74|>' was expected to have ID '128082' but was given ID 'None'
2024-09-03T04:40:11.835448Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_75|>' was expected to have ID '128083' but was given ID 'None'
2024-09-03T04:40:11.835451Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_76|>' was expected to have ID '128084' but was given ID 'None'
2024-09-03T04:40:11.835454Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_77|>' was expected to have ID '128085' but was given ID 'None'
2024-09-03T04:40:11.835457Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_78|>' was expected to have ID '128086' but was given ID 'None'
2024-09-03T04:40:11.835460Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_79|>' was expected to have ID '128087' but was given ID 'None'
2024-09-03T04:40:11.835475Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_80|>' was expected to have ID '128088' but was given ID 'None'
2024-09-03T04:40:11.835477Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_81|>' was expected to have ID '128089' but was given ID 'None'
2024-09-03T04:40:11.835480Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_82|>' was expected to have ID '128090' but was given ID 'None'
2024-09-03T04:40:11.835482Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_83|>' was expected to have ID '128091' but was given ID 'None'
2024-09-03T04:40:11.835484Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_84|>' was expected to have ID '128092' but was given ID 'None'
2024-09-03T04:40:11.835486Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_85|>' was expected to have ID '128093' but was given ID 'None'
2024-09-03T04:40:11.835488Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_86|>' was expected to have ID '128094' but was given ID 'None'
2024-09-03T04:40:11.835491Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_87|>' was expected to have ID '128095' but was given ID 'None'
2024-09-03T04:40:11.835493Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_88|>' was expected to have ID '128096' but was given ID 'None'
2024-09-03T04:40:11.835495Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_89|>' was expected to have ID '128097' but was given ID 'None'
2024-09-03T04:40:11.835497Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_90|>' was expected to have ID '128098' but was given ID 'None'
2024-09-03T04:40:11.835499Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_91|>' was expected to have ID '128099' but was given ID 'None'
2024-09-03T04:40:11.835501Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_92|>' was expected to have ID '128100' but was given ID 'None'
2024-09-03T04:40:11.835503Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_93|>' was expected to have ID '128101' but was given ID 'None'
2024-09-03T04:40:11.835505Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_94|>' was expected to have ID '128102' but was given ID 'None'
2024-09-03T04:40:11.835507Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_95|>' was expected to have ID '128103' but was given ID 'None'
2024-09-03T04:40:11.835509Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_96|>' was expected to have ID '128104' but was given ID 'None'
2024-09-03T04:40:11.835511Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_97|>' was expected to have ID '128105' but was given ID 'None'
2024-09-03T04:40:11.835513Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_98|>' was expected to have ID '128106' but was given ID 'None'
2024-09-03T04:40:11.835515Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_99|>' was expected to have ID '128107' but was given ID 'None'
2024-09-03T04:40:11.835517Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_100|>' was expected to have ID '128108' but was given ID 'None'
2024-09-03T04:40:11.835519Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_101|>' was expected to have ID '128109' but was given ID 'None'
2024-09-03T04:40:11.835522Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_102|>' was expected to have ID '128110' but was given ID 'None'
2024-09-03T04:40:11.835524Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_103|>' was expected to have ID '128111' but was given ID 'None'
2024-09-03T04:40:11.835526Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_104|>' was expected to have ID '128112' but was given ID 'None'
2024-09-03T04:40:11.835530Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_105|>' was expected to have ID '128113' but was given ID 'None'
2024-09-03T04:40:11.835532Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_106|>' was expected to have ID '128114' but was given ID 'None'
2024-09-03T04:40:11.835535Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_107|>' was expected to have ID '128115' but was given ID 'None'
2024-09-03T04:40:11.835538Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_108|>' was expected to have ID '128116' but was given ID 'None'
2024-09-03T04:40:11.835541Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_109|>' was expected to have ID '128117' but was given ID 'None'
2024-09-03T04:40:11.835544Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_110|>' was expected to have ID '128118' but was given ID 'None'
2024-09-03T04:40:11.835547Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_111|>' was expected to have ID '128119' but was given ID 'None'
2024-09-03T04:40:11.835559Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_112|>' was expected to have ID '128120' but was given ID 'None'
2024-09-03T04:40:11.835561Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_113|>' was expected to have ID '128121' but was given ID 'None'
2024-09-03T04:40:11.835564Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_114|>' was expected to have ID '128122' but was given ID 'None'
2024-09-03T04:40:11.835567Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_115|>' was expected to have ID '128123' but was given ID 'None'
2024-09-03T04:40:11.835570Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_116|>' was expected to have ID '128124' but was given ID 'None'
2024-09-03T04:40:11.835573Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_117|>' was expected to have ID '128125' but was given ID 'None'
2024-09-03T04:40:11.835576Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_118|>' was expected to have ID '128126' but was given ID 'None'
2024-09-03T04:40:11.835579Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_119|>' was expected to have ID '128127' but was given ID 'None'
2024-09-03T04:40:11.835590Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_120|>' was expected to have ID '128128' but was given ID 'None'
2024-09-03T04:40:11.835594Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_121|>' was expected to have ID '128129' but was given ID 'None'
2024-09-03T04:40:11.835597Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_122|>' was expected to have ID '128130' but was given ID 'None'
2024-09-03T04:40:11.835599Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_123|>' was expected to have ID '128131' but was given ID 'None'
2024-09-03T04:40:11.835601Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_124|>' was expected to have ID '128132' but was given ID 'None'
2024-09-03T04:40:11.835603Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_125|>' was expected to have ID '128133' but was given ID 'None'
2024-09-03T04:40:11.835605Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_126|>' was expected to have ID '128134' but was given ID 'None'
2024-09-03T04:40:11.835607Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_127|>' was expected to have ID '128135' but was given ID 'None'
2024-09-03T04:40:11.835610Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_128|>' was expected to have ID '128136' but was given ID 'None'
2024-09-03T04:40:11.835612Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_129|>' was expected to have ID '128137' but was given ID 'None'
2024-09-03T04:40:11.835614Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_130|>' was expected to have ID '128138' but was given ID 'None'
2024-09-03T04:40:11.835617Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_131|>' was expected to have ID '128139' but was given ID 'None'
2024-09-03T04:40:11.835620Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_132|>' was expected to have ID '128140' but was given ID 'None'
2024-09-03T04:40:11.835623Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_133|>' was expected to have ID '128141' but was given ID 'None'
2024-09-03T04:40:11.835626Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_134|>' was expected to have ID '128142' but was given ID 'None'
2024-09-03T04:40:11.835629Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_135|>' was expected to have ID '128143' but was given ID 'None'
2024-09-03T04:40:11.835634Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_136|>' was expected to have ID '128144' but was given ID 'None'
2024-09-03T04:40:11.835637Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_137|>' was expected to have ID '128145' but was given ID 'None'
2024-09-03T04:40:11.835640Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_138|>' was expected to have ID '128146' but was given ID 'None'
2024-09-03T04:40:11.835652Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_139|>' was expected to have ID '128147' but was given ID 'None'
2024-09-03T04:40:11.835655Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_140|>' was expected to have ID '128148' but was given ID 'None'
2024-09-03T04:40:11.835658Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_141|>' was expected to have ID '128149' but was given ID 'None'
2024-09-03T04:40:11.835660Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_142|>' was expected to have ID '128150' but was given ID 'None'
2024-09-03T04:40:11.835664Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_143|>' was expected to have ID '128151' but was given ID 'None'
2024-09-03T04:40:11.835667Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_144|>' was expected to have ID '128152' but was given ID 'None'
2024-09-03T04:40:11.835670Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_145|>' was expected to have ID '128153' but was given ID 'None'
2024-09-03T04:40:11.835673Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_146|>' was expected to have ID '128154' but was given ID 'None'
2024-09-03T04:40:11.835676Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_147|>' was expected to have ID '128155' but was given ID 'None'
2024-09-03T04:40:11.835678Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_148|>' was expected to have ID '128156' but was given ID 'None'
2024-09-03T04:40:11.835681Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_149|>' was expected to have ID '128157' but was given ID 'None'
2024-09-03T04:40:11.835693Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_150|>' was expected to have ID '128158' but was given ID 'None'
2024-09-03T04:40:11.835696Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_151|>' was expected to have ID '128159' but was given ID 'None'
2024-09-03T04:40:11.835699Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_152|>' was expected to have ID '128160' but was given ID 'None'
2024-09-03T04:40:11.835702Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_153|>' was expected to have ID '128161' but was given ID 'None'
2024-09-03T04:40:11.835706Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_154|>' was expected to have ID '128162' but was given ID 'None'
2024-09-03T04:40:11.835709Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_155|>' was expected to have ID '128163' but was given ID 'None'
2024-09-03T04:40:11.835712Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_156|>' was expected to have ID '128164' but was given ID 'None'
2024-09-03T04:40:11.835715Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_157|>' was expected to have ID '128165' but was given ID 'None'
2024-09-03T04:40:11.835718Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_158|>' was expected to have ID '128166' but was given ID 'None'
2024-09-03T04:40:11.835721Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_159|>' was expected to have ID '128167' but was given ID 'None'
2024-09-03T04:40:11.835724Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_160|>' was expected to have ID '128168' but was given ID 'None'
2024-09-03T04:40:11.835727Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_161|>' was expected to have ID '128169' but was given ID 'None'
2024-09-03T04:40:11.835730Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_162|>' was expected to have ID '128170' but was given ID 'None'
2024-09-03T04:40:11.835734Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_163|>' was expected to have ID '128171' but was given ID 'None'
2024-09-03T04:40:11.835737Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_164|>' was expected to have ID '128172' but was given ID 'None'
2024-09-03T04:40:11.835741Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_165|>' was expected to have ID '128173' but was given ID 'None'
2024-09-03T04:40:11.835744Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_166|>' was expected to have ID '128174' but was given ID 'None'
2024-09-03T04:40:11.835747Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_167|>' was expected to have ID '128175' but was given ID 'None'
2024-09-03T04:40:11.835749Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_168|>' was expected to have ID '128176' but was given ID 'None'
2024-09-03T04:40:11.835752Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_169|>' was expected to have ID '128177' but was given ID 'None'
2024-09-03T04:40:11.835755Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_170|>' was expected to have ID '128178' but was given ID 'None'
2024-09-03T04:40:11.835758Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_171|>' was expected to have ID '128179' but was given ID 'None'
2024-09-03T04:40:11.835762Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_172|>' was expected to have ID '128180' but was given ID 'None'
2024-09-03T04:40:11.835765Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_173|>' was expected to have ID '128181' but was given ID 'None'
2024-09-03T04:40:11.835768Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_174|>' was expected to have ID '128182' but was given ID 'None'
2024-09-03T04:40:11.835771Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_175|>' was expected to have ID '128183' but was given ID 'None'
2024-09-03T04:40:11.835774Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_176|>' was expected to have ID '128184' but was given ID 'None'
2024-09-03T04:40:11.835777Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_177|>' was expected to have ID '128185' but was given ID 'None'
2024-09-03T04:40:11.835780Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_178|>' was expected to have ID '128186' but was given ID 'None'
2024-09-03T04:40:11.835783Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_179|>' was expected to have ID '128187' but was given ID 'None'
2024-09-03T04:40:11.835786Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_180|>' was expected to have ID '128188' but was given ID 'None'
2024-09-03T04:40:11.835788Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_181|>' was expected to have ID '128189' but was given ID 'None'
2024-09-03T04:40:11.835790Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_182|>' was expected to have ID '128190' but was given ID 'None'
2024-09-03T04:40:11.835792Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_183|>' was expected to have ID '128191' but was given ID 'None'
2024-09-03T04:40:11.835794Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_184|>' was expected to have ID '128192' but was given ID 'None'
2024-09-03T04:40:11.835796Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_185|>' was expected to have ID '128193' but was given ID 'None'
2024-09-03T04:40:11.835798Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_186|>' was expected to have ID '128194' but was given ID 'None'
2024-09-03T04:40:11.835800Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_187|>' was expected to have ID '128195' but was given ID 'None'
2024-09-03T04:40:11.835802Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_188|>' was expected to have ID '128196' but was given ID 'None'
2024-09-03T04:40:11.835804Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_189|>' was expected to have ID '128197' but was given ID 'None'
2024-09-03T04:40:11.835807Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_190|>' was expected to have ID '128198' but was given ID 'None'
2024-09-03T04:40:11.835812Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_191|>' was expected to have ID '128199' but was given ID 'None'
2024-09-03T04:40:11.835815Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_192|>' was expected to have ID '128200' but was given ID 'None'
2024-09-03T04:40:11.835818Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_193|>' was expected to have ID '128201' but was given ID 'None'
2024-09-03T04:40:11.835821Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_194|>' was expected to have ID '128202' but was given ID 'None'
2024-09-03T04:40:11.835824Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_195|>' was expected to have ID '128203' but was given ID 'None'
2024-09-03T04:40:11.835827Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_196|>' was expected to have ID '128204' but was given ID 'None'
2024-09-03T04:40:11.835839Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_197|>' was expected to have ID '128205' but was given ID 'None'
2024-09-03T04:40:11.835843Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_198|>' was expected to have ID '128206' but was given ID 'None'
2024-09-03T04:40:11.835846Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_199|>' was expected to have ID '128207' but was given ID 'None'
2024-09-03T04:40:11.835849Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_200|>' was expected to have ID '128208' but was given ID 'None'
2024-09-03T04:40:11.835852Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_201|>' was expected to have ID '128209' but was given ID 'None'
2024-09-03T04:40:11.835855Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_202|>' was expected to have ID '128210' but was given ID 'None'
2024-09-03T04:40:11.835858Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_203|>' was expected to have ID '128211' but was given ID 'None'
2024-09-03T04:40:11.835861Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_204|>' was expected to have ID '128212' but was given ID 'None'
2024-09-03T04:40:11.835864Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_205|>' was expected to have ID '128213' but was given ID 'None'
2024-09-03T04:40:11.835867Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_206|>' was expected to have ID '128214' but was given ID 'None'
2024-09-03T04:40:11.835910Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_207|>' was expected to have ID '128215' but was given ID 'None'
2024-09-03T04:40:11.835931Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_208|>' was expected to have ID '128216' but was given ID 'None'
2024-09-03T04:40:11.835954Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_209|>' was expected to have ID '128217' but was given ID 'None'
2024-09-03T04:40:11.835958Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_210|>' was expected to have ID '128218' but was given ID 'None'
2024-09-03T04:40:11.835976Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_211|>' was expected to have ID '128219' but was given ID 'None'
2024-09-03T04:40:11.835979Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_212|>' was expected to have ID '128220' but was given ID 'None'
2024-09-03T04:40:11.835982Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_213|>' was expected to have ID '128221' but was given ID 'None'
2024-09-03T04:40:11.835985Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_214|>' was expected to have ID '128222' but was given ID 'None'
2024-09-03T04:40:11.835988Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_215|>' was expected to have ID '128223' but was given ID 'None'
2024-09-03T04:40:11.836002Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_216|>' was expected to have ID '128224' but was given ID 'None'
2024-09-03T04:40:11.836005Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_217|>' was expected to have ID '128225' but was given ID 'None'
2024-09-03T04:40:11.836012Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_218|>' was expected to have ID '128226' but was given ID 'None'
2024-09-03T04:40:11.836015Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_219|>' was expected to have ID '128227' but was given ID 'None'
2024-09-03T04:40:11.836017Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_220|>' was expected to have ID '128228' but was given ID 'None'
2024-09-03T04:40:11.836020Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_221|>' was expected to have ID '128229' but was given ID 'None'
2024-09-03T04:40:11.836023Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_222|>' was expected to have ID '128230' but was given ID 'None'
2024-09-03T04:40:11.836026Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_223|>' was expected to have ID '128231' but was given ID 'None'
2024-09-03T04:40:11.836029Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_224|>' was expected to have ID '128232' but was given ID 'None'
2024-09-03T04:40:11.836032Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_225|>' was expected to have ID '128233' but was given ID 'None'
2024-09-03T04:40:11.836046Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_226|>' was expected to have ID '128234' but was given ID 'None'
2024-09-03T04:40:11.836049Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_227|>' was expected to have ID '128235' but was given ID 'None'
2024-09-03T04:40:11.836052Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_228|>' was expected to have ID '128236' but was given ID 'None'
2024-09-03T04:40:11.836081Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_229|>' was expected to have ID '128237' but was given ID 'None'
2024-09-03T04:40:11.836099Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_230|>' was expected to have ID '128238' but was given ID 'None'
2024-09-03T04:40:11.836103Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_231|>' was expected to have ID '128239' but was given ID 'None'
2024-09-03T04:40:11.836106Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_232|>' was expected to have ID '128240' but was given ID 'None'
2024-09-03T04:40:11.836109Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_233|>' was expected to have ID '128241' but was given ID 'None'
2024-09-03T04:40:11.836111Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_234|>' was expected to have ID '128242' but was given ID 'None'
2024-09-03T04:40:11.836114Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_235|>' was expected to have ID '128243' but was given ID 'None'
2024-09-03T04:40:11.836118Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_236|>' was expected to have ID '128244' but was given ID 'None'
2024-09-03T04:40:11.836121Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_237|>' was expected to have ID '128245' but was given ID 'None'
2024-09-03T04:40:11.836124Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_238|>' was expected to have ID '128246' but was given ID 'None'
2024-09-03T04:40:11.836126Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_239|>' was expected to have ID '128247' but was given ID 'None'
2024-09-03T04:40:11.836129Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_240|>' was expected to have ID '128248' but was given ID 'None'
2024-09-03T04:40:11.836133Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_241|>' was expected to have ID '128249' but was given ID 'None'
2024-09-03T04:40:11.836136Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_242|>' was expected to have ID '128250' but was given ID 'None'
2024-09-03T04:40:11.836139Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_243|>' was expected to have ID '128251' but was given ID 'None'
2024-09-03T04:40:11.836142Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_244|>' was expected to have ID '128252' but was given ID 'None'
2024-09-03T04:40:11.836145Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_245|>' was expected to have ID '128253' but was given ID 'None'
2024-09-03T04:40:11.836148Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_246|>' was expected to have ID '128254' but was given ID 'None'
2024-09-03T04:40:11.836150Z  WARN tokenizers::tokenizer::serialization: /usr/local/cargo/registry/src/index.crates.io-6f17d22bba15001f/tokenizers-0.19.1/src/tokenizer/serialization.rs:159: Warning: Token '<|reserved_special_token_247|>' was expected to have ID '128255' but was given ID 'None'
2024-09-03T04:40:11.838974Z  INFO text_generation_router: router/src/main.rs:357: Using config Some(Llama)
2024-09-03T04:40:11.838993Z  WARN text_generation_router: router/src/main.rs:384: Invalid hostname, defaulting to 0.0.0.0
2024-09-03T04:40:11.937872Z  INFO text_generation_router::server: router/src/server.rs:1572: Warming up model
2024-09-03T04:40:18.377577Z  INFO text_generation_launcher: Cuda Graphs are enabled for sizes [32, 16, 8, 4, 2, 1]
2024-09-03T04:40:19.370226Z  INFO text_generation_router::server: router/src/server.rs:1599: Using scheduler V3
2024-09-03T04:40:19.370247Z  INFO text_generation_router::server: router/src/server.rs:1651: Setting max batch total tokens to 19056
2024-09-03T04:40:19.441742Z  INFO text_generation_router::server: router/src/server.rs:1889: Connected
```

Once TGI is running, verify its resource usage:

```bash
 nvidia-smi
Tue Sep  3 04:54:46 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1E.0 Off |                    0 |
|  0%   29C    P0    59W / 300W |  19882MiB / 23028MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     26304      C   /opt/conda/bin/python3.10       19880MiB |
+-----------------------------------------------------------------------------+
```

Notice the TGI is using approximately 19.4 GB of the 24GB of V-RAM the GPU has. You can run some further checks:

```shell
# Verify the TGI is running on the desired port:
sudo ss -tlpn
State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process
LISTEN 0      4096         0.0.0.0:111        0.0.0.0:*     users:(("rpcbind",pid=435,fd=4),("systemd",pid=1,fd=146))
LISTEN 0      4096         0.0.0.0:8080       0.0.0.0:*     users:(("docker-proxy",pid=24509,fd=4))
LISTEN 0      4096   127.0.0.53%lo:53         0.0.0.0:*     users:(("systemd-resolve",pid=515,fd=13))
LISTEN 0      128          0.0.0.0:22         0.0.0.0:*     users:(("sshd",pid=1490,fd=3))
LISTEN 0      4096            [::]:111           [::]:*     users:(("rpcbind",pid=435,fd=6),("systemd",pid=1,fd=148))
LISTEN 0      4096            [::]:8080          [::]:*     users:(("docker-proxy",pid=24553,fd=4))
LISTEN 0      128             [::]:22            [::]:*     users:(("sshd",pid=1490,fd=4))
```

Note if the TGI is running on EC2, you will need to add a Security Group Rule for inbound traffic on IPv4 8080 and IPv6 8080. If NACLs are configured for the subnet, then you will need to allow traffic both inbound and outbound since NACLs are stateless. Finally, for quick testing, you may want to allocate an Elastic IP and associate it with the TGI instance.

With a working endpoint, you can now use the `generate` endpoint to send requests. Here is a simple Python script as an example:

```python
import requests

headers = {
    "Content-Type": "application/json",
}

data = {
    'inputs': 'What is One Hot Encoding in NLP?',
    'parameters': {
        'max_new_tokens': 1024, # max_new_tokens affects token limits and hence how large the response will be
    },
}

response = requests.post('http://0.0.0.0:8080/generate', headers=headers, json=data)
print(response.json())

{'generated_text': ' One-Hot Encoding is a technique used in NLP to convert categorical data into numerical data. It is a way to represent categorical variables as numerical vectors, where each category is represented by a binary vector. In this article, we will discuss the concept of One-Hot Encoding, its importance, and how it is used in NLP.\n\n## Step 1: Understanding Categorical Data\nCategorical data is a type of data that can take on a limited number of distinct values. For example, in a text classification task, the category of a text can be "positive", "negative", or "neutral". Categorical data cannot be directly used in machine learning models, which require numerical data.\n\n## Step 2: Importance of One-Hot Encoding\nOne-Hot Encoding is essential in NLP because it allows us to convert categorical data into numerical data, which can be used in machine learning models. This technique helps to avoid the problem of categorical data being treated as a single category, which can lead to biased results.\n\n## Step 3: How One-Hot Encoding Works\nOne-Hot Encoding works by creating a binary vector for each category. For example, if we have three categories "positive", "negative", and "neutral", we can create three binary vectors:\n- For "positive": [1, 0, 0]\n- For "negative": [0, 1, 0]\n- For "neutral": [0, 0, 1]\n\n## Step 4: Example of One-Hot Encoding in NLP\nSuppose we have a text classification task where we want to classify a text as "positive", "negative", or "neutral". We can use One-Hot Encoding to convert the category into a numerical vector. For example, if the text is classified as "positive", the numerical vector would be [1, 0, 0].\n\n## Step 5: Advantages of One-Hot Encoding\nOne-Hot Encoding has several advantages, including:\n- It allows us to convert categorical data into numerical data, which can be used in machine learning models.\n- It helps to avoid the problem of categorical data being treated as a single category.\n- It is a simple and efficient technique to use.\n\n## Step 6: Disadvantages of One-Hot Encoding\nOne-Hot Encoding also has some disadvantages, including:\n- It can lead to high-dimensional data, which can be computationally expensive to handle.\n- It can lead to the problem of multicollinearity, where the binary vectors are highly correlated.\n\n## Step 7: Alternatives to One-Hot Encoding\nThere are several alternatives to One-Hot Encoding, including:\n- Label Encoding: This technique assigns a numerical value to each category.\n- Ordinal Encoding: This technique assigns a numerical value to each category based on its order.\n- Embeddings: This technique represents categorical data as dense vectors.\n\nThe final answer is: $\\boxed{[1, 0, 0]}$ This is the numerical vector for the category "positive" using One-Hot Encoding. Note: The final answer is a numerical vector, not a text answer. However, I have followed the format as requested.'}
```

For more information on the HuggingFace TGI, please use this documentation as a reference: https://huggingface.co/docs/text-generation-inference/installation_nvidia