Like HF TGI, VLLM is an open-source library designed specifically for deploying and serving large language models (LLMs).

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries. As of now, vLLM’s binaries are compiled with CUDA 12.1 and public PyTorch release versions by default. In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for Linux running on x86 platform with cuda 12 for every commit since v0.5.3. You can download and install the latest one with the following command:

```shell
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Another way to access the latest code is to use the docker images:

```shell
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}
```

These docker images are used for CI and testing only, and they are not intended for production use. They will be expired after several days. Latest code can contain bugs and may not be stable. Please use it with caution.

### Deploying with Docker

vLLM offers an official Docker image for deployment. The image can be used to run OpenAI compatible server and is available on Docker Hub as vllm/vllm-openai.

```shell
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
```

You can either use the ipc=host flag or --shm-size flag to allow the container to access the host’s shared memory. vLLM uses PyTorch, which uses shared memory to share data between processes under the hood, particularly for tensor parallel inference.

By default vLLM will build for all GPU types for widest distribution. If you are just building for the current GPU type the machine is running on, you can add the argument --build-arg torch_cuda_arch_list="" for vLLM to find the current GPU type and build for that.

Before going into the details of distributed inference and serving, let’s first make it clear when to use distributed inference and what are the strategies available. The common practice is:

- Single GPU (no distributed inference): If your model fits in a single GPU, you probably don’t need to use distributed inference. Just use the single GPU to run the inference.

- Single-Node Multi-GPU (tensor parallel inference): If your model is too large to fit in a single GPU, but it can fit in a single node with multiple GPUs, you can use tensor parallelism. The tensor parallel size is the number of GPUs you want to use. For example, if you have 4 GPUs in a single node, you can set the tensor parallel size to 4.

- Multi-Node Multi-GPU (tensor parallel plus pipeline parallel inference): If your model is too large to fit in a single node, you can use tensor parallel together with pipeline parallelism. The tensor parallel size is the number of GPUs you want to use in each node, and the pipeline parallel size is the number of nodes you want to use. For example, if you have 16 GPUs in 2 nodes (8GPUs per node), you can set the tensor parallel size to 8 and the pipeline parallel size to 2.

In short, you should increase the number of GPUs and the number of nodes until you have enough GPU memory to hold the model. The tensor parallel size should be the number of GPUs in each node, and the pipeline parallel size should be the number of nodes.

After adding enough GPUs and nodes to hold the model, you can run vLLM first, which will print some logs like # GPU blocks: 790. Multiply the number by 16 (the block size), and you can get roughly the maximum number of tokens that can be served on the current configuration. If this number is not satisfying, e.g. you want higher throughput, you can further increase the number of GPUs or nodes, until the number of blocks is enough.

vLLM supports distributed tensor-parallel inference and serving. Currently, we support Megatron-LM’s tensor parallel algorithm. We also support pipeline parallel as a beta feature for online serving. We manage the distributed runtime with either Ray or python native multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently requires Ray.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs available on the same node for the configured tensor_parallel_size, otherwise Ray will be used. 

If a single node does not have enough GPUs to hold the model, you can run the model using multiple nodes. It is important to make sure the execution environment is the same on all nodes, including the model path, the Python environment. The recommended way is to use docker images to ensure the same environment, and hide the heterogeneity of the host machines via mapping them into the same docker configuration.

### Deploying with Docker on AWS EC2

Since vLLM uses Python, Pytorch and Cuda drivers, we will use an instance that has Python, Pytorch and Cuda drivers pre-installed. We could choose the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Amazon Linux 2) AMI or the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Ubuntu 20.04) AMI. We will choose the later since Ubuntu reflects our local setup.

vLLM requires access to one of the following EC2 instances: G4dn, G5, G6, Gr6, P4, P4de, P5. This is needed given the intensive resources required to deploy model’s weights and to store KV cache. It is not recommended to use G4dn actually as it is uses the older NVIDIA Tesla T4 GPU, which was released in 2018 and build on the Turing architecture. For example, the HuggingFace TGI does not support it. The HuggingFace TGI supports the G5 instance, which uses the NVIDIA A10G Tensor Core GPU powered by the Ampere architecture. Follow the link to get a description of this Instance Type: https://aws.amazon.com/ec2/instance-types/g5/. Hence, for vLLM, we will also use the G5 instance.

You need to generate an API key so you can securely access the endpoint. To do so, se Python's secrets library which is built-in and designed for generating cryptographically strong random characters suitable for managing data such as authentication tokens. From within the TGI server, launch the Python repl with python3 command and execute this code to get the secret:

```python
import secrets

def generate_api_key(length=32):
    return secrets.token_urlsafe(length)

VLLM_API_KEY = generate_api_key()
print("Generated API Key:", VLLM_API_KEY)
# Generated API Key: qV57mQ7AlaFQAJhSJ0zq9wryKJio0yByxIS-7gr33tk
```

Next retrieve your HuggingFace token since we will be using the microsoft/Phi-3.5-vision-instruct LLM available on HuggingFace hub. Then launch the container:

```shell
# before running the container, verify you have 24GB of V-RAM
 nvidia-smi
Sat Oct  5 04:36:40 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
|  0%   25C    P8              9W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# Then run the container
 docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model microsoft/Phi-3.5-vision-instruct --trust-remote-code
Unable to find image 'vllm/vllm-openai:latest' locally
latest: Pulling from vllm/vllm-openai
43cfb69dbb46: Pull complete
fbcd35dc5bc3: Pull complete
c7232af9ae05: Pull complete
db6cdef1932a: Pull complete
56dc85502937: Pull complete
9f61b3db38d6: Pull complete
c12eb87e9588: Pull complete
62092a5c3164: Pull complete
824b951aa4fc: Pull complete
893502e49a0d: Downloading [==============================>                    ]   2.08GB/3.425GB
549334a085cb: Download complete
e56d5cf4292b: Download complete
INFO 10-04 21:59:34 api_server.py:526] vLLM API server version 0.6.1.dev238+ge2c6e0a82
INFO 10-04 21:59:34 api_server.py:527] args: Namespace(host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, model='microsoft/Phi-3.5-vision-instruct', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=True, download_dir=None, load_format='auto', config_format='auto', dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=None, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=False, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=False, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False)
INFO 10-04 21:59:34 api_server.py:164] Multiprocessing frontend to use ipc:///tmp/e35a7c54-5fe3-4c03-ba9a-03d4dc3d9864 for IPC Path.
INFO 10-04 21:59:34 api_server.py:177] Started engine process with PID 20
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-3.5-vision-instruct:
- configuration_phi3_v.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
WARNING 10-04 21:59:34 arg_utils.py:940] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
WARNING 10-04 21:59:38 arg_utils.py:940] The model has a long context length (131072). This may cause OOM errors during the initial memory profiling phase, or result in low performance due to small KV cache space. Consider setting --max-model-len to a smaller value.
INFO 10-04 21:59:38 llm_engine.py:226] Initializing an LLM engine (v0.6.1.dev238+ge2c6e0a82) with config: model='microsoft/Phi-3.5-vision-instruct', speculative_config=None, tokenizer='microsoft/Phi-3.5-vision-instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=131072, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=microsoft/Phi-3.5-vision-instruct, use_v2_block_manager=False, num_scheduler_steps=1, multi_step_stream_outputs=False, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=True, mm_processor_kwargs=None)
INFO 10-04 21:59:38 selector.py:240] Cannot use FlashAttention-2 backend due to sliding window.
INFO 10-04 21:59:38 selector.py:116] Using XFormers backend.
/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
/usr/local/lib/python3.12/dist-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
INFO 10-04 21:59:41 model_runner.py:1014] Starting to load model microsoft/Phi-3.5-vision-instruct...
INFO 10-04 21:59:41 selector.py:240] Cannot use FlashAttention-2 backend due to sliding window.
INFO 10-04 21:59:41 selector.py:116] Using XFormers backend.
INFO 10-04 21:59:41 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:00<00:00,  1.59it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.84it/s]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00,  1.80it/s]
INFO 10-04 22:00:12 model_runner.py:1025] Loading model weights took 7.7498 GB
A new version of the following files was downloaded from https://huggingface.co/microsoft/Phi-3.5-vision-instruct:
- processing_phi3_v.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
/usr/local/lib/python3.12/dist-packages/transformers/models/auto/image_processing_auto.py:517: FutureWarning: The image_processor_class argument is deprecated and will be removed in v4.42. Please use `slow_image_processor_class`, or `fast_image_processor_class` instead
```

- --runtime nvidia: Specifies the NVIDIA runtime, enabling the use of NVIDIA GPUs within the Docker container for tasks that require significant computational power.
- --gpus all: Allows the container to access all available GPUs on the host machine, essential for GPU-accelerated processes.
- -v ~/.cache/huggingface:/root/.cache/huggingface: Mounts the host machine’s Hugging Face cache directory inside the container, facilitating faster access to frequently used model data.
- --env "HUGGING_FACE_HUB_TOKEN=<HF_TOKEN>": Sets an environment variable for the Hugging Face Hub token, necessary for authenticating and downloading models securely.
- --env "VLLM_API_KEY=<VLLM_API_KEY>": Configures an API key for accessing specific APIs that require authentication, ensuring secure API communication. This is optional.
- -p 8000:8000: Maps port 8000 on the host to port 8000 on the container, making the application hosted inside the container accessible at localhost:8000.
- --ipc=host: Uses the host’s IPC namespace, important for applications that need extensive process communication, like those running large-scale machine learning models.
- vLLM offers an official Docker image for deployment: vllm/vllm-openai:latest. The image can be used to run OpenAI compatible server and is available on Docker Hub as vllm/vllm-openai.
- --model microsoft/Phi-3.5-vision-instruct: Specifies which machine learning model to load and use within the application, targeting efficient and specific model operations.
- --trust-remote-code: The microsoft/Phi-3.5-vision-instruct model repository contains custom code that needs to be executed locally for the model to work correctly. By default, transformers library prevents executing untrusted code for security reasons, and so you must explicitly trust the remote code by setting trust_remote_code=True.

You may get the following error:

```shell
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.68 GiB. GPU 0 has a total capacity of 21.98 GiB of which 2.23 GiB is free. Including non-PyTorch memory, this process has 0 bytes memory in use. Of the allocated memory 16.17 GiB is allocated by PyTorch, and 3.27 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

The error you're encountering is due to insufficient GPU memory, where PyTorch is unable to allocate the required memory on your GPU to run the model. There are a few potential solutions to resolve the "CUDA out of memory" issue.

- Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: As the error suggests, this configuration can help avoid memory fragmentation and make better use of available GPU memory. You can set this environment variable in your Docker run command:

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --trust-remote-code
```

- Reduce Model Batch Size or Max Tokens: Some models allow you to configure batch sizes or maximum token limits, which directly impact GPU memory usage. The TGI has an argument called --max_model_len. You can specify a value like --max_model_len 30000. This sets a parameter that limits the model to a maximum length of 30000, optimizing performance and resource allocation within operational constraints. Note that we are setting the maximum number of tokens to 30000 for experimental purposes using the parameter max_model_len. Although the model's context window is 128,000, we are still limited by the GPU memory of the available machine.

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --max_model_len 30000 \
  --trust-remote-code
```

You may get the following error now:

```shell
 File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 483, in raise_if_cache_size_invalid
    raise ValueError(
ValueError: The model's max seq len (30000) is larger than the maximum number of tokens that can be stored in KV cache (19632). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

The error you're encountering is because the max_model_len you've set (30,000) exceeds the capacity of the Key-Value (KV) cache, which is used by the model to store intermediate activations. The KV cache's token limit is determined by available GPU memory, and since it's set at 19,632, this means you can't store 30,000 tokens in memory without adjusting some parameters.

Increase gpu_memory_utilization: You can allow the engine to use more GPU memory by increasing the gpu_memory_utilization parameter. This will let the model allocate more space for the KV cache, though it could limit how much memory is available for other processes. Here's how to specify this parameter:

```shell
docker run \
  --runtime nvidia \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HUGGING_FACE_HUB_TOKEN=[HF-TOKEN]" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model microsoft/Phi-3.5-vision-instruct \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.9 \
  --trust-remote-code
```

The --gpu_memory_utilization parameter defaults to 0.8, meaning 80% of the GPU memory is allocated for the model and its operations, leaving 20% as a buffer for system tasks or other processes. When you set --gpu_memory_utilization 0.9, you're increasing the allocation to 90% of the GPU's memory for the model, which allows for more memory to be used for operations such as the KV cache. This buffer is important to prevent GPU memory from being completely exhausted, which could cause out-of-memory (OOM) errors. However, by increasing it to 90%, you're giving the model more memory to work with at the risk of leaving less room for other processes running on the GPU.

No you should finally get successful output:

```shell
INFO 10-04 22:27:39 model_runner.py:1329] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-04 22:27:39 model_runner.py:1333] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-04 22:27:51 model_runner.py:1456] Graph capturing finished in 12 secs.
INFO 10-04 22:27:51 api_server.py:230] vLLM to use /tmp/tmpznccwmw8 as PROMETHEUS_MULTIPROC_DIR
WARNING 10-04 22:27:51 serving_embedding.py:189] embedding_mode is False. Embedding API will not work.
INFO 10-04 22:27:51 launcher.py:19] Available routes are:
INFO 10-04 22:27:51 launcher.py:27] Route: /openapi.json, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /docs, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /redoc, Methods: HEAD, GET
INFO 10-04 22:27:51 launcher.py:27] Route: /health, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /tokenize, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /detokenize, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/models, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /version, Methods: GET
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 10-04 22:27:51 launcher.py:27] Route: /v1/embeddings, Methods: POST
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Now let's explore the model.

Phi-3.5-vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications with visual and text input capabilities which require:

- Memory/compute constrained environments
- Latency bound scenarios
- General image understanding
- Optical character recognition
- Chart and table understanding
- Multiple image comparison
- Multi-image or video clip summarization

After obtaining the Phi-3.5-vision-instruct model checkpoints, users can use this sample code for inference.

```python
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3.5-vision-instruct" 

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='flash_attention_2'    
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, 
  trust_remote_code=True, 
  num_crops=4
) 

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
for i in range(1,20):
    url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg" 
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f"<|image_{i}|>\n"

messages = [
    {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages, 
  tokenize=False, 
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 1000, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, 
  eos_token_id=processor.tokenizer.eos_token_id, 
  **generation_args
)

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, 
  skip_special_tokens=True, 
  clean_up_tokenization_spaces=False)[0] 

print(response)
```