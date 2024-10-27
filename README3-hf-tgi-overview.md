### HuggingFace Text Generation Inference: Main Features

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama. Key features include Tensor Parallelism. Tensor Parallelism is a technique used in deep learning to parallelize the computation of large tensors across multiple GPUs.

The TGI also features Token Streaming, a technique used in NLP to process large amounts of text data in a streaming fashion. In traditional NLP models, the input text is typically processed in batches, where the entire batch is processed at once. However, this can be inefficient for large amounts of text data, as it requires a significant amount of memory and computation. Token Streaming addresses this issue by processing the input text in a streaming fashion, where the text is processed one token at a time. This allows for more efficient use of memory and computation, and enables the processing of large amounts of text data in real-time. DO NOT CONFUSE TOKEN STREAMING with HTTP chunked encoding streaming. Token Streaming here refers to a feature of Flash Attention. Flash Attention performs attention computation in a streaming fashion, meaning it processes chunks of data directly on the GPU without having to store the full attention matrix in memory. This technique is key in managing long sequences, as it allows attention to be calculated incrementally, only on the needed tokens. By processing smaller chunks dynamically, Flash Attention is able to maintain high throughput while avoiding the overhead of managing large matrices filled with padding tokens.

The TGI also performs continuous batching of incoming requests to improve the throughput by grouping multiple incoming requests together and processing them as a single batch (dynamic batching?). In traditional systems, each incoming request is processed individually.

Processing NLP tasks, such as text generation, is computationally expensive and memory-intensive. The TGI uses Flash Attention, a technique that reduces the computational cost of attention mechanisms in transformers by using a more efficient algorithm. 

It also leverages Paged Attention, a technique that reduces the memory requirements of transformers by processing the input data in smaller chunks, rather than all at once.

The TGI uses various techniques for quantization. This reduces the precision of the model's weights and activations from high-precision values to lower-precision ones, significantly reducing the memory requirements and computational cost of the model.

The TGI also supports Safetensors weight loading, which provides a safe and efficient way to load and store model weights, which can be especially important for large models.

Importantly, in the context of Text Generation Inference (TGI), the TGI supports a Stop Sequence, which is a special sequence of tokens that can be used to indicate the end of a generated text sequence. In other words, a Stop Sequence is a way to tell the model to stop generating text, usually because the desired output has been reached or because the model has reached a certain threshold of confidence. The TGI supports this as well.

The HuggingFace TGI has many other features, but the top ones are highlighted above.

### NVidia GPUs

TGI optimized models are supported on NVIDIA H100, A100, A10G and T4 GPUs with CUDA 12.2+. Note that you have to install NVIDIA Container Toolkit to use it. For other NVIDIA GPUs, continuous batching will still apply, but some operations like flash attention and paged attention will not be executed. TGI can be used on NVIDIA GPUs through its official docker image:

```shell
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data

docker run --gpus all --shm-size 64g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:2.3.1 \
    --model-id $model
```
