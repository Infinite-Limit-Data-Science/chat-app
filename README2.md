### HuggingFace Text Generation Inference: Main Features

Source: https://github.com/huggingface/text-generation-inference

Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). You have the option to use Rust or Python as the codebase. TGI enables high-performance text generation for the most popular open-source LLMs, including Llama. Key features include Tensor Parallelism. Tensor Parallelism is a technique used in deep learning to parallelize the computation of large tensors across multiple GPUs.

The TGI also features Token Streaming, a technique used in NLP to process large amounts of text data in a streaming fashion. In traditional NLP models, the input text is typically processed in batches, where the entire batch is processed at once. However, this can be inefficient for large amounts of text data, as it requires a significant amount of memory and computation. Token Streaming addresses this issue by processing the input text in a streaming fashion, where the text is processed one token at a time. This allows for more efficient use of memory and computation, and enables the processing of large amounts of text data in real-time.

The TGI also performs continuous batching of incoming requests to improve the throughput by grouping multiple incoming requests together and processing them as a single batch. In traditional systems, each incoming request is processed individually.

Processing NLP tasks, such as text generation, is computationally expensive and memory-intensive. The TGI uses Flash Attention, a technique that reduces the computational cost of attention mechanisms in transformers by using a more efficient algorithm. It also leverages Paged Attention, a technique that reduces the memory requirements of transformers by processing the input data in smaller chunks, rather than all at once.

The TGI uses various techniques for quantization. This reduces the precision of the model's weights and activations from high-precision values to lower-precision ones, significantly reducing the memory requirements and computational cost of the model.

The TGI also supports Safetensors weight loading, which provides a safe and efficient way to load and store model weights, which can be especially important for large models.

Importantly, in the context of Text Generation Inference (TGI), a Stop Sequence is a special sequence of tokens that can be used to indicate the end of a generated text sequence. In other words, a Stop Sequence is a way to tell the model to stop generating text, usually because the desired output has been reached or because the model has reached a certain threshold of confidence. The TGI supports this as well.

The HuggingFace TGI has many other features, but only crucial ones are highlighted above.

### Launching TGI as a Docker Container

The easiest way of deploying the HuggingFace TGI is using the official Docker image. As a simple illustration, we will launch the Meta LLama 3 13B LLM on a g4dn.2xlarge EC2 Instance using the official Docker image in the us-east-1 AWS region. We will use the Deep Learning AMI GPU TensorFlow 2.12.0 (Ubuntu 20.04) 20230324 AMI as the base. This AMI supports the following EC2 instances: G3, P3, P3dn, P4d, G5, G4dn. Furthermore, it includes TensorFlow 2.12.0, which is compatible with the Meta Llama 3 13B model. Here are more specs:

- TensorFlow Version: 2.12.0
- CUDA Version: 11.2
- cuDNN Version: 8.1.0
- Ubuntu Version: 20.04

Once selecting the AMI, choose the g4dn.2xlarge Instance Type. Configure VPC settings as appropriate. Use 512GB GP3 EBS Volume. For Purchasing Option, select "Spot Instances" to potentially save a lot in cost, sometimes 90 percent savings. Pricing for the g4db,2xlarge Instance Type is available here: https://aws-pricing.com/g4dn.2xlarge.html.