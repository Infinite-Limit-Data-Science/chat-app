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

token=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
model=BAAI/bge-large-en-v1.5
volume=$PWD/teidata

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8070:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-client-batch-size 65 --auto-truncate

Once you have deployed a model, you can use the embed endpoint by sending requests:

curl 100.28.34.190:8070/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'

curl 100.28.34.190:8070/embed -X POST -d '{"inputs":"What is Deep Learning?"}' -H 'Content-Type: application/json'
[[0.018113142,0.00302585,-0.049911194,-0.03506436,0.0142142335,-0.023612928,-0.01585384,-0.021689294,-0.005009194,0.063208796,0.0146531835,0.028402645,0.023948595,-0.034057356,-0.019468723,-0.0037794884,-0.016912485,-0.0066746217,-0.04616721,-0.0037213922,-0.027137438,0.05071163,-0.027782952,-0.0048316773,-0.037000902,0.011477251,0.071368106,0.029125623,0.047742266,0.034986895,-0.019042684,-0.025898049,0.0071845786,-0.03245648,-0.02070811,-0.012503619,0.009986112,-0.024891047,-0.06362193,-0.051537894,-0.015195415,0.0324823,0.023173977,-0.094038576,-0.051150583,0.011231955,0.038446855,0.0021301985,-0.016331522,-0.003314718,-0.008236768,0.030055163,-0.009024295,-0.014420798,0.016822113,-0.021418177,0.014988851, ...

Look at all those beautiful vectors above!

### Launching Vector Store

```shell
# server
docker container run -d -p 6379:6379 --name redisearch redis/redis-stack-server:latest

# locally
sudo apt install redis-tools
redis-cli 
```