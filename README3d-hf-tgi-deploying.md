### Deploying

The easiest way of deploying the HuggingFace TGI is using the official Docker image. We will use the Deep Learning AMI GPU TensorFlow 2.12.0 (Ubuntu 20.04) 20230324 AMI as the base. This AMI supports the following EC2 instances: G3, P3, P3dn, P4d, G5, G4dn. Note the g4dn instance type is built on the NVIDIA Tesla T4 GPU, which was released in 2018 and build on the Turing architecture. Recent versions of the HuggingFace TGI will not run on this older GPU. Follow the link to get a description of this Instance Type: https://aws.amazon.com/ec2/instance-types/g5/.

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

At this point, you have the infrastructure, with the necessary NVidia GPUs, and you have the model name. You may think you are ready to run the LLM. But you require two more prerequisites. First, you need to generate and use a HuggingFace Token from the HuggingFace Settings > Access Tokens panel. Second, you will need to seek approval before being able to use the Meta Llama family of models. Hence, go to the Model's Model Card, click on "Files and versions", and then request approval. It can take up to a day to be granted approval. Once you have these prerequisites satisfied, connect to the EC2 Instance as the Ubuntu user. The actual parameters to pass to the Docker container depends a lot on your hardware. The `nvidia-smi` command tells you the name of your GPU and how much Cuda memory in use and how much Cuda memory the system has.

Use the following command to launch the TGI in a container:

```shell
token=[token-from-huggingface]
model=meta-llama/Meta-Llama-3.1-8B-Instruct
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $model
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

Note if the TGI is running on EC2, you will need to add a Security Group Rule for inbound traffic on IPv4 8080 and IPv6 8080. If NACLs are configured for the subnet, then you will need to allow traffic both inbound and outbound since NACLs are stateless. Finally, for quick testing, **You may want to allocate an Elastic IP and associate it with the TGI instance**.

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

Here are examples of how to use the HF TGI arguments:

```shell
model=meta-llama/Llama-3.1-70B-Instruct
token=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
volume=$PWD/data
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $model --max-batch-prefill-tokens 4800 --max-input-length 4768 --max-total-tokens 10960  --num-shard 4

model=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
token=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
volume=$PWD/data
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $model --max-batch-prefill-tokens 4800 --max-input-length 4768 --max-total-tokens 10960 --num-shard 4

model=meta-llama/Llama-Guard-3-8B
token=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
volume=$PWD/data
docker container run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.3.1 --model-id $model
```

Note shared memory (/dev/shm), which is managed by the --shm-size flag in Docker, is part of the system's CPU memory and does not directly impact GPU memory usage. GPU memory is managed separately by the GPU drivers and frameworks like PyTorch, which handle loading models into GPU memory for inference and training.

In the context of PyTorch inference APIs (like those used in Hugging Face TGI), GPU memory is used to store model weights, activations, and other data necessary for inference. Shared memory is more relevant to CPU-based processes that might require inter-process communication or that share data structures between processes running on the same machine, such as if multiple workers are handling requests and need access to the same memory space.



 


