# Chat App

### Local development backend API installation

```shell
# local
cd chat-app/api
pip3 install pipenv
pipenv install
pipenv shell
fastapi dev main.py

# emulate prod locally
cd chat-app
pipenv shell
uvicorn api.main:app --reload

# a true reproduction of prod

# physical nic
ip addr show dev eth0
    IP Address: 172.30.142.11
    Subnet Mask: /20 (equivalent to 255.255.240.0)
    Broadcast Address: 172.30.143.255



docker container run \
    --name chat-api \
    --rm \
    -d \
    -p 8000:8000 \
    -v $(pwd):/chat-app \
    -e MODELS='[{"name":"meta-llama/Llama-3.2-90B-Vision-Instruct","endpoints":[{"url":"http://llama-3.2:8080/","type":"tgi"}],"classification":"text-generation","active": true},{"name":"mistralai/Mistral-7B-Instruct-v0.3","endpoints":[{"url":"http://mistral-7b:8080/","type":"tgi"}],"classification":"text-generation"},{"name":"meta-llama/Llama-Guard-3-8B","endpoints":[{"url":"http://llama-guard-3.8:8080", "type":"tgi"}],"description":"Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. It can be used to classify content in both prompt classification and response classification.","stream":false,"classification":"content-safety","active":false}]' \
    -e EMBEDDING_MODELS='[{"name":"BAAI/bge-large-en-v1.5","endpoints":[{"url":"http://bge-large:8070", "type":"tei"}],"dimensions":1024,"max_batch_tokens":32768,"max_client_batch_size":128,"max_batch_requests":64,"num_workers":8,"auto_truncate":false,"active":true}]' \
    -e REDIS_URL='redis://100.28.34.190:6379/0' \
    -e VECTOR_STORE='redis' \
    -e VECTOR_STORE_SCHEMA='[{"name":"uuid","type":"tag"},{"name":"conversation_id","type":"tag"},{"name":"source","type":"tag"}]' \
    -e DATABASE_NAME=chat_history \
    -e MONGODB_URL='mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=chat-app' \
    -e NLP_HARMONY=true \
    -e LOG_LEVEL=INFO \
    -e LANGCHAIN_TRACING_V2=true \
    -e LANGCHAIN_ENDPOINT="https://api.smith.langchain.com" \
    -e LANGCHAIN_API_KEY="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" \
    -e LANGCHAIN_PROJECT="chat-app-v1.0" \
    chat-api:0.0.1
docker container logs
docker container top
docker container inspect
docker container stats
docker container exec -it chat-api bash

# testing
cd chat-app/api
pipenv shell
pipenv install --dev
pipenv run pytest -s
```