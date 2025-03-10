### Launching Vector Store

```shell
# server
docker container run -d -p 6379:6379 --name redisearch redis/redis-stack-server:latest

# locally
sudo apt install redis-tools
redis-cli 

redis-cli -u 'redis://100.28.34.190:6379/0'
```