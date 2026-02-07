import redis.asyncio as redis

redis_client = redis.Redis(
    host='redis-14331.crce217.ap-south-1-1.ec2.cloud.redislabs.com',
    port=14331,
    decode_responses=True,
    username="default",
    password="4yGL6q48dnJyy5cFtT6fNoGADknpHxMG",
)

