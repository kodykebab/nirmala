from fastapi import FastAPI
from models import AgentIntent
from redis_client import redis_client
import json
import uuid

app = FastAPI()

@app.post("/intent")
async def receive_intent(intent: AgentIntent):
    intent_id = str(uuid.uuid4())

    data = intent.model_dump()
    data["intent_id"] = intent_id

    await redis_client.lpush("intent_queue", json.dumps(data))

    return {
        "status": "accepted",
        "intent_id": intent_id
    }

