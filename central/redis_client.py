import redis.asyncio as redis
import json
import os
from models import AgentIntent, Visibility
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
)

async def publish_intent(intent: AgentIntent):
    message = intent.model_dump_json()
    
    if intent.visibility == Visibility.PUBLIC:
        await redis_client.publish("public_intents", message)
    elif intent.visibility == Visibility.PRIVATE:
        # Check payload for destination/target
        # Common keys for target in the examples: "target", "borrower_bank_id", "final_destination"
        target = intent.payload.get("target") or \
                 intent.payload.get("borrower_bank_id") or \
                 intent.payload.get("final_destination")
        
        if target:
            # Publish to the specific target's channel
            await redis_client.publish(f"private_intents:{target}", message)
            
            # Also publish to the sender's own private channel so they have a record
            if intent.agent_id:
                await redis_client.publish(f"private_intents:{intent.agent_id}", message)
        else:
            print(f"Warning: Private intent {intent.intent_id} has no clear target in payload.")
            pass


