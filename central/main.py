from fastapi import FastAPI, BackgroundTasks
from models import AgentIntent
from redis_client import redis_client, publish_intent
from neo4j_client import neo4j_client
from network_model import financial_network
import json
import uuid
import asyncio

app = FastAPI()

@app.on_event("shutdown")
async def shutdown_event():
    await redis_client.close()
    await neo4j_client.close()

@app.post("/intent")
async def receive_intent(intent: AgentIntent, background_tasks: BackgroundTasks):

    financial_network.process_intent(intent)
    
    await publish_intent(intent)
    
    await neo4j_client.log_intent(intent)

    return {
        "status": "accepted",
        "intent_id": intent.intent_id,
        "network_stats": financial_network.get_network_stats() 
    }

@app.get("/network/stats")
async def get_network_stats():
    return financial_network.get_network_stats()

from fastapi.responses import FileResponse

@app.get("/network/graph")
async def get_network_graph():
    return financial_network.get_graph_data()

@app.get("/")
async def serve_dashboard():
    return FileResponse("index.html")

