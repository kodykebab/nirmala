from neo4j import AsyncGraphDatabase
import os
from models import AgentIntent
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def log_intent(self, intent: AgentIntent):
        try:
            async with self.driver.session() as session:
                await session.execute_write(self._create_intent_node, intent)
                print(f"Logged intent {intent.intent_id} to Neo4j")
        except Exception as e:
            print(f"Failed to log intent to Neo4j: {e}")

    @staticmethod
    async def _create_intent_node(tx, intent: AgentIntent):
        import json
        
        # 1. Prepare properties from payload (flattening)
        # We perform a shallow merge of the payload into the node properties
        # excluding complex objects (dicts/lists) which we keep in payload_json
        properties = {
            "id": intent.intent_id,
            "type": intent.action_type,
            "tick": intent.tick,
            "visibility": intent.visibility.value,
            "payload_json": json.dumps(intent.payload)
        }
        
        # Extract potential target ID
        target_id = (intent.payload.get("target") or 
                     intent.payload.get("final_destination") or 
                     intent.payload.get("borrower_bank_id"))
        # Extract scalar fields for easier querying
        for k, v in intent.payload.items():
            if isinstance(v, (str, int, float, bool)):
                properties[k] = v

        # 2. Base Query: Create Intent and Link to Sender
        query = (
            "MERGE (a:Agent {id: $agent_id}) "
            "CREATE (i:Intent) "
            "SET i = $props "
            "MERGE (a)-[:ISSUED]->(i) "
        )
        
        # 3. Dynamic Relationship to Target (if exists)
        # 3. Dynamic Relationship to Target (if exists)
        if target_id:
            # Map action type to relationship type
            rel_type = "TARGETS" # Default
            action = intent.action_type.lower()
            
            if action == "provide_credit":
                rel_type = "LENDS_TO"
            elif action == "repay_interbank_loan":
                rel_type = "PAYS_TO"
            elif "otc" in action:
                rel_type = "PROPOSES_TO"
            elif action == "fire_sale_asset":
                rel_type = "SELLS_TO"
            elif "margin_call" in action:
                rel_type = "PAYS_MARGIN_TO"
            elif action == "deposit_default_fund":
                rel_type = "DEPOSITS_TO"
            elif action == "declare_default":
                rel_type = "DEFAULTS_ON"

            # Use APOC or dynamic cypher for variable relationship types if simple parameter doesn't work.
            # However, simpler approach in standard Cypher without APOC for variable RelTypes is strictly not supported directly in MERGE patterns like MERGE (a)-[:$rel]->(b).
            # We must construct the query string dynamically. Safe here since rel_type is internally controlled.
            
            query += (
                f"WITH i "
                f"MERGE (t:Agent {{id: $target_id}}) "
                f"MERGE (i)-[:{rel_type}]->(t) "
            )
            
        # Execute
        await tx.run(query, 
               agent_id=intent.agent_id, 
               props=properties,
               target_id=target_id
        )
        
neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
