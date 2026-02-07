from pydantic import BaseModel
from typing import Dict, Any

class AgentIntent(BaseModel):
    agent_id: str
    action_type: str
    payload: Dict[str, Any]
    tick: int

