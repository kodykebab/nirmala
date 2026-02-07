from pydantic import BaseModel
from typing import Dict, Any

from typing import Dict, Any, Optional
from enum import Enum

class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"

class AgentIntent(BaseModel):
    intent_id: str
    tick: int
    agent_id: str
    action_type: str
    payload: Dict[str, Any]
    belief_snapshot: Optional[Dict[str, Any]] = None
    risk_preference: Optional[Dict[str, Any]] = None
    visibility: Visibility

