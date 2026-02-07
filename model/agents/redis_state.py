"""
RedisStateManager — Publishes and reads system-state snapshots via Redis.

In production this would be a real Redis instance shared across processes.
For local simulation/testing we use ``fakeredis`` which is an in-memory
drop-in replacement with the same API.

Snapshot schema (per bank, stored as a Redis hash):
    bank:{id}:state  →  {
        "liquidity":       float,
        "capital":         float,
        "total_exposure":  float,
        "stressed":        0 | 1,
        "defaulted":       0 | 1,
        "missed_payment":  0 | 1,
    }

Global keys:
    system:n_banks           →  int
    system:aggregate_liq     →  float
    system:aggregate_exp     →  float
    system:n_stressed        →  int
    system:n_defaulted       →  int
    system:margin_rate       →  float   (externally set by CCP, read-only for banks)
    system:step              →  int

Intent queue (banks publish, CCP rule engine consumes):
    intents:queue            →  list of JSON-serialized ActionIntent dicts

Market data (exchange publishes, banks consume):
    market:latest            →  JSON string of latest {new_volatility, price_change_signal}

Margin calls (CCP publishes per-bank, banks consume):
    margin_calls:bank:{id}   →  list of JSON-serialized issue_margin_call intents
"""

from __future__ import annotations

import json
from typing import Any

import fakeredis


class RedisStateManager:
    """Thin wrapper around a Redis connection for the financial simulation."""

    def __init__(self, use_fake: bool = True, **redis_kwargs):
        if use_fake:
            self._r = fakeredis.FakeRedis(decode_responses=True)
        else:
            import redis
            self._r = redis.Redis(decode_responses=True, **redis_kwargs)

    # ── bank state publish / read ────────────────────────────────────────

    def publish_bank_state(self, bank_id: int, state: dict[str, Any]) -> None:
        """Write one bank's observable state into Redis."""
        key = f"bank:{bank_id}:state"
        self._r.hset(key, mapping={k: str(v) for k, v in state.items()})

    def get_bank_state(self, bank_id: int) -> dict[str, float]:
        """Read one bank's latest snapshot.  Returns numeric values."""
        key = f"bank:{bank_id}:state"
        raw = self._r.hgetall(key)
        return {k: float(v) for k, v in raw.items()} if raw else {}

    def get_all_bank_ids(self) -> list[int]:
        """Return sorted list of bank IDs present in Redis."""
        n = self._r.get("system:n_banks")
        if n is None:
            return []
        return list(range(int(float(n))))

    # ── system state publish / read ──────────────────────────────────────

    def publish_system_state(self, data: dict[str, Any]) -> None:
        """Write global / aggregate system metrics."""
        for k, v in data.items():
            self._r.set(f"system:{k}", str(v))

    def get_system_value(self, key: str) -> float | None:
        """Read a single system-level scalar (e.g. 'margin_rate')."""
        v = self._r.get(f"system:{key}")
        return float(v) if v is not None else None

    def get_full_snapshot(self) -> dict:
        """
        Return the complete system snapshot that a bank agent would pull.

        {
            "step":            int,
            "n_banks":         int,
            "aggregate_liq":   float,
            "aggregate_exp":   float,
            "n_stressed":      int,
            "n_defaulted":     int,
            "margin_rate":     float,
            "banks": {
                0: { "liquidity": ..., "capital": ..., ... },
                1: { ... },
                ...
            }
        }
        """
        snap: dict[str, Any] = {}
        for field in ("step", "n_banks", "aggregate_liq", "aggregate_exp",
                       "n_stressed", "n_defaulted", "margin_rate"):
            v = self._r.get(f"system:{field}")
            snap[field] = float(v) if v is not None else 0.0

        bank_ids = self.get_all_bank_ids()
        snap["banks"] = {bid: self.get_bank_state(bid) for bid in bank_ids}
        return snap

    # ── intent queue ─────────────────────────────────────────────────────

    def publish_intent(self, intent_dict: dict) -> None:
        """Push an action intent onto the global queue (for CCP consumption)."""
        self._r.rpush("intents:queue", json.dumps(intent_dict))

    def get_all_intents(self, clear: bool = False) -> list[dict]:
        """Read all intents from the queue. Optionally clear after reading."""
        raw = self._r.lrange("intents:queue", 0, -1)
        intents = [json.loads(r) for r in raw] if raw else []
        if clear:
            self._r.delete("intents:queue")
        return intents

    # ── market data channel ──────────────────────────────────────────────

    def publish_market_data(self, data: dict) -> None:
        """Write the latest market data snapshot (from exchange)."""
        self._r.set("market:latest", json.dumps(data))

    def get_market_data(self) -> dict:
        """Read the latest market data."""
        raw = self._r.get("market:latest")
        if raw is None:
            return {"new_volatility": 0.2, "price_change_signal": 0.0}
        return json.loads(raw)

    # ── margin call channel ──────────────────────────────────────────────

    def publish_margin_call(self, bank_id: int, call_dict: dict) -> None:
        """Push a margin call intent to a specific bank's queue."""
        self._r.rpush(f"margin_calls:bank:{bank_id}", json.dumps(call_dict))

    def get_pending_margin_calls(self, bank_id: int) -> list[dict]:
        """Read and consume all pending margin calls for a bank."""
        key = f"margin_calls:bank:{bank_id}"
        raw = self._r.lrange(key, 0, -1)
        calls = [json.loads(r) for r in raw] if raw else []
        self._r.delete(key)  # consume: clear after reading
        return calls

    # ── cleanup ──────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Clear all keys (useful between test runs)."""
        self._r.flushall()
