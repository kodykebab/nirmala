"""
RedisStateManager — Publishes and reads system-state snapshots via Redis.

Connects to a real Redis instance (default: localhost:6379).
A fakeredis fallback is available for offline testing.

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

Market impact / sale tracking:
    sales:{tick}:{asset_type}  →  cumulative volume sold (float, via INCRBYFLOAT)
    market:depth               →  market depth parameter (float)
"""

from __future__ import annotations

import json
from typing import Any

import redis as _redis


class RedisStateManager:
    """Thin wrapper around a Redis connection for the financial simulation."""

    def __init__(
        self,
        use_fake: bool = False,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        username: str | None = None,
        password: str | None = None,
        **redis_kwargs,
    ):
        if use_fake:
            import fakeredis
            self._r = fakeredis.FakeRedis(decode_responses=True)
        else:
            self._r = _redis.Redis(
                host=host,
                port=port,
                db=db,
                username=username,
                password=password,
                decode_responses=True,
                socket_timeout=redis_kwargs.pop("socket_timeout", 5),
                socket_connect_timeout=redis_kwargs.pop("socket_connect_timeout", 5),
                **redis_kwargs,
            )

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

    # ── market impact / sale tracking ────────────────────────────────────

    def record_sale(self, tick: int, asset_type: str, amount: float) -> float:
        """
        Atomically increment cumulative sale volume for this tick + asset.
        Returns the NEW cumulative total (post-increment).
        Uses INCRBYFLOAT so concurrent writers are safe.
        """
        key = f"sales:{tick}:{asset_type}"
        new_total = self._r.incrbyfloat(key, amount)
        self._r.expire(key, 300)       # auto-expire after 300s
        return float(new_total)

    def get_cumulative_sales(self, tick: int, asset_type: str) -> float:
        """Total volume sold of an asset type in a specific tick."""
        val = self._r.get(f"sales:{tick}:{asset_type}")
        return float(val) if val is not None else 0.0

    def get_recent_sale_pressure(
        self, tick: int, asset_type: str, lookback: int = 3,
    ) -> float:
        """Total volume sold over the last ``lookback`` ticks."""
        keys = [f"sales:{t}:{asset_type}"
                for t in range(max(0, tick - lookback + 1), tick + 1)]
        if not keys:
            return 0.0
        vals = self._r.mget(keys)            # single round-trip
        return sum(float(v) for v in vals if v is not None)

    def compute_sale_price(
        self,
        tick: int,
        asset_type: str,
        amount: float,
        base_volatility: float,
        is_fire_sale: bool = False,
    ) -> dict:
        """
        Compute the effective per-unit sale price after market impact.

        Model (square-root impact — standard in market microstructure):
            base_price  = 1 − base_discount(vol, fire_sale)
            instant_imp = k_i · √( (cum_this_tick + amount) / depth )
            persist_imp = k_p · √( recent_3_tick_volume  / (depth·3) )
            total_imp   = min(0.50, instant + persist)
            eff_price   = max(0.05, base_price · (1 − total_imp))

        Fire-sales use higher k values → worse execution.

        Side-effect: atomically records ``amount`` into cumulative sales
        via INCRBYFLOAT, so the **next** caller in the same tick will see
        a higher cumulative volume and get a worse price.

        Returns dict:
            price_per_unit      effective per-unit price
            base_price          price without impact
            impact_discount     fraction lost to impact  [0, 0.50]
            cumulative_volume   total sold this tick (after this sale)
        """
        # ── base price (no impact) ──────────────────────────────────────
        if is_fire_sale:
            base_discount = min(0.45, 0.10 + base_volatility * 0.4)
        else:
            base_discount = min(0.20, 0.05 + base_volatility * 0.3)
        base_price = 1.0 - base_discount

        # ── read cumulative volume BEFORE recording this sale ───────────
        cum_before = self.get_cumulative_sales(tick, asset_type)

        # ── recent multi-tick pressure ──────────────────────────────────
        recent_pressure = self.get_recent_sale_pressure(
            tick, asset_type, lookback=3,
        )

        # ── market depth (set once per simulation in model.setup) ───────
        raw_depth = self._r.get("market:depth")
        depth = float(raw_depth) if raw_depth is not None else 200.0

        # ── impact coefficients ─────────────────────────────────────────
        k_instant = 0.15 if is_fire_sale else 0.08
        k_persist = 0.05 if is_fire_sale else 0.02

        instantaneous = k_instant * (
            (cum_before + amount) / max(depth, 1.0)
        ) ** 0.5

        persistent = k_persist * (
            recent_pressure / max(depth * 3, 1.0)
        ) ** 0.5

        total_impact = min(0.50, instantaneous + persistent)

        # ── effective price ─────────────────────────────────────────────
        effective_price = max(0.05, base_price * (1.0 - total_impact))

        # ── atomically record this sale (next caller sees updated cum) ──
        new_cum = self.record_sale(tick, asset_type, amount)

        return {
            "price_per_unit": round(effective_price, 4),
            "base_price": round(base_price, 4),
            "impact_discount": round(total_impact, 4),
            "cumulative_volume": round(new_cum, 2),
        }

    def set_market_depth(self, depth: float) -> None:
        """Set the market depth parameter (how liquid the market is)."""
        self._r.set("market:depth", str(depth))

    # ── public / private intent streams ──────────────────────────────────
    # Mirrors the channel architecture in central/redis_client.py:
    #   public_intents           → all agents see these
    #   private_intents:{agent}  → only the target agent sees these

    def publish_to_public_stream(self, intent_dict: dict) -> None:
        """Push a public intent to the per-tick broadcast stream."""
        tick = intent_dict.get("tick", 0)
        self._r.rpush(f"stream:public:{tick}", json.dumps(intent_dict))
        self._r.expire(f"stream:public:{tick}", 600)

    def publish_to_private_stream(
        self, target_agent_id: str, intent_dict: dict,
    ) -> None:
        """Push a private intent to a specific agent's private stream."""
        key = f"stream:private:{target_agent_id}"
        self._r.rpush(key, json.dumps(intent_dict))
        self._r.expire(key, 600)

    def read_public_stream(self, tick: int) -> list[dict]:
        """
        Read all public intents for a given tick (non-destructive).
        Every bank reads the same set of public intents.
        """
        key = f"stream:public:{tick}"
        raw = self._r.lrange(key, 0, -1)
        return [json.loads(r) for r in raw] if raw else []

    def read_private_stream(self, agent_id: str) -> list[dict]:
        """
        Read and consume all private intents directed at this agent.
        Destructive: clears the queue after reading (each message
        is delivered exactly once, matching Pub/Sub semantics).
        """
        key = f"stream:private:{agent_id}"
        raw = self._r.lrange(key, 0, -1)
        intents = [json.loads(r) for r in raw] if raw else []
        self._r.delete(key)
        return intents

    def route_intent(self, intent_dict: dict) -> None:
        """
        Smart router: inspects the visibility field and pushes the intent
        to the correct stream.  Mirrors central/redis_client.py logic:
          - PUBLIC  → stream:public:{tick}
          - PRIVATE → stream:private:{target} + stream:private:{sender}

        Also pushes to the legacy ``intents:queue`` so the CCP rule
        engine can still consume a global ordered log.
        """
        # legacy global queue (CCP / analytics)
        self._r.rpush("intents:queue", json.dumps(intent_dict))

        visibility = intent_dict.get("visibility", "private")

        if visibility == "public":
            self.publish_to_public_stream(intent_dict)
        else:
            # private — route to target + sender (mirror central logic)
            payload = intent_dict.get("payload", {})
            target = (
                payload.get("target")
                or payload.get("target_agent_id")
                or payload.get("borrower_bank_id")
                or payload.get("final_destination")
            )
            sender = intent_dict.get("agent_id")

            if target:
                self.publish_to_private_stream(target, intent_dict)
                # sender gets a copy too (own record of sent message)
                if sender and sender != target:
                    self.publish_to_private_stream(sender, intent_dict)

    def clear_public_stream(self, tick: int) -> None:
        """Remove a tick's public stream (housekeeping)."""
        self._r.delete(f"stream:public:{tick}")
