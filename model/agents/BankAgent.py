"""
BankAgent — Autonomous Bayesian agent in a CCP-coordinated financial network.

Each bank:
  1. Reads incoming events (margin calls from CCP, market data from exchange)
  2. Pulls the latest system snapshot from Redis
  3. Extracts an observation vector
  4. Updates FOUR private Bayesian belief channels:
       a. Counterparty default risk   (Beta-Bernoulli per neighbor)
       b. Network liquidity stress    (Gaussian with unknown mean)
       c. Expected margin call size   (Gaussian with unknown mean)
       d. Market volatility           (Gaussian with unknown mean)
     These beliefs are PRIVATE — invisible to the CCP.
  5. Computes risk metrics from the updated beliefs
  6. Chooses an action intent via expected-utility maximisation
  7. Publishes its action intent (to be consumed by the CCP rule engine)

Action intents emitted by banks (JSON schemas):
  Batch 1 (CCP-routed):
    - route_otc_proposal         (bank → CCP → target bank)
    - pay_margin_call            (bank → CCP)
    - sell_asset_standard        (bank → exchange)
    - reduce_exposure            (bank internal)
    - hoard_liquidity            (bank internal)
    - borrow                     (bank → target bank)
  Batch 2 (interbank / systemic):
    - PROVIDE_INTERBANK_CREDIT   (bank → target bank, direct)
    - REPAY_INTERBANK_LOAN       (bank → lender bank)
    - FIRE_SALE_ASSET            (bank → exchange, distressed)
    - DECLARE_DEFAULT            (bank → system, voluntary)
    - DEPOSIT_DEFAULT_FUND       (bank → CCP default fund)

Incoming events consumed by banks:
  - issue_margin_call            (CCP → bank)
  - update_market_data           (exchange → all banks)

Every intent carries:
  belief_snapshot   — agent's private beliefs at time of decision
  risk_preference   — agent's current risk attitude (liquidity_aversion,
                      default_tolerance)
"""

from __future__ import annotations

import uuid

import agentpy as ap
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import json


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian belief primitives
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BetaBelief:
    """Beta-Bernoulli belief about a binary event (e.g. neighbor defaults)."""
    alpha: float = 1.0
    beta: float = 9.0          # prior mean ≈ 0.10

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1))

    def update(self, observation: float) -> None:
        """observation ∈ [0, 1]: strength of distress signal."""
        self.alpha += observation
        self.beta += (1.0 - observation)


@dataclass
class GaussianBelief:
    """Normal-Normal conjugate belief about an unknown mean (known variance)."""
    mu: float = 0.0            # posterior mean
    tau: float = 1.0           # posterior precision  (1/variance)

    @property
    def mean(self) -> float:
        return self.mu

    @property
    def std(self) -> float:
        return (1.0 / self.tau) ** 0.5

    def update(self, observation: float, obs_precision: float = 1.0) -> None:
        """
        Conjugate update:  τ_new = τ_old + τ_obs
                           μ_new = (τ_old·μ_old + τ_obs·x) / τ_new
        """
        tau_new = self.tau + obs_precision
        self.mu = (self.tau * self.mu + obs_precision * observation) / tau_new
        self.tau = tau_new


# ═══════════════════════════════════════════════════════════════════════════
# ActionIntent — matches the JSON schema format used across the system
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActionIntent:
    """
    Structured intent matching the system-wide JSON schema.

    Every intent (emitted or received) has the same envelope:
        intent_id        — UUID v4
        tick             — simulation timestep
        agent_id         — originator  (e.g. "bank_03", "ccp_01")
        action_type      — one of the registered action types
        visibility       — "private" (routed) or "public" (broadcast)
        payload          — action-specific nested dict
        belief_snapshot  — agent's private beliefs at decision time
        risk_preference  — agent's risk attitude dict
    """
    intent_id: str
    tick: int
    agent_id: str
    action_type: str
    visibility: str                                    # "private" | "public"
    payload: dict = field(default_factory=dict)
    belief_snapshot: dict = field(default_factory=dict)
    risk_preference: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "intent_id": self.intent_id,
            "tick": self.tick,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "visibility": self.visibility,
            "payload": dict(self.payload),
            "belief_snapshot": dict(self.belief_snapshot),
            "risk_preference": dict(self.risk_preference),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ActionIntent:
        """Deserialize from a dict (e.g. from Redis JSON)."""
        return cls(
            intent_id=d["intent_id"],
            tick=int(d["tick"]),
            agent_id=d["agent_id"],
            action_type=d["action_type"],
            visibility=d.get("visibility", "private"),
            payload=d.get("payload", {}),
            belief_snapshot=d.get("belief_snapshot", {}),
            risk_preference=d.get("risk_preference", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# IntentFactory — builders for every registered intent schema
# ═══════════════════════════════════════════════════════════════════════════

class IntentFactory:
    """Convenience builders for all registered intent schemas."""

    # ─────────────────────────────────────────────────────────────────────
    # BATCH 1 — CCP-routed / original schemas
    # ─────────────────────────────────────────────────────────────────────

    # ── 1. route_otc_proposal  (bank → CCP → target bank) ───────────────
    @staticmethod
    def route_otc_proposal(
        tick: int,
        agent_id: str,
        target_agent_id: str,
        amount: float,
        interest_rate: float = 0.035,
        tenor_ticks: int = 10,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="route_otc_proposal",
            visibility="private",
            payload={
                "encrypted_content": {
                    "type": "otc_loan",
                    "amount": amount,
                    "interest_rate": interest_rate,
                    "tenor_ticks": tenor_ticks,
                },
                "target_agent_id": target_agent_id,
                "routing": "ccp",
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 2. issue_margin_call  (CCP → bank) ──────────────────────────────
    @staticmethod
    def issue_margin_call(
        tick: int,
        agent_id: str,
        target_agent_id: str,
        margin_amount: float,
        deadline_tick: int,
        reason: str = "position_change",
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="issue_margin_call",
            visibility="private",
            payload={
                "target_agent_id": target_agent_id,
                "margin_amount": margin_amount,
                "deadline_tick": deadline_tick,
                "reason": reason,
            },
        )

    # ── 3. update_market_data  (exchange → all banks) ────────────────────
    @staticmethod
    def update_market_data(
        tick: int,
        agent_id: str,
        new_volatility: float,
        price_change_signal: float,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="update_market_data",
            visibility="public",
            payload={
                "new_volatility": new_volatility,
                "price_change_signal": price_change_signal,
            },
        )

    # ── 4. pay_margin_call  (bank → CCP) ────────────────────────────────
    @staticmethod
    def pay_margin_call(
        tick: int,
        agent_id: str,
        amount: float,
        margin_call_id: str,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="pay_margin_call",
            visibility="private",
            payload={
                "amount": amount,
                "margin_call_id": margin_call_id,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 5. sell_asset_standard  (bank → exchange) ────────────────────────
    @staticmethod
    def sell_asset_standard(
        tick: int,
        agent_id: str,
        asset_type: str,
        amount: float,
        order_type: str = "market",
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="sell_asset_standard",
            visibility="public",
            payload={
                "asset_type": asset_type,
                "amount": amount,
                "order_type": order_type,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── supplementary bank actions (batch 1 cont.) ──────────────────────

    @staticmethod
    def borrow(
        tick: int,
        agent_id: str,
        target_agent_id: str,
        amount: float,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="borrow",
            visibility="private",
            payload={
                "amount": amount,
                "target_agent_id": target_agent_id,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    @staticmethod
    def reduce_exposure(
        tick: int,
        agent_id: str,
        target_neighbor_id: int | None = None,
        amount: float = 0.0,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="reduce_exposure",
            visibility="private",
            payload={
                "target_neighbor_id": target_neighbor_id,
                "amount": amount,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    @staticmethod
    def hoard_liquidity(
        tick: int,
        agent_id: str,
        estimated_recovery: float = 0.0,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="hoard_liquidity",
            visibility="private",
            payload={
                "estimated_recovery": estimated_recovery,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ─────────────────────────────────────────────────────────────────────
    # BATCH 2 — Interbank / systemic schemas
    # ─────────────────────────────────────────────────────────────────────

    # ── 6. PROVIDE_INTERBANK_CREDIT  (bank → target bank, direct) ───────
    @staticmethod
    def provide_interbank_credit(
        tick: int,
        agent_id: str,
        borrower_bank_id: str,
        principal: float,
        interest_rate: float = 0.05,
        maturity_tick: int = 50,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="PROVIDE_INTERBANK_CREDIT",
            visibility="private",
            payload={
                "borrower_bank_id": borrower_bank_id,
                "principal": principal,
                "interest_rate": interest_rate,
                "maturity_tick": maturity_tick,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 7. REPAY_INTERBANK_LOAN  (borrower → lender) ────────────────────
    @staticmethod
    def repay_interbank_loan(
        tick: int,
        agent_id: str,
        loan_id: str,
        principal: float,
        interest: float,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="REPAY_INTERBANK_LOAN",
            visibility="public",
            payload={
                "loan_id": loan_id,
                "principal": principal,
                "interest": interest,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 8. FIRE_SALE_ASSET  (bank → exchange, distressed) ───────────────
    @staticmethod
    def fire_sale_asset(
        tick: int,
        agent_id: str,
        exchange_id: str,
        asset_id: str,
        quantity: float,
        max_acceptable_discount: float = 0.20,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="FIRE_SALE_ASSET",
            visibility="public",
            payload={
                "exchange_id": exchange_id,
                "asset_id": asset_id,
                "quantity": quantity,
                "max_acceptable_discount": max_acceptable_discount,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 9. DECLARE_DEFAULT  (bank → system) ─────────────────────────────
    @staticmethod
    def declare_default(
        tick: int,
        agent_id: str,
        reason: str = "liquidity_exhaustion",
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="DECLARE_DEFAULT",
            visibility="public",
            payload={
                "reason": reason,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )

    # ── 10. DEPOSIT_DEFAULT_FUND  (bank → CCP default fund) ─────────────
    @staticmethod
    def deposit_default_fund(
        tick: int,
        agent_id: str,
        amount: float,
        belief_snapshot: dict | None = None,
        risk_preference: dict | None = None,
    ) -> ActionIntent:
        return ActionIntent(
            intent_id=str(uuid.uuid4()),
            tick=tick,
            agent_id=agent_id,
            action_type="DEPOSIT_DEFAULT_FUND",
            visibility="public",
            payload={
                "amount": amount,
            },
            belief_snapshot=belief_snapshot or {},
            risk_preference=risk_preference or {},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Observation vector
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ObservationVector:
    """
    The structured observation a bank extracts from the Redis snapshot
    plus incoming events.  This is the bank's *partial* view of the system.
    """
    own_liquidity: float = 0.0
    own_capital: float = 0.0
    own_total_exposure: float = 0.0
    own_total_assets: float = 0.0

    # per-neighbor observations  {neighbor_id: {...}}
    neighbor_states: dict[int, dict[str, float]] = field(default_factory=dict)

    # aggregate system signals
    system_aggregate_liq: float = 0.0
    system_aggregate_exp: float = 0.0
    system_n_stressed: int = 0
    system_n_defaulted: int = 0
    system_n_banks: int = 0
    system_margin_rate: float = 0.10
    system_step: int = 0

    # market data (from update_market_data events)
    market_volatility: float = 0.20
    price_change_signal: float = 0.0

    # pending margin calls (from issue_margin_call events)
    pending_margin_calls: list[dict] = field(default_factory=list)
    total_margin_due: float = 0.0

    # interbank loans due this tick (for REPAY_INTERBANK_LOAN decision)
    loans_due: list[dict] = field(default_factory=list)
    total_repayment_due: float = 0.0

    # ── intent streams (from Redis public/private channels) ─────────
    # public intents from the previous tick (visible to ALL banks)
    public_intents: list[dict] = field(default_factory=list)
    # private intents directed at THIS bank (consumed on read)
    private_intents: list[dict] = field(default_factory=list)
    # derived signals from stream data
    observed_fire_sales: int = 0          # count of fire-sale intents seen
    observed_defaults: int = 0            # count of default declarations seen
    observed_sell_volume: float = 0.0     # total sell volume in public stream


# ═══════════════════════════════════════════════════════════════════════════
# BankAgent
# ═══════════════════════════════════════════════════════════════════════════

class BankAgent(ap.Agent):
    """
    Autonomous Bayesian-RL bank agent.

    Lifecycle per timestep:
        process_incoming → pull_snapshot → extract_observation →
        update_beliefs   → compute_risk  → choose_action       →
        publish_intent   → execute_intent → tick_loans
    """

    # ─────────────────────────────────────────────────── setup (once)
    def setup(self):
        p = self.model.p

        # ── core state ──────────────────────────────────────────────────
        self.liquidity: float = np.random.uniform(
            p.get("init_liquidity_lo", 50),
            p.get("init_liquidity_hi", 150),
        )
        self.capital: float = np.random.uniform(
            p.get("init_capital_lo", 80),
            p.get("init_capital_hi", 200),
        )
        self.exposure_to_neighbors: dict[int, float] = {}

        # ── asset holdings ───────────────────────────────────────────────
        self.assets: dict[str, float] = {
            "liquid_bond": np.random.uniform(
                p.get("init_liquid_bond_lo", 20),
                p.get("init_liquid_bond_hi", 80),
            ),
            "illiquid_asset": np.random.uniform(
                p.get("init_illiquid_lo", 10),
                p.get("init_illiquid_hi", 50),
            ),
        }

        # ── OTC loans (routed through CCP — batch-1 schema) ─────────────
        self.otc_loans: list[dict] = []

        # ── Interbank credit (direct — batch-2 schema) ──────────────────
        self.interbank_loans_given: list[dict] = []      # I am lender
        self.interbank_loans_received: list[dict] = []   # I am borrower

        # ── Default fund contribution ────────────────────────────────────
        self.default_fund_contribution: float = 0.0

        # ── pending margin calls from CCP ────────────────────────────────
        self.pending_margin_calls: list[dict] = []

        # ── status flags ────────────────────────────────────────────────
        self.defaulted: bool = False
        self.stressed: bool = False
        self.missed_payment: bool = False

        # ── PRIVATE Bayesian beliefs ────────────────────────────────────
        # Channel A: counterparty default risk  (one Beta per neighbor)
        self._default_beliefs: dict[int, BetaBelief] = {}

        # Channel B: network-wide liquidity stress  (single Gaussian)
        self._liquidity_stress_belief = GaussianBelief(mu=0.15, tau=2.0)

        # Channel C: expected margin call I will face  (single Gaussian)
        self._margin_call_belief = GaussianBelief(mu=5.0, tau=0.5)

        # Channel D: market volatility  (single Gaussian)
        self._volatility_belief = GaussianBelief(mu=0.20, tau=1.0)

        # ── latest observation & intent (public for metrics) ────────────
        self.last_observation: ObservationVector | None = None
        self.last_intent: ActionIntent | None = None

        # ── intent history log ──────────────────────────────────────────
        self.intent_log: list[ActionIntent] = []

        # ── bank-index (graph node id), set by model after placement ────
        self.bank_index: int = -1

        # ── cumulative stream counters (for observability / debugging) ──
        self.total_public_intents_seen: int = 0
        self.total_private_intents_seen: int = 0
        self.total_fire_sales_seen: int = 0
        self.total_defaults_seen: int = 0

    def init_neighbor_data(self):
        """Called by the model AFTER all agents are placed on the network."""
        for nbr in self.model.network.neighbors(self):
            self._default_beliefs[nbr.id] = BetaBelief(alpha=1.0, beta=9.0)
            self.exposure_to_neighbors[nbr.id] = np.random.uniform(5, 30)

    # ═══════════════════════════════════════════════════ STEP (per tick)
    def step(self):
        if self.defaulted:
            return

        tick = self.model.t

        # 0 ─ Process incoming events (margin calls, market data)
        self._process_incoming_events()

        # 1 ─ Pull latest Redis snapshot
        snapshot = self._pull_snapshot()

        # 2 ─ Extract observation vector
        obs = self._extract_observation(snapshot, tick)
        self.last_observation = obs

        # 2b ─ Accumulate stream counters (for observability)
        self.total_public_intents_seen += len(obs.public_intents)
        self.total_private_intents_seen += len(obs.private_intents)
        self.total_fire_sales_seen += obs.observed_fire_sales
        self.total_defaults_seen += obs.observed_defaults

        # 3 ─ Update Bayesian beliefs  (PRIVATE — CCP cannot see this)
        self._update_beliefs(obs)

        # 4 ─ Compute risk metrics from beliefs
        risk = self._compute_risk(obs)

        # 5 ─ Choose action via expected utility
        intent = self._choose_action(obs, risk, tick)

        # 6 ─ Store and log intent
        self.last_intent = intent
        self.intent_log.append(intent)

        # 7 ─ Publish intent to Redis (for CCP rule engine consumption)
        self._publish_intent(intent)

        # 8 ─ Execute locally (self-execute for standalone testing;
        #     in production the CCP rule engine would mediate)
        self._execute_intent(intent)

        # 9 ─ Tick OTC loan durations + process overdue interbank loans
        self._tick_otc_loans()
        self._tick_interbank_loans(tick)

        # 10 ─ Post-action housekeeping
        self.stressed = self.liquidity < self.model.p.get("stress_threshold", 30)
        if self.liquidity <= 0 or self.capital <= 0:
            self._default()

    # ═════════════════════════════════════════════ 0. INCOMING EVENTS
    def _process_incoming_events(self) -> None:
        """
        Read pending margin calls, public stream, and private stream
        from Redis.  Mirrors the central/redis_client.py channel layout:
          - public_intents        → stream:public:{tick}  (broadcast)
          - private_intents:{id}  → stream:private:{id}   (targeted)
        """
        redis = self.model.redis
        agent_id = f"bank_{self.bank_index:02d}"

        # ── margin calls from CCP (issue_margin_call schema) ────────────
        calls = redis.get_pending_margin_calls(self.bank_index)
        for call in calls:
            self.pending_margin_calls.append(call)

        # ── public intent stream (previous tick's broadcast intents) ────
        # Banks see last tick's public intents (current tick is in-flight)
        prev_tick = max(0, self.model.t - 1)
        self._public_intents_received = redis.read_public_stream(prev_tick)

        # ── private intent stream (intents addressed to THIS bank) ──────
        self._private_intents_received = redis.read_private_stream(agent_id)

        # Market data is global — read in _extract_observation.

    # ═════════════════════════════════════════════ 1. PULL SNAPSHOT
    def _pull_snapshot(self) -> dict:
        """Read the full system snapshot from Redis."""
        return self.model.redis.get_full_snapshot()

    # ═════════════════════════════════════════════ 2. EXTRACT OBSERVATION
    def _extract_observation(self, snapshot: dict, tick: int) -> ObservationVector:
        """Transform raw Redis snapshot + events into a structured observation."""
        obs = ObservationVector()

        # own state
        obs.own_liquidity = self.liquidity
        obs.own_capital = self.capital
        obs.own_total_exposure = sum(self.exposure_to_neighbors.values())
        obs.own_total_assets = sum(self.assets.values())

        # system-level aggregates
        obs.system_aggregate_liq = snapshot.get("aggregate_liq", 0.0)
        obs.system_aggregate_exp = snapshot.get("aggregate_exp", 0.0)
        obs.system_n_stressed = int(snapshot.get("n_stressed", 0))
        obs.system_n_defaulted = int(snapshot.get("n_defaulted", 0))
        obs.system_n_banks = int(snapshot.get("n_banks", 0))
        obs.system_margin_rate = snapshot.get("margin_rate", 0.10)
        obs.system_step = int(snapshot.get("step", 0))

        # per-neighbor partial state
        banks_data = snapshot.get("banks", {})
        for nbr in self.model.network.neighbors(self):
            nbr_data = banks_data.get(nbr.bank_index, {})
            obs.neighbor_states[nbr.id] = {
                "liquidity": nbr_data.get("liquidity", 0.0),
                "capital": nbr_data.get("capital", 0.0),
                "total_exposure": nbr_data.get("total_exposure", 0.0),
                "stressed": nbr_data.get("stressed", 0.0),
                "defaulted": nbr_data.get("defaulted", 0.0),
                "missed_payment": nbr_data.get("missed_payment", 0.0),
            }

        # market data from Redis  (update_market_data schema)
        mkt = self.model.redis.get_market_data()
        obs.market_volatility = mkt.get("new_volatility", 0.2)
        obs.price_change_signal = mkt.get("price_change_signal", 0.0)

        # pending margin calls  (issue_margin_call schema)
        obs.pending_margin_calls = list(self.pending_margin_calls)
        obs.total_margin_due = sum(
            c.get("payload", c).get("margin_amount", 0)
            for c in self.pending_margin_calls
        )

        # interbank loans due (REPAY_INTERBANK_LOAN decision input)
        obs.loans_due = [
            loan for loan in self.interbank_loans_received
            if loan["maturity_tick"] <= tick + 1
        ]
        obs.total_repayment_due = sum(
            loan["principal"] * (1 + loan["interest_rate"])
            for loan in obs.loans_due
        )

        # ── intent streams (public broadcast + private targeted) ────────
        pub = getattr(self, '_public_intents_received', [])
        priv = getattr(self, '_private_intents_received', [])
        obs.public_intents = pub
        obs.private_intents = priv

        # derive aggregate signals from the public stream
        for intent in pub:
            atype = intent.get("action_type", "")
            if atype == "FIRE_SALE_ASSET":
                obs.observed_fire_sales += 1
                obs.observed_sell_volume += (
                    intent.get("payload", {}).get("quantity", 0)
                )
            elif atype == "sell_asset_standard":
                obs.observed_sell_volume += (
                    intent.get("payload", {}).get("amount", 0)
                )
            elif atype == "DECLARE_DEFAULT":
                obs.observed_defaults += 1

        return obs

    # ═════════════════════════════════════════════ 3. UPDATE BELIEFS
    def _update_beliefs(self, obs: ObservationVector) -> None:
        """
        Update FOUR PRIVATE belief channels.  The CCP never sees these.

        A. Counterparty default risk  — Beta-Bernoulli per neighbor
        B. Network liquidity stress   — Gaussian (unknown mean)
        C. Expected margin call size  — Gaussian (unknown mean)
        D. Market volatility          — Gaussian (unknown mean)

        Stream signals (from public/private intents) are folded in:
          - Public fire-sales / sell orders → increase volatility belief
          - Public defaults → bump counterparty-risk beliefs upward
          - Public sell volume → increase liquidity-stress belief
          - Private margin calls already handled via margin_call channel
        """
        # ── A. counterparty default risk ────────────────────────────────
        for nbr_id, nbr_state in obs.neighbor_states.items():
            if nbr_id not in self._default_beliefs:
                self._default_beliefs[nbr_id] = BetaBelief()

            if nbr_state["defaulted"] > 0.5:
                signal = 1.0
            elif nbr_state["stressed"] > 0.5:
                signal = 0.7
            elif nbr_state["missed_payment"] > 0.5:
                signal = 0.5
            elif nbr_state["liquidity"] < 40:
                signal = 0.2
            else:
                signal = 0.0

            self._default_beliefs[nbr_id].update(signal)

        # A+ — if defaults were observed in the public stream, nudge ALL
        #      counterparty beliefs upward (contagion fear)
        if obs.observed_defaults > 0:
            contagion_signal = min(0.3, 0.15 * obs.observed_defaults)
            for belief in self._default_beliefs.values():
                belief.update(contagion_signal)

        # ── B. network liquidity stress ─────────────────────────────────
        n = max(obs.system_n_banks, 1)
        stress_frac = obs.system_n_stressed / n
        self._liquidity_stress_belief.update(stress_frac, obs_precision=2.0)

        # B+ — heavy public sell volume signals market-wide stress
        if obs.observed_sell_volume > 0:
            depth = self.model.p.get("market_depth", 200.0)
            sell_stress = min(1.0, obs.observed_sell_volume / depth)
            self._liquidity_stress_belief.update(
                sell_stress, obs_precision=1.5,
            )

        # ── C. expected margin call ─────────────────────────────────────
        estimated_call = obs.own_total_exposure * obs.system_margin_rate
        if obs.total_margin_due > 0:
            self._margin_call_belief.update(obs.total_margin_due, obs_precision=3.0)
        else:
            self._margin_call_belief.update(estimated_call, obs_precision=1.0)

        # ── D. market volatility ────────────────────────────────────────
        self._volatility_belief.update(obs.market_volatility, obs_precision=2.0)

        # D+ — fire-sale activity in the public stream implies higher
        #      realised volatility (price impact → price moves)
        if obs.observed_fire_sales > 0:
            fire_sale_vol_bump = min(
                0.30, 0.05 * obs.observed_fire_sales,
            )
            self._volatility_belief.update(
                obs.market_volatility + fire_sale_vol_bump,
                obs_precision=1.5,
            )

    # ═════════════════════════════════════════════ 4. COMPUTE RISK
    def _compute_risk(self, obs: ObservationVector) -> dict[str, float]:
        """Derive risk metrics from updated beliefs."""
        # expected loss from counterparty defaults
        expected_loss = 0.0
        for nbr_id, exposure in self.exposure_to_neighbors.items():
            pd = (self._default_beliefs[nbr_id].mean
                  if nbr_id in self._default_beliefs else 0.1)
            lgd = 0.6
            expected_loss += pd * lgd * exposure

        # liquidity buffer needed
        margin_est = self._margin_call_belief.mean
        stress_level = self._liquidity_stress_belief.mean
        vol_level = self._volatility_belief.mean
        min_liq = self.model.p.get("min_liquidity", 20)
        liquidity_shortfall = max(0.0, min_liq + margin_est - self.liquidity)

        # risk-weighted exposure
        rw_exposure = sum(
            exp * self._default_beliefs.get(nid, BetaBelief()).mean
            for nid, exp in self.exposure_to_neighbors.items()
        )

        # margin call urgency
        margin_urgency = (
            obs.total_margin_due / max(self.liquidity, 1.0)
            if self.pending_margin_calls else 0.0
        )

        # loan repayment urgency
        repay_urgency = (
            obs.total_repayment_due / max(self.liquidity, 1.0)
            if obs.loans_due else 0.0
        )

        return {
            "expected_loss": expected_loss,
            "liquidity_shortfall": liquidity_shortfall,
            "rw_exposure": rw_exposure,
            "stress_level": stress_level,
            "margin_estimate": margin_est,
            "volatility": vol_level,
            "margin_urgency": margin_urgency,
            "total_margin_due": obs.total_margin_due,
            "repay_urgency": repay_urgency,
            "total_repayment_due": obs.total_repayment_due,
        }

    # ═════════════════════════════════════════════ BELIEF / RISK HELPERS
    def _build_belief_snapshot(self) -> dict:
        """Build the belief_snapshot dict from current private beliefs."""
        snap: dict[str, Any] = {
            "market_volatility_estimate": round(self._volatility_belief.mean, 4),
            "own_liquidity_stress": round(self._liquidity_stress_belief.mean, 4),
            "expected_future_margin": round(self._margin_call_belief.mean, 2),
        }
        cpd: dict[str, float] = {}
        for nbr in self.model.network.neighbors(self):
            if nbr.id in self._default_beliefs:
                cpd[f"bank_{nbr.bank_index:02d}"] = round(
                    self._default_beliefs[nbr.id].mean, 4
                )
        if cpd:
            snap["counterparty_default_prob"] = cpd
        return snap

    def _build_risk_preference(self) -> dict:
        """Derive risk_preference from the bank's current state."""
        init_hi = max(self.model.p.get("init_liquidity_hi", 150), 1)
        liq_ratio = self.liquidity / init_hi
        liquidity_aversion = max(0.0, min(1.0, 1.0 - liq_ratio))

        if self.liquidity <= 0 or self.capital <= 0:
            default_tolerance = 0.0
        elif self.stressed:
            default_tolerance = 0.05
        else:
            default_tolerance = 0.2

        return {
            "liquidity_aversion": round(liquidity_aversion, 2),
            "default_tolerance": round(default_tolerance, 2),
        }

    # ═════════════════════════════════════════════ 5. CHOOSE ACTION
    def _choose_action(
        self, obs: ObservationVector, risk: dict[str, float], tick: int,
    ) -> ActionIntent:
        """
        Expected-utility maximisation over ELEVEN action intents.

        Batch 1:
            route_otc_proposal  — propose OTC loan via CCP
            borrow              — request liquidity from neighbor
            reduce_exposure     — cut riskiest counterparty exposure
            hoard_liquidity     — defensively reclaim liquidity
            pay_margin_call     — pay outstanding CCP margin call
            sell_asset_standard — sell liquid assets on exchange
        Batch 2:
            PROVIDE_INTERBANK_CREDIT — direct loan to neighbor
            REPAY_INTERBANK_LOAN     — repay a due loan
            FIRE_SALE_ASSET          — distressed asset dump
            DECLARE_DEFAULT          — voluntary default declaration
            DEPOSIT_DEFAULT_FUND     — contribute to systemic stability

        Returns an ActionIntent in the full JSON schema format.
        """
        agent_id = f"bank_{self.bank_index:02d}"
        el = risk["expected_loss"]
        ls = risk["liquidity_shortfall"]
        sl = risk["stress_level"]
        me = risk["margin_estimate"]
        vol = risk["volatility"]
        mu = risk["margin_urgency"]
        ru = risk["repay_urgency"]

        bs = self._build_belief_snapshot()
        rp = self._build_risk_preference()

        utilities: dict[str, float] = {}
        live_nbrs = self._live_neighbors()

        # ─── BATCH 2 — highest-priority actions first ───────────────────

        # ── REPAY_INTERBANK_LOAN: top priority when loans are due ───────
        if obs.loans_due:
            utilities["REPAY_INTERBANK_LOAN"] = 60.0 + 20.0 * ru
        else:
            utilities["REPAY_INTERBANK_LOAN"] = -999.0

        # ── DECLARE_DEFAULT: last resort (liq AND capital critically low)
        if self.liquidity < 5 and self.capital < 10:
            recovery_prob = max(0, self.liquidity / 50 + self.capital / 100)
            utilities["DECLARE_DEFAULT"] = max(
                0, (1.0 - recovery_prob) * 30.0 - 15.0
            )
        else:
            utilities["DECLARE_DEFAULT"] = -999.0

        # ── DEPOSIT_DEFAULT_FUND: well-capitalised and calm markets ─────
        if self.liquidity > 80 and sl < 0.2 and not self.stressed:
            utilities["DEPOSIT_DEFAULT_FUND"] = (
                5.0 + (self.liquidity - 80) * 0.1
            )
        else:
            utilities["DEPOSIT_DEFAULT_FUND"] = -999.0

        # ── PROVIDE_INTERBANK_CREDIT: very liquid, neighbor is worthy ───
        if live_nbrs and self.liquidity > 100:
            utilities["PROVIDE_INTERBANK_CREDIT"] = max(
                0, (self.liquidity - 100) * 0.3 - el * 0.5 - sl * 5
            )
        else:
            utilities["PROVIDE_INTERBANK_CREDIT"] = -999.0

        # ── FIRE_SALE_ASSET: desperate liquidity grab ───────────────────
        total_assets = sum(self.assets.values())
        if total_assets > 0 and (ls > 5 or mu > 0.5 or self.liquidity < 15):
            utilities["FIRE_SALE_ASSET"] = (
                5.0 * ls + 4.0 * mu + max(0, (20 - self.liquidity) * 0.8)
                + 2.0 * vol
            )
        else:
            utilities["FIRE_SALE_ASSET"] = -999.0

        # ─── BATCH 1 — existing actions ─────────────────────────────────

        # ── pay_margin_call: HIGHEST priority when pending ──────────────
        if self.pending_margin_calls:
            utilities["pay_margin_call"] = 50.0 + 20.0 * mu
        else:
            utilities["pay_margin_call"] = -999.0

        # ── sell_asset_standard: low liq and holds liquid assets ─────────
        liquid_assets = self.assets.get("liquid_bond", 0)
        if liquid_assets > 0:
            utilities["sell_asset_standard"] = (
                3.0 * ls + 2.0 * vol + 1.5 * mu
                + max(0, (30 - self.liquidity) * 0.3)
            )
        else:
            utilities["sell_asset_standard"] = -999.0

        # ── hoard_liquidity: shortfall or stress is high ────────────────
        utilities["hoard_liquidity"] = 2.0 * ls + 3.0 * sl + 1.0 * vol

        # ── reduce_exposure: expected loss is high ──────────────────────
        utilities["reduce_exposure"] = 1.5 * el + 1.0 * me + 0.5 * vol

        # ── borrow: low liquidity but capital supports it ───────────────
        cap_ratio = self.capital / max(self.liquidity, 1)
        utilities["borrow"] = (
            max(0, (40 - self.liquidity) * 0.5) if cap_ratio > 1.0 else 0.0
        )

        # ── route_otc_proposal: liquid and risk is low ──────────────────
        utilities["route_otc_proposal"] = max(
            0, (self.liquidity - 80) * 0.4 - el - sl * 10 - vol * 5
        )

        # ═══════════════════════════════ SELECT BEST ACTION ═════════════
        best = max(utilities, key=utilities.get)  # type: ignore[arg-type]

        # ── build the intent ────────────────────────────────────────────

        # ─── BATCH 2 intent builders ────────────────────────────────────

        if best == "REPAY_INTERBANK_LOAN" and obs.loans_due:
            loan = obs.loans_due[0]
            interest = loan["principal"] * loan["interest_rate"]
            return IntentFactory.repay_interbank_loan(
                tick=tick,
                agent_id=agent_id,
                loan_id=loan["loan_id"],
                principal=loan["principal"],
                interest=round(interest, 2),
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "DECLARE_DEFAULT":
            bs_decl = dict(bs)
            bs_decl["recovery_probability"] = round(
                max(0, self.liquidity / 50 + self.capital / 100), 4
            )
            return IntentFactory.declare_default(
                tick=tick,
                agent_id=agent_id,
                reason="liquidity_exhaustion",
                belief_snapshot=bs_decl,
                risk_preference={"liquidity_aversion": 1.0,
                                 "default_tolerance": 0.0},
            )

        elif best == "DEPOSIT_DEFAULT_FUND":
            deposit_amt = min(
                self.liquidity * self.model.p.get("default_fund_rate", 0.05),
                self.liquidity - 60,
            )
            deposit_amt = max(deposit_amt, 0)
            return IntentFactory.deposit_default_fund(
                tick=tick,
                agent_id=agent_id,
                amount=round(deposit_amt, 2),
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "PROVIDE_INTERBANK_CREDIT" and live_nbrs:
            # pick the healthiest-looking neighbor as borrower
            scored = [
                (nbr, self._default_beliefs.get(nbr.id, BetaBelief()).mean)
                for nbr in live_nbrs
            ]
            scored.sort(key=lambda x: x[1])
            target_nbr = scored[0][0]
            principal = min(self.liquidity * 0.10, 20)
            maturity_tick = tick + self.model.random.choice([5, 10, 15])
            return IntentFactory.provide_interbank_credit(
                tick=tick,
                agent_id=agent_id,
                borrower_bank_id=f"bank_{target_nbr.bank_index:02d}",
                principal=round(principal, 2),
                interest_rate=round(0.03 + vol * 0.04, 4),
                maturity_tick=maturity_tick,
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "FIRE_SALE_ASSET":
            # sell whichever asset has highest holdings
            if self.assets.get("liquid_bond", 0) >= \
               self.assets.get("illiquid_asset", 0):
                asset_id = "liquid_bond"
            else:
                asset_id = "illiquid_asset"
            qty = min(self.assets.get(asset_id, 0),
                      max(10, ls * 3 + vol * 25))
            max_disc = round(min(0.40, 0.10 + vol * 0.5), 2)
            return IntentFactory.fire_sale_asset(
                tick=tick,
                agent_id=agent_id,
                exchange_id="exchange_main",
                asset_id=asset_id,
                quantity=round(qty, 2),
                max_acceptable_discount=max_disc,
                belief_snapshot=bs,
                risk_preference=rp,
            )

        # ─── BATCH 1 intent builders ────────────────────────────────────

        elif best == "pay_margin_call" and self.pending_margin_calls:
            call = self.pending_margin_calls[0]
            payload = call.get("payload", call)
            payable = min(
                payload.get("margin_amount", 0),
                self.liquidity * 0.9,
            )
            return IntentFactory.pay_margin_call(
                tick=tick,
                agent_id=agent_id,
                amount=payable,
                margin_call_id=call.get("intent_id", "unknown"),
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "sell_asset_standard":
            sell_amount = min(liquid_assets, max(10, ls * 2 + vol * 20))
            return IntentFactory.sell_asset_standard(
                tick=tick,
                agent_id=agent_id,
                asset_type="liquid_bond",
                amount=sell_amount,
                order_type="market",
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "route_otc_proposal" and live_nbrs:
            target_nbr = self.model.random.choice(live_nbrs)
            amount = min(self.liquidity * 0.10, 15)
            return IntentFactory.route_otc_proposal(
                tick=tick,
                agent_id=agent_id,
                target_agent_id=f"bank_{target_nbr.bank_index:02d}",
                amount=amount,
                interest_rate=round(0.02 + vol * 0.05, 4),
                tenor_ticks=self.model.random.choice([5, 10, 15]),
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "borrow" and live_nbrs:
            target_nbr = self.model.random.choice(live_nbrs)
            amount = min(10, max(0, 40 - self.liquidity))
            return IntentFactory.borrow(
                tick=tick,
                agent_id=agent_id,
                target_agent_id=f"bank_{target_nbr.bank_index:02d}",
                amount=amount,
                belief_snapshot=bs,
                risk_preference=rp,
            )

        elif best == "reduce_exposure" and self.exposure_to_neighbors:
            riskiest = max(
                self.exposure_to_neighbors,
                key=lambda nid: self._default_beliefs.get(
                    nid, BetaBelief()
                ).mean,
            )
            amount = self.exposure_to_neighbors[riskiest] * 0.20
            return IntentFactory.reduce_exposure(
                tick=tick,
                agent_id=agent_id,
                target_neighbor_id=riskiest,
                amount=amount,
                belief_snapshot=bs,
                risk_preference=rp,
            )

        else:  # hoard_liquidity (default fallback)
            est_recovery = sum(
                v * 0.05 for v in self.exposure_to_neighbors.values()
            )
            return IntentFactory.hoard_liquidity(
                tick=tick,
                agent_id=agent_id,
                estimated_recovery=est_recovery,
                belief_snapshot=bs,
                risk_preference=rp,
            )

    # ═════════════════════════════════════════════ 6. PUBLISH INTENT
    def _publish_intent(self, intent: ActionIntent) -> None:
        """
        Publish intent to Redis with public/private stream routing.
        Mirrors the central redis_client.py visibility architecture:
          PUBLIC  → stream:public:{tick}  (all agents see it)
          PRIVATE → stream:private:{target}  (only target sees it)
        Also pushes to the legacy intents:queue for CCP consumption.
        """
        self.model.redis.route_intent(intent.to_dict())

    # ═════════════════════════════════════════════ 7. EXECUTE INTENT
    def _execute_intent(self, intent: ActionIntent) -> None:
        """
        Apply the intent locally (self-execute).
        In production the CCP rule engine would validate and execute.
        """
        cost = self.model.p.get("step_operating_cost", 2)
        self.liquidity -= cost

        # ── passive income: interest on capital and liquid bonds ─────────
        # This gives banks a positive income stream each tick so they don't
        # bleed out from operating costs alone.
        interest_on_capital = self.capital * 0.002    # 0.2% per tick
        interest_on_bonds = self.assets.get("liquid_bond", 0) * 0.001
        self.liquidity += interest_on_capital + interest_on_bonds

        self.missed_payment = False

        action = intent.action_type

        # ── batch 1 ─────────────────────────────────────────────────────
        if action == "route_otc_proposal":
            self._exec_otc_proposal(intent)
        elif action == "borrow":
            self._exec_borrow(intent)
        elif action == "reduce_exposure":
            self._exec_reduce_exposure(intent)
        elif action == "hoard_liquidity":
            self._exec_hoard()
        elif action == "pay_margin_call":
            self._exec_pay_margin_call(intent)
        elif action == "sell_asset_standard":
            self._exec_sell_asset(intent)
        # ── batch 2 ─────────────────────────────────────────────────────
        elif action == "PROVIDE_INTERBANK_CREDIT":
            self._exec_provide_interbank_credit(intent)
        elif action == "REPAY_INTERBANK_LOAN":
            self._exec_repay_interbank_loan(intent)
        elif action == "FIRE_SALE_ASSET":
            self._exec_fire_sale_asset(intent)
        elif action == "DECLARE_DEFAULT":
            self._exec_declare_default(intent)
        elif action == "DEPOSIT_DEFAULT_FUND":
            self._exec_deposit_default_fund(intent)

    # ═════════════════════════════════════════════════════════════════════
    # EXECUTION SUB-ROUTINES — Batch 1
    # ═════════════════════════════════════════════════════════════════════

    def _exec_otc_proposal(self, intent: ActionIntent) -> None:
        """Execute an OTC loan proposal (lend to target via CCP)."""
        content = intent.payload.get("encrypted_content", {})
        amount = content.get("amount", 0)
        target_id_str = intent.payload.get("target_agent_id", "")

        if amount <= 0 or not target_id_str:
            return

        target = self._resolve_target_by_str(target_id_str)
        if target is None:
            return

        self.liquidity -= amount
        target.liquidity += amount
        self.exposure_to_neighbors[target.id] = (
            self.exposure_to_neighbors.get(target.id, 0) + amount
        )

        self.otc_loans.append({
            "loan_id": intent.intent_id,
            "target": target_id_str,
            "amount": amount,
            "rate": content.get("interest_rate", 0.035),
            "remaining_ticks": content.get("tenor_ticks", 10),
        })

    def _exec_borrow(self, intent: ActionIntent) -> None:
        target_id_str = intent.payload.get("target_agent_id", "")
        lender = self._resolve_target_by_str(target_id_str)
        if lender is None:
            self.missed_payment = True
            return

        amount = min(intent.payload.get("amount", 0), lender.liquidity * 0.10)
        if amount <= 1:
            self.missed_payment = True
            return

        lender.liquidity -= amount
        self.liquidity += amount
        lender.exposure_to_neighbors[self.id] = (
            lender.exposure_to_neighbors.get(self.id, 0) + amount
        )

    def _exec_reduce_exposure(self, intent: ActionIntent) -> None:
        if not self.exposure_to_neighbors:
            return

        target_nid = intent.payload.get("target_neighbor_id")
        amount = intent.payload.get("amount", 0)

        if target_nid is not None and target_nid in self.exposure_to_neighbors:
            reduction = min(amount, self.exposure_to_neighbors[target_nid])
            self.exposure_to_neighbors[target_nid] -= reduction
            self.liquidity += reduction * 0.5
        else:
            riskiest = max(
                self.exposure_to_neighbors,
                key=lambda nid: self._default_beliefs.get(
                    nid, BetaBelief()
                ).mean,
            )
            reduction = min(amount, self.exposure_to_neighbors[riskiest])
            self.exposure_to_neighbors[riskiest] -= reduction
            self.liquidity += reduction * 0.5

    def _exec_hoard(self) -> None:
        for nbr_id in list(self.exposure_to_neighbors):
            cut = self.exposure_to_neighbors[nbr_id] * 0.05
            self.exposure_to_neighbors[nbr_id] -= cut
            self.liquidity += cut * 0.3

    def _exec_pay_margin_call(self, intent: ActionIntent) -> None:
        """Pay a CCP margin call. Deducts liquidity; removes from pending."""
        amount = intent.payload.get("amount", 0)
        margin_call_id = intent.payload.get("margin_call_id", "")

        if amount <= 0:
            self.missed_payment = True
            return

        if amount > self.liquidity:
            amount = self.liquidity * 0.9

        self.liquidity -= amount
        self.capital -= amount * 0.1

        self.pending_margin_calls = [
            c for c in self.pending_margin_calls
            if c.get("intent_id") != margin_call_id
        ]

    def _exec_sell_asset(self, intent: ActionIntent) -> None:
        """Sell assets on exchange with market-impact pricing.

        Uses redis.compute_sale_price() so that successive sellers in
        the same tick get progressively worse prices.
        """
        asset_type = intent.payload.get("asset_type", "liquid_bond")
        amount = intent.payload.get("amount", 0)

        held = self.assets.get(asset_type, 0)
        actual_sell = min(amount, held)
        if actual_sell <= 0:
            return

        vol = self._volatility_belief.mean
        pricing = self.model.redis.compute_sale_price(
            tick=self.model.t,
            asset_type=asset_type,
            amount=actual_sell,
            base_volatility=vol,
            is_fire_sale=False,
        )

        proceeds = actual_sell * pricing["price_per_unit"]
        self.assets[asset_type] = held - actual_sell
        self.liquidity += proceeds

        # record execution details on intent for audit trail
        intent.payload["execution_price"] = pricing["price_per_unit"]
        intent.payload["market_impact"] = pricing["impact_discount"]
        intent.payload["cumulative_volume"] = pricing["cumulative_volume"]

    # ═════════════════════════════════════════════════════════════════════
    # EXECUTION SUB-ROUTINES — Batch 2
    # ═════════════════════════════════════════════════════════════════════

    def _exec_provide_interbank_credit(self, intent: ActionIntent) -> None:
        """
        PROVIDE_INTERBANK_CREDIT — lend principal directly to borrower.
        Records the loan on both lender (self) and borrower sides.
        """
        p = intent.payload
        borrower_str = p.get("borrower_bank_id", "")
        principal = p.get("principal", 0)
        rate = p.get("interest_rate", 0.05)
        mat_tick = p.get("maturity_tick", self.model.t + 10)

        if principal <= 0 or not borrower_str:
            return

        borrower = self._resolve_target_by_str(borrower_str)
        if borrower is None or borrower.defaulted:
            return

        if principal > self.liquidity * 0.5:
            principal = self.liquidity * 0.5

        loan_record = {
            "loan_id": intent.intent_id,
            "borrower": borrower_str,
            "lender": f"bank_{self.bank_index:02d}",
            "principal": principal,
            "interest_rate": rate,
            "maturity_tick": mat_tick,
        }

        # lender side
        self.liquidity -= principal
        self.exposure_to_neighbors[borrower.id] = (
            self.exposure_to_neighbors.get(borrower.id, 0) + principal
        )
        self.interbank_loans_given.append(loan_record)

        # borrower side
        borrower.liquidity += principal
        borrower.interbank_loans_received.append(loan_record)

    def _exec_repay_interbank_loan(self, intent: ActionIntent) -> None:
        """
        REPAY_INTERBANK_LOAN — borrower repays principal + interest.
        Finds the lender and transfers funds; removes the loan from both sides.
        """
        p = intent.payload
        loan_id = p.get("loan_id", "")
        principal = p.get("principal", 0)
        interest = p.get("interest", 0)
        total = principal + interest

        if total <= 0 or not loan_id:
            return

        # find the matching loan on our received list
        matching = [l for l in self.interbank_loans_received
                    if l["loan_id"] == loan_id]
        if not matching:
            return
        loan = matching[0]
        lender = self._resolve_target_by_str(loan["lender"])

        # attempt repayment
        if total > self.liquidity:
            # partial pay — whatever we can
            actual = self.liquidity * 0.9
            self.missed_payment = True
        else:
            actual = total

        self.liquidity -= actual

        if lender and not lender.defaulted:
            lender.liquidity += actual
            # reduce lender's exposure
            exp = lender.exposure_to_neighbors.get(self.id, 0)
            lender.exposure_to_neighbors[self.id] = max(0, exp - principal)
            # remove from lender's given list
            lender.interbank_loans_given = [
                l for l in lender.interbank_loans_given
                if l["loan_id"] != loan_id
            ]

        # remove from our received list
        self.interbank_loans_received = [
            l for l in self.interbank_loans_received
            if l["loan_id"] != loan_id
        ]

    def _exec_fire_sale_asset(self, intent: ActionIntent) -> None:
        """
        FIRE_SALE_ASSET — sell assets at a deep discount (distressed).
        Uses market-impact pricing with is_fire_sale=True → worse
        execution than sell_asset_standard and compounds sell pressure.
        """
        p = intent.payload
        asset_id = p.get("asset_id", "liquid_bond")
        quantity = p.get("quantity", 0)

        held = self.assets.get(asset_id, 0)
        actual_sell = min(quantity, held)
        if actual_sell <= 0:
            return

        vol = self._volatility_belief.mean
        pricing = self.model.redis.compute_sale_price(
            tick=self.model.t,
            asset_type=asset_id,
            amount=actual_sell,
            base_volatility=vol,
            is_fire_sale=True,
        )

        proceeds = actual_sell * pricing["price_per_unit"]
        self.assets[asset_id] = held - actual_sell
        self.liquidity += proceeds

        # record execution details on intent for audit trail
        intent.payload["execution_price"] = pricing["price_per_unit"]
        intent.payload["market_impact"] = pricing["impact_discount"]
        intent.payload["cumulative_volume"] = pricing["cumulative_volume"]

    def _exec_declare_default(self, intent: ActionIntent) -> None:
        """
        DECLARE_DEFAULT — voluntary default.  Triggers the default cascade.
        """
        self._default()

    def _exec_deposit_default_fund(self, intent: ActionIntent) -> None:
        """
        DEPOSIT_DEFAULT_FUND — contribute liquidity to the CCP's
        centralised default fund.
        """
        amount = intent.payload.get("amount", 0)
        if amount <= 0:
            return
        actual = min(amount, self.liquidity * 0.5)
        self.liquidity -= actual
        self.default_fund_contribution += actual
        # Route deposit to CCP's centralised fund
        if hasattr(self.model, 'ccp'):
            self.model.ccp.accept_default_fund_deposit(actual)

    # ═════════════════════════════════════════════════════════════════════
    # LOAN LIFECYCLE
    # ═════════════════════════════════════════════════════════════════════

    def _tick_otc_loans(self) -> None:
        """Decrement OTC loan durations; collect interest on matured loans."""
        still_active = []
        for loan in self.otc_loans:
            loan["remaining_ticks"] -= 1
            if loan["remaining_ticks"] <= 0:
                repayment = loan["amount"] * (1 + loan["rate"])
                target = self._resolve_target_by_str(loan["target"])
                if target and not target.defaulted:
                    if target.liquidity >= repayment:
                        target.liquidity -= repayment
                        self.liquidity += repayment
                    else:
                        recovery = target.liquidity * 0.5
                        target.liquidity -= recovery
                        self.liquidity += recovery
                        self.missed_payment = True
                if target:
                    exp = self.exposure_to_neighbors.get(target.id, 0)
                    self.exposure_to_neighbors[target.id] = max(
                        0, exp - loan["amount"]
                    )
            else:
                still_active.append(loan)
        self.otc_loans = still_active

    def _tick_interbank_loans(self, tick: int) -> None:
        """
        Process overdue interbank loans.

        If a loan in interbank_loans_received is past maturity and the
        borrower didn't repay this tick via REPAY_INTERBANK_LOAN, the
        loan is auto-settled: deduct whatever the borrower can pay and
        credit the lender, then remove the loan.
        """
        grace = 2  # ticks past maturity before forced settlement
        still_active = []
        for loan in self.interbank_loans_received:
            if loan["maturity_tick"] + grace < tick:
                # overdue → forced settlement
                total = loan["principal"] * (1 + loan["interest_rate"])
                actual = min(total, self.liquidity * 0.8)
                self.liquidity -= actual
                self.missed_payment = True

                lender = self._resolve_target_by_str(loan["lender"])
                if lender and not lender.defaulted:
                    lender.liquidity += actual
                    exp = lender.exposure_to_neighbors.get(self.id, 0)
                    lender.exposure_to_neighbors[self.id] = max(
                        0, exp - loan["principal"]
                    )
                    lender.interbank_loans_given = [
                        l for l in lender.interbank_loans_given
                        if l["loan_id"] != loan["loan_id"]
                    ]
                # don't re-add — loan is settled (possibly at a loss)
            else:
                still_active.append(loan)
        self.interbank_loans_received = still_active

    # ═════════════════════════════════════════════════════════════════════
    # DEFAULT
    # ═════════════════════════════════════════════════════════════════════

    def _default(self) -> None:
        """Mark as defaulted and propagate contagion losses via CCP."""
        self.defaulted = True
        self.stressed = True

        # Let the CCP handle the default waterfall:
        #   1. Default fund absorbs what it can
        #   2. Remaining loss mutualised across ALL surviving banks
        if hasattr(self.model, 'ccp'):
            self.model.ccp.handle_bank_default(self)

        # Direct bilateral contagion to neighbors (exposure-based)
        agent_map = {a.id: a for a in self.model.banks}
        for nbr_id, exp in self.exposure_to_neighbors.items():
            nbr = agent_map.get(nbr_id)
            if nbr is not None and not nbr.defaulted:
                loss = exp * 0.3           # was 0.6 — LGD reduced to 30%
                nbr.capital -= loss
                nbr.liquidity -= loss * 0.15   # was 0.3 — less liquidity drain
        self.exposure_to_neighbors.clear()
        self.liquidity = 0
        self.capital = 0
        self.assets = {k: 0.0 for k in self.assets}
        self.otc_loans.clear()
        self.interbank_loans_given.clear()
        # NOTE: interbank_loans_received are NOT cleared — lenders still
        # hold the exposure (they will take the default loss via contagion)

    # ═════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═════════════════════════════════════════════════════════════════════

    def _resolve_target(self, bank_index: int | None):
        """Look up an agent by its bank_index (int)."""
        if bank_index is None:
            return None
        idx_map = {b.bank_index: b for b in self.model.banks}
        return idx_map.get(bank_index)

    def _resolve_target_by_str(self, agent_id_str: str):
        """
        Look up an agent by its string agent_id (e.g. 'bank_03').
        Falls back to int-based lookup.
        """
        if not agent_id_str:
            return None
        try:
            idx = int(agent_id_str.split("_")[-1])
        except (ValueError, IndexError):
            return None
        return self._resolve_target(idx)

    def _live_neighbors(self) -> list:
        return [n for n in self.model.network.neighbors(self) if not n.defaulted]

    # ═════════════════════════════════════════════════════════════════════
    # BELIEF ACCESSORS (visualization / debugging)
    # ═════════════════════════════════════════════════════════════════════

    def belief_default_prob(self, nbr_id: int) -> float:
        b = self._default_beliefs.get(nbr_id)
        return b.mean if b else 0.1

    @property
    def belief_summary(self) -> dict[str, float]:
        """Return a snapshot of all four belief channels for logging."""
        avg_pd = (
            np.mean([b.mean for b in self._default_beliefs.values()])
            if self._default_beliefs else 0.1
        )
        return {
            "avg_counterparty_pd": float(avg_pd),
            "network_stress": self._liquidity_stress_belief.mean,
            "network_stress_std": self._liquidity_stress_belief.std,
            "expected_margin_call": self._margin_call_belief.mean,
            "expected_margin_std": self._margin_call_belief.std,
            "market_volatility": self._volatility_belief.mean,
            "market_volatility_std": self._volatility_belief.std,
        }
