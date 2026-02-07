"""
FinancialNetworkModel — AgentPy model for autonomous Bayesian bank agents.

Architecture:
    Redis (fakeredis) ←→ CCPAgent ←→ BankAgents ←→ NetworkX graph

Each timestep the model:
    1. Publishes the current system state to Redis
    2. Publishes market data (simulated exchange feed — update_market_data)
    3. CCP agent: observe → update margin rate → panic check →
       compute risk scores → issue margin calls → publish → utility
    4. Each bank pulls snapshot + events, updates beliefs, emits an intent
    5. Banks self-execute intents (CCP validates via default waterfall)
    6. Model records systemic metrics + per-bank belief snapshots + CCP metrics

CCP Agent (game-theoretic):
    - Utility:  w1*(1-Panic) + w2*(Fund/SafeLimit) - w3*Defaults - w4*FireSales
    - Dynamic margin rate: base_margin + volatility * sensitivity
    - Panic mode: Total_Exposure > Default_Fund * safe_multiplier
    - Default waterfall: Fund absorbs losses → remainder mutualised
    - Private info: Total_Exposure, Member_Risk_Score (banks can't see)
    - Public info: Current_Margin_Rate (banks see this via Redis)
"""

import agentpy as ap
import networkx as nx
import numpy as np

from agents.BankAgent import BankAgent, IntentFactory
from agents.CCPAgent import CCPAgent
from agents.redis_state import RedisStateManager
from agents.neo4j_state import Neo4jStateManager

# All possible bank action types
ACTION_TYPES = [
    # batch 1
    "route_otc_proposal",
    "borrow",
    "reduce_exposure",
    "hoard_liquidity",
    "pay_margin_call",
    "sell_asset_standard",
    # batch 2
    "PROVIDE_INTERBANK_CREDIT",
    "REPAY_INTERBANK_LOAN",
    "FIRE_SALE_ASSET",
    "DECLARE_DEFAULT",
    "DEPOSIT_DEFAULT_FUND",
]


class FinancialNetworkModel(ap.Model):
    """
    Parameters (via ``p``):
        n_banks              – number of bank agents
        network_type         – 'erdos_renyi' | 'scale_free' | 'small_world'
        er_prob              – edge probability (Erdős–Rényi)
        steps                – timesteps to run
        init_liquidity_lo/hi – uniform range for initial liquidity
        init_capital_lo/hi   – uniform range for initial capital
        init_liquid_bond_lo/hi – uniform range for liquid bond holdings
        init_illiquid_lo/hi  – uniform range for illiquid asset holdings
        stress_threshold     – liquidity below this → stressed
        min_liquidity        – target used by risk calc
        step_operating_cost  – per-step cost per bank
        margin_rate          – externally-set margin rate (read by banks)
        margin_call_threshold – exposure/capital ratio triggering a call
        base_volatility      – baseline market volatility
        vol_shock_step       – timestep of volatility spike (optional)
        shock_step           – timestep of exogenous shock (optional)
        shock_intensity      – fraction of liquidity lost
        seed                 – random seed
    """

    def setup(self):
        n = self.p.get("n_banks", 8)
        net_type = self.p.get("network_type", "erdos_renyi")

        # ── build graph ──────────────────────────────────────────────────
        if net_type == "scale_free":
            G = nx.barabasi_albert_graph(n, 2, seed=self.p.get("seed", 42))
        elif net_type == "small_world":
            G = nx.watts_strogatz_graph(n, 4, 0.3, seed=self.p.get("seed", 42))
        else:
            G = nx.erdos_renyi_graph(n, self.p.get("er_prob", 0.4),
                                     seed=self.p.get("seed", 42))
        self._nx_graph = G
        self.network = ap.Network(self, G)

        # ── bank agents ──────────────────────────────────────────────────
        self.banks = ap.AgentList(self, n, BankAgent)
        nodes = list(self.network.nodes)
        self.network.add_agents(self.banks, nodes)

        for idx, bank in enumerate(self.banks):
            bank.bank_index = idx

        for bank in self.banks:
            bank.init_neighbor_data()

        # ── Redis state layer (try cloud → local → fakeredis) ─────────
        redis_ok = False
        redis_host = self.p.get("redis_host", "localhost")
        redis_port = self.p.get("redis_port", 6379)
        if not self.p.get("redis_use_fake", False):
            # attempt cloud / specified Redis
            try:
                self.redis = RedisStateManager(
                    use_fake=False,
                    host=redis_host,
                    port=redis_port,
                    db=self.p.get("redis_db", 0),
                    username=self.p.get("redis_username"),
                    password=self.p.get("redis_password"),
                )
                self.redis._r.ping()           # verify connection
                redis_ok = True
                print(f"  ✓ Redis connected: {redis_host}:{redis_port}")
            except Exception as e:
                print(f"  ✗ Cloud Redis failed ({e}), trying localhost...")
                # fallback to local Redis
                try:
                    self.redis = RedisStateManager(
                        use_fake=False, host="localhost", port=6379, db=0,
                    )
                    self.redis._r.ping()
                    redis_ok = True
                    print("  ✓ Redis connected: localhost:6379 (fallback)")
                except Exception as e2:
                    print(f"  ✗ Local Redis also failed ({e2}), using fakeredis")
        if not redis_ok:
            self.redis = RedisStateManager(use_fake=True)
            print("  ✓ Using fakeredis (in-memory)")
        self.redis.flush()                    # clean slate each run
        self.redis.set_market_depth(
            self.p.get("market_depth", 200.0)
        )

        # ── Neo4j graph persistence (optional — skip if unreachable) ──
        neo4j_uri = self.p.get("neo4j_uri", "")
        self.neo4j = None
        if neo4j_uri:
            try:
                _neo = Neo4jStateManager(
                    uri=neo4j_uri,
                    user=self.p.get("neo4j_user", "neo4j"),
                    password=self.p.get("neo4j_password", ""),
                )
                _neo._driver.verify_connectivity()  # fast check
                _neo.ensure_constraints()
                _neo.init_run(dict(self.p))
                _neo.create_bank_nodes(list(range(n)))
                _neo.create_ccp_node()
                # persist initial network edges
                edges = []
                for bank in self.banks:
                    for nbr_id, exp in bank.exposure_to_neighbors.items():
                        nbr_bank = next(
                            (b for b in self.banks if b.id == nbr_id), None
                        )
                        if nbr_bank is not None:
                            edges.append(
                                (bank.bank_index, nbr_bank.bank_index, round(exp, 2))
                            )
                if edges:
                    _neo.create_edges(edges)
                self.neo4j = _neo
                print(f"  ✓ Neo4j connected: {neo4j_uri}")
            except Exception as e:
                print(f"  ✗ Neo4j unavailable ({e}), skipping graph persistence")
                self.neo4j = None

        # ── CCP agent (game-theoretic strategic actor) ───────────────────
        self.ccp = CCPAgent(model=self, params=dict(self.p))

        # ── market state (simulated exchange) ────────────────────────────
        self._current_volatility = self.p.get("base_volatility", 0.20)
        self._price_signal = 0.0

        # ── metric accumulators ──────────────────────────────────────────
        self._ts_defaults: list[int] = []
        self._ts_active: list[int] = []
        self._ts_liquidity: list[float] = []
        self._ts_exposure: list[float] = []
        self._ts_freeze_events: list[int] = []
        self._ts_avg_pd: list[float] = []
        self._ts_avg_stress_belief: list[float] = []
        self._ts_avg_margin_belief: list[float] = []
        self._ts_avg_volatility_belief: list[float] = []
        self._ts_actions: list[dict[str, int]] = []
        self._ts_margin_calls_issued: list[int] = []
        self._ts_default_fund: list[float] = []
        self._ts_interbank_loans: list[int] = []

    # ═══════════════════════════════════════════════════════════ STEP
    def step(self):
        # 0. Apply exogenous shock (if scheduled)
        self._apply_shock()

        # 1. Publish current state to Redis (banks read this)
        self._publish_to_redis()

        # 2. Publish market data (simulated exchange — update_market_data)
        self._publish_market_data()

        # 3. CCP agent: observe → update margin rate → panic check →
        #    compute risk scores → issue margin calls → publish → utility
        n_calls = self.ccp.step(tick=self.t)

        # 4. Banks pull snapshot + events → update beliefs → emit & execute
        self.banks.step()

        # 5. Record metrics
        self._record_metrics(n_calls)

        # 6. Persist tick to Neo4j (if connected)
        self._persist_tick_to_neo4j(n_calls)

    # ─────────────────────────────────────────── publish to Redis
    def _publish_to_redis(self):
        n_stressed = sum(1 for b in self.banks if b.stressed and not b.defaulted)
        n_defaulted = sum(1 for b in self.banks if b.defaulted)
        agg_liq = sum(b.liquidity for b in self.banks if not b.defaulted)
        agg_exp = sum(
            sum(b.exposure_to_neighbors.values())
            for b in self.banks if not b.defaulted
        )

        self.redis.publish_system_state({
            "step": self.t,
            "n_banks": len(self.banks),
            "aggregate_liq": agg_liq,
            "aggregate_exp": agg_exp,
            "n_stressed": n_stressed,
            "n_defaulted": n_defaulted,
            "margin_rate": self.ccp.current_margin_rate,
        })

        for bank in self.banks:
            self.redis.publish_bank_state(bank.bank_index, {
                "liquidity": bank.liquidity,
                "capital": bank.capital,
                "total_exposure": sum(bank.exposure_to_neighbors.values()),
                "stressed": int(bank.stressed),
                "defaulted": int(bank.defaulted),
                "missed_payment": int(bank.missed_payment),
            })

    # ─────────────────────────────────────────── market data
    def _publish_market_data(self):
        """
        Simulate the exchange publishing market data each tick.
        Published via the update_market_data JSON schema.
        """
        base_vol = self.p.get("base_volatility", 0.20)
        vol_shock_step = self.p.get("vol_shock_step", None)

        # random walk with mean reversion
        noise = self.random.gauss(0, 0.02)
        self._current_volatility += (
            0.1 * (base_vol - self._current_volatility) + noise
        )
        self._current_volatility = max(0.05, min(0.80, self._current_volatility))

        # volatility spike at scheduled step
        if vol_shock_step is not None and self.t == vol_shock_step:
            self._current_volatility = min(
                0.80, self._current_volatility + 0.25
            )

        # price signal: random with slight mean reversion
        self._price_signal = self.random.gauss(-0.01, 0.03)
        self._price_signal = max(-0.15, min(0.15, self._price_signal))

        # publish via Redis using the schema's payload structure
        market_intent = IntentFactory.update_market_data(
            tick=self.t,
            agent_id="exchange_01",
            new_volatility=round(self._current_volatility, 4),
            price_change_signal=round(self._price_signal, 4),
        )
        self.redis.publish_market_data(market_intent.payload)

    # ─────────────────────────────────────────── margin calls
    # Margin calls are now issued by self.ccp (CCPAgent) in its step().
    # The old _issue_margin_calls() method has been removed.

    # ─────────────────────────────────────────── shock
    def _apply_shock(self):
        shock_step = self.p.get("shock_step", None)
        if shock_step is None or self.t != shock_step:
            return
        intensity = self.p.get("shock_intensity", 0.3)
        fraction = self.p.get("shock_fraction", 0.3)
        for bank in self.banks:
            if bank.defaulted:
                continue
            if self.random.random() < fraction:
                drain = bank.liquidity * intensity
                bank.liquidity -= drain
                bank.capital -= drain * 0.8
                bank.stressed = True

    # ─────────────────────────────────────────── neo4j tick persist
    def _persist_tick_to_neo4j(self, n_margin_calls: int = 0) -> None:
        """Write this tick's state to Neo4j (if connected)."""
        if self.neo4j is None:
            return

        tick = self.t
        self.neo4j.create_tick(tick)

        # bank states
        for bank in self.banks:
            self.neo4j.record_bank_state(tick, bank.bank_index, {
                "liquidity": round(bank.liquidity, 2),
                "capital": round(bank.capital, 2),
                "exposure": round(sum(bank.exposure_to_neighbors.values()), 2),
                "assets": round(sum(bank.assets.values()), 2),
                "stressed": bank.stressed,
                "defaulted": bank.defaulted,
            })
            # record intents emitted this tick
            if bank.last_intent and bank.last_intent.tick == tick:
                self.neo4j.record_intent(bank.last_intent.to_dict())

        # CCP state
        ccp_idx = len(self.ccp.utility_history) - 1
        if ccp_idx >= 0:
            self.neo4j.record_ccp_state(tick, {
                "utility": self.ccp.utility_history[ccp_idx],
                "margin_rate": self.ccp.margin_rate_history[ccp_idx],
                "panic_mode": self.ccp.panic_mode_history[ccp_idx],
                "default_fund": self.ccp.default_fund_history[ccp_idx],
                "fire_sale_volume": self.ccp.fire_sale_history[ccp_idx],
            })

        # record defaults that happened this tick
        for bank in self.banks:
            if bank.defaulted and bank.last_intent and \
               bank.last_intent.tick == tick and \
               bank.last_intent.action_type == "DECLARE_DEFAULT":
                self.neo4j.record_default(tick, bank.bank_index)

    # ─────────────────────────────────────────── metrics
    def _record_metrics(self, n_margin_calls: int = 0):
        n_defaulted = sum(1 for b in self.banks if b.defaulted)
        n_active = len(self.banks) - n_defaulted
        total_liq = sum(b.liquidity for b in self.banks if not b.defaulted)
        total_exp = sum(
            sum(b.exposure_to_neighbors.values())
            for b in self.banks if not b.defaulted
        )
        n_stressed = sum(
            1 for b in self.banks if b.stressed and not b.defaulted
        )
        freeze = 1 if (n_active > 0 and n_stressed / n_active > 0.5) else 0

        # belief averages (only from living banks)
        live = [b for b in self.banks if not b.defaulted]
        if live:
            avg_pd = float(np.mean([
                b.belief_summary["avg_counterparty_pd"] for b in live
            ]))
            avg_sl = float(np.mean([
                b.belief_summary["network_stress"] for b in live
            ]))
            avg_mc = float(np.mean([
                b.belief_summary["expected_margin_call"] for b in live
            ]))
            avg_vol = float(np.mean([
                b.belief_summary["market_volatility"] for b in live
            ]))
        else:
            avg_pd, avg_sl, avg_mc, avg_vol = 0.0, 0.0, 0.0, 0.0

        # action distribution
        action_counts = {a: 0 for a in ACTION_TYPES}
        for b in live:
            if b.last_intent:
                at = b.last_intent.action_type
                action_counts[at] = action_counts.get(at, 0) + 1

        self._ts_defaults.append(n_defaulted)
        self._ts_active.append(n_active)
        self._ts_liquidity.append(total_liq)
        self._ts_exposure.append(total_exp)
        self._ts_freeze_events.append(freeze)
        self._ts_avg_pd.append(avg_pd)
        self._ts_avg_stress_belief.append(avg_sl)
        self._ts_avg_margin_belief.append(avg_mc)
        self._ts_avg_volatility_belief.append(avg_vol)
        self._ts_actions.append(action_counts)
        self._ts_margin_calls_issued.append(n_margin_calls)

        # default-fund: CCP centralised fund (already includes bank deposits)
        total_ib_loans = sum(
            len(b.interbank_loans_given) for b in live
        )
        self._ts_default_fund.append(self.ccp.default_fund)
        self._ts_interbank_loans.append(total_ib_loans)

        self.record("n_defaulted", n_defaulted)
        self.record("n_active", n_active)
        self.record("total_liquidity", total_liq)
        self.record("total_exposure", total_exp)
        self.record("liquidity_freeze", freeze)

    # ─────────────────────────────────────────── end
    def end(self):
        self.report("total_defaults",
                     self._ts_defaults[-1] if self._ts_defaults else 0)
        self.report("total_freeze_events", sum(self._ts_freeze_events))
        # Finalize Neo4j run with summary
        if self.neo4j is not None:
            m = self.metrics
            self.neo4j.finalize_run({
                "final_defaults": m["defaults"][-1] if m["defaults"] else 0,
                "final_active": m["active"][-1] if m["active"] else 0,
                "final_liquidity": m["liquidity"][-1] if m["liquidity"] else 0,
                "freeze_events": sum(m["freeze_events"]),
                "ccp_final_utility": m["ccp_utility"][-1] if m["ccp_utility"] else 0,
                "ccp_final_fund": m["ccp_default_fund"][-1] if m["ccp_default_fund"] else 0,
                "total_margin_calls": sum(m["margin_calls_issued"]),
            })
            self.neo4j.close()
        # Clean up Redis keys so the next run starts fresh
        self.redis.flush()

    # ─────────────────────────────────────────── accessors
    @property
    def metrics(self) -> dict:
        base = {
            "defaults": list(self._ts_defaults),
            "active": list(self._ts_active),
            "liquidity": list(self._ts_liquidity),
            "exposure": list(self._ts_exposure),
            "freeze_events": list(self._ts_freeze_events),
            "avg_pd": list(self._ts_avg_pd),
            "avg_stress_belief": list(self._ts_avg_stress_belief),
            "avg_margin_belief": list(self._ts_avg_margin_belief),
            "avg_volatility_belief": list(self._ts_avg_volatility_belief),
            "actions": list(self._ts_actions),
            "margin_calls_issued": list(self._ts_margin_calls_issued),
            "default_fund": list(self._ts_default_fund),
            "interbank_loans": list(self._ts_interbank_loans),
        }
        # Merge CCP agent metrics
        base.update(self.ccp.metrics)
        return base
