"""
Neo4jStateManager — Persists simulation graph structure, agent states,
intents, and CCP actions into a Neo4j Aura graph database.

Node labels:
    (:Bank       {bank_id, run_id})
    (:CCP        {ccp_id, run_id})
    (:SimRun     {run_id, timestamp, params_json})
    (:Tick       {tick, run_id})
    (:Intent     {intent_id, tick, action_type, agent_id, ...})

Relationship types:
    (:Bank)-[:CONNECTED_TO   {exposure}]->(:Bank)
    (:Bank)-[:EMITTED        {tick}]->(:Intent)
    (:Bank)-[:STATE_AT       {tick, liquidity, capital, ...}]->(:Tick)
    (:CCP)-[:MARGIN_CALL     {tick, amount}]->(:Bank)
    (:CCP)-[:STATE_AT        {tick, utility, margin_rate, ...}]->(:Tick)
    (:Bank)-[:DEFAULTED_AT   {tick}]->(:Tick)
    (:SimRun)-[:HAS_TICK]->(:Tick)
    (:SimRun)-[:HAS_BANK]->(:Bank)
    (:SimRun)-[:HAS_CCP]->(:CCP)

Credentials are loaded from environment variables (see .env):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase


class Neo4jStateManager:
    """Manages Neo4j connection and writes simulation data as a graph."""

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._run_id: str = ""

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()

    def verify_connectivity(self) -> bool:
        """Test that the connection works.  Returns True on success."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"  [Neo4j] connectivity check failed: {e}")
            return False

    # ── schema / constraints ─────────────────────────────────────────────

    def ensure_constraints(self) -> None:
        """Create uniqueness constraints (idempotent)."""
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:SimRun)  REQUIRE r.run_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Intent)  REQUIRE i.intent_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Bank)    REQUIRE (b.bank_id, b.run_id) IS UNIQUE",
        ]
        with self._driver.session() as s:
            for stmt in stmts:
                try:
                    s.run(stmt)
                except Exception:
                    pass  # older Neo4j versions may not support composite

    # ── simulation run lifecycle ─────────────────────────────────────────

    def init_run(self, params: dict) -> str:
        """
        Create a (:SimRun) node and return its run_id.
        Must be called once at the start of each simulation.
        """
        self._run_id = str(uuid.uuid4())[:8]
        with self._driver.session() as s:
            s.run(
                """
                CREATE (r:SimRun {
                    run_id:     $run_id,
                    timestamp:  $ts,
                    n_banks:    $n_banks,
                    steps:      $steps,
                    seed:       $seed,
                    params_json: $params_json
                })
                """,
                run_id=self._run_id,
                ts=datetime.now(timezone.utc).isoformat(),
                n_banks=params.get("n_banks", 0),
                steps=params.get("steps", 0),
                seed=params.get("seed", 0),
                params_json=json.dumps(
                    {k: v for k, v in params.items()
                     if not k.startswith(("neo4j_", "redis_"))},
                    default=str,
                ),
            )
        return self._run_id

    def create_bank_nodes(self, bank_indices: list[int]) -> None:
        """Create (:Bank) nodes and link them to the (:SimRun)."""
        with self._driver.session() as s:
            s.run(
                """
                UNWIND $banks AS bid
                MATCH (r:SimRun {run_id: $run_id})
                CREATE (b:Bank {bank_id: bid, run_id: $run_id})
                CREATE (r)-[:HAS_BANK]->(b)
                """,
                banks=bank_indices,
                run_id=self._run_id,
            )

    def create_ccp_node(self) -> None:
        """Create a (:CCP) node linked to the (:SimRun)."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (r:SimRun {run_id: $run_id})
                CREATE (c:CCP {ccp_id: 'ccp_01', run_id: $run_id})
                CREATE (r)-[:HAS_CCP]->(c)
                """,
                run_id=self._run_id,
            )

    def create_edges(self, edges: list[tuple[int, int, float]]) -> None:
        """
        Create (:Bank)-[:CONNECTED_TO]->(:Bank) relationships.

        edges: list of (bank_i, bank_j, exposure)
        """
        with self._driver.session() as s:
            s.run(
                """
                UNWIND $edges AS e
                MATCH (a:Bank {bank_id: e[0], run_id: $run_id})
                MATCH (b:Bank {bank_id: e[1], run_id: $run_id})
                CREATE (a)-[:CONNECTED_TO {exposure: e[2]}]->(b)
                """,
                edges=edges,
                run_id=self._run_id,
            )

    # ── per-tick writes ──────────────────────────────────────────────────

    def create_tick(self, tick: int) -> None:
        """Create a (:Tick) node linked to the (:SimRun)."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (r:SimRun {run_id: $run_id})
                CREATE (t:Tick {tick: $tick, run_id: $run_id})
                CREATE (r)-[:HAS_TICK]->(t)
                """,
                tick=tick,
                run_id=self._run_id,
            )

    def record_bank_state(
        self, tick: int, bank_id: int, state: dict[str, Any],
    ) -> None:
        """Create (:Bank)-[:STATE_AT {…}]->(:Tick) relationship."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (b:Bank {bank_id: $bid, run_id: $run_id})
                MATCH (t:Tick {tick: $tick, run_id: $run_id})
                CREATE (b)-[:STATE_AT {
                    tick:       $tick,
                    liquidity:  $liq,
                    capital:    $cap,
                    exposure:   $exp,
                    assets:     $assets,
                    stressed:   $stressed,
                    defaulted:  $defaulted
                }]->(t)
                """,
                bid=bank_id,
                tick=tick,
                run_id=self._run_id,
                liq=state.get("liquidity", 0.0),
                cap=state.get("capital", 0.0),
                exp=state.get("exposure", 0.0),
                assets=state.get("assets", 0.0),
                stressed=state.get("stressed", False),
                defaulted=state.get("defaulted", False),
            )

    def record_ccp_state(self, tick: int, state: dict[str, Any]) -> None:
        """Create (:CCP)-[:STATE_AT {…}]->(:Tick) relationship."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (c:CCP {ccp_id: 'ccp_01', run_id: $run_id})
                MATCH (t:Tick {tick: $tick, run_id: $run_id})
                CREATE (c)-[:STATE_AT {
                    tick:         $tick,
                    utility:      $utility,
                    margin_rate:  $mr,
                    panic_mode:   $panic,
                    default_fund: $fund,
                    fire_sale_vol: $fsv
                }]->(t)
                """,
                tick=tick,
                run_id=self._run_id,
                utility=state.get("utility", 0.0),
                mr=state.get("margin_rate", 0.0),
                panic=state.get("panic_mode", False),
                fund=state.get("default_fund", 0.0),
                fsv=state.get("fire_sale_volume", 0.0),
            )

    def record_intent(self, intent_dict: dict) -> None:
        """
        Create an (:Intent) node and link it to its emitting (:Bank).
        """
        with self._driver.session() as s:
            s.run(
                """
                MATCH (t:Tick {tick: $tick, run_id: $run_id})
                CREATE (i:Intent {
                    intent_id:   $iid,
                    tick:        $tick,
                    agent_id:    $aid,
                    action_type: $atype,
                    visibility:  $vis,
                    payload_json: $pjson,
                    run_id:      $run_id
                })
                CREATE (i)-[:AT_TICK]->(t)
                WITH i
                OPTIONAL MATCH (b:Bank {bank_id: $bid, run_id: $run_id})
                FOREACH (_ IN CASE WHEN b IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (b)-[:EMITTED {tick: $tick}]->(i)
                )
                """,
                iid=intent_dict.get("intent_id", ""),
                tick=intent_dict.get("tick", 0),
                aid=intent_dict.get("agent_id", ""),
                atype=intent_dict.get("action_type", ""),
                vis=intent_dict.get("visibility", "private"),
                pjson=json.dumps(intent_dict.get("payload", {})),
                run_id=self._run_id,
                bid=self._parse_bank_id(intent_dict.get("agent_id", "")),
            )

    def record_default(self, tick: int, bank_id: int) -> None:
        """Create (:Bank)-[:DEFAULTED_AT]->(:Tick)."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (b:Bank {bank_id: $bid, run_id: $run_id})
                MATCH (t:Tick {tick: $tick, run_id: $run_id})
                CREATE (b)-[:DEFAULTED_AT {tick: $tick}]->(t)
                """,
                bid=bank_id,
                tick=tick,
                run_id=self._run_id,
            )

    def record_margin_call(
        self, tick: int, target_bank_id: int, amount: float,
    ) -> None:
        """Create (:CCP)-[:MARGIN_CALL]->(:Bank)."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (c:CCP {ccp_id: 'ccp_01', run_id: $run_id})
                MATCH (b:Bank {bank_id: $bid, run_id: $run_id})
                CREATE (c)-[:MARGIN_CALL {tick: $tick, amount: $amt}]->(b)
                """,
                bid=target_bank_id,
                tick=tick,
                amt=amount,
                run_id=self._run_id,
            )

    # ── summary write (at end of simulation) ─────────────────────────────

    def finalize_run(self, summary: dict[str, Any]) -> None:
        """Update the SimRun node with final summary metrics."""
        with self._driver.session() as s:
            s.run(
                """
                MATCH (r:SimRun {run_id: $run_id})
                SET r.final_defaults      = $defaults,
                    r.final_active         = $active,
                    r.final_liquidity      = $liq,
                    r.freeze_events        = $freeze,
                    r.ccp_final_utility    = $ccp_util,
                    r.ccp_final_fund       = $ccp_fund,
                    r.total_margin_calls   = $mc,
                    r.completed            = true
                """,
                run_id=self._run_id,
                defaults=summary.get("final_defaults", 0),
                active=summary.get("final_active", 0),
                liq=summary.get("final_liquidity", 0.0),
                freeze=summary.get("freeze_events", 0),
                ccp_util=summary.get("ccp_final_utility", 0.0),
                ccp_fund=summary.get("ccp_final_fund", 0.0),
                mc=summary.get("total_margin_calls", 0),
            )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_bank_id(agent_id_str: str) -> int:
        """Extract numeric bank id from 'bank_03' → 3."""
        try:
            return int(agent_id_str.split("_")[-1])
        except (ValueError, IndexError):
            return -1
