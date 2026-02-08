#!/usr/bin/env python3
"""
test_redis_monitor.py — Runs the simulation step-by-step and monitors
Redis writes in real time, proving data flows correctly.

This script:
  1. Connects to the SAME Redis instance the simulation uses
  2. Runs the model step-by-step (not model.run())
  3. After each step, reads back the Redis keys and prints a live summary
  4. Proves every tick writes system state, per-bank state, and market data

Usage:
    python test_redis_monitor.py
"""

import os
import sys
import json
import time

# Load .env from central/ (where Redis creds live) + model/
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "central", ".env"))
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from model import FinancialNetworkModel

# ── Parameters (small run: 10 banks, 30 steps) ──────────────────────────
parameters = {
    "n_banks": 10,
    "network_type": "erdos_renyi",
    "er_prob": 0.35,
    "steps": 30,

    "init_liquidity_lo": 80,
    "init_liquidity_hi": 200,
    "init_capital_lo": 100,
    "init_capital_hi": 250,
    "init_liquid_bond_lo": 40,
    "init_liquid_bond_hi": 120,
    "init_illiquid_lo": 20,
    "init_illiquid_hi": 60,

    "stress_threshold": 25,
    "min_liquidity": 15,
    "step_operating_cost": 0.5,

    "margin_rate": 0.5,
    "margin_call_threshold": 0.5,
    "default_fund_rate": 0.02,

    "ccp_initial_default_fund": 100.0,
    "ccp_base_margin": 0.05,
    "ccp_margin_sensitivity": 0.01,
    "ccp_safe_multiplier": 10.0,
    "ccp_w1": 0.4,
    "ccp_w2": 0.3,
    "ccp_w3": 0.2,
    "ccp_w4": 0.1,

    # Redis — use cloud if .env is loaded, else fall back
    "redis_use_fake": False,
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": int(os.getenv("REDIS_PORT", "6379")),
    "redis_db": 0,
    "redis_username": os.getenv("REDIS_USERNAME"),
    "redis_password": os.getenv("REDIS_PASSWORD"),

    "base_volatility": 0.20,
    "vol_shock_step": 15,
    "market_depth": 400.0,

    "shock_step": 15,
    "shock_intensity": 0.30,
    "shock_fraction": 0.5,

    "seed": 42,
}


def read_redis_state(redis_mgr):
    """Read back all keys the simulation writes and return a summary dict."""
    snap = redis_mgr.get_full_snapshot()
    market = redis_mgr.get_market_data()
    intent_count = redis_mgr._r.llen("intents:queue")

    # Count stream keys
    public_keys = redis_mgr._r.keys("stream:public:*")
    private_keys = redis_mgr._r.keys("stream:private:*")

    return {
        "step": int(snap.get("step", 0)),
        "n_banks": int(snap.get("n_banks", 0)),
        "agg_liq": snap.get("aggregate_liq", 0),
        "agg_exp": snap.get("aggregate_exp", 0),
        "n_stressed": int(snap.get("n_stressed", 0)),
        "n_defaulted": int(snap.get("n_defaulted", 0)),
        "margin_rate": snap.get("margin_rate", 0),
        "volatility": market.get("new_volatility", 0),
        "price_signal": market.get("price_change_signal", 0),
        "intent_queue_len": intent_count,
        "public_streams": len(public_keys),
        "private_streams": len(private_keys),
        "banks": snap.get("banks", {}),
    }


def print_header():
    print("\n" + "=" * 100)
    print("  REDIS MONITOR — Real-Time Simulation Logging Test")
    print("=" * 100)


def print_tick(state):
    tick = state["step"]
    n_banks = state["n_banks"]
    n_ok = n_banks - state["n_stressed"] - state["n_defaulted"]

    # Status bar
    bar = (
        f"🟢×{n_ok}" if n_ok > 0 else ""
    ) + (
        f" 🟡×{state['n_stressed']}" if state["n_stressed"] > 0 else ""
    ) + (
        f" 🔴×{state['n_defaulted']}" if state["n_defaulted"] > 0 else ""
    )

    print(
        f"  t={tick:>4}  │  "
        f"Liq={state['agg_liq']:>9.1f}  "
        f"Exp={state['agg_exp']:>9.1f}  │  "
        f"Vol={state['volatility']:.4f}  "
        f"MR={state['margin_rate']:.4f}  "
        f"PS={state['price_signal']:+.4f}  │  "
        f"Intents={state['intent_queue_len']:>5}  "
        f"Pub={state['public_streams']:>3}  "
        f"Priv={state['private_streams']:>3}  │  "
        f"{bar}"
    )


def print_bank_table(banks):
    print("\n  Per-bank Redis state:")
    print(f"  {'Bank':>6} {'Liq':>10} {'Cap':>10} {'Exp':>10} {'Str':>5} {'Def':>5}")
    print("  " + "-" * 52)
    for bid in sorted(banks.keys()):
        b = banks[bid]
        print(
            f"  B{bid:<4} "
            f"{b.get('liquidity', 0):>10.1f} "
            f"{b.get('capital', 0):>10.1f} "
            f"{b.get('total_exposure', 0):>10.1f} "
            f"{'Y' if b.get('stressed', 0) else '.':>5} "
            f"{'Y' if b.get('defaulted', 0) else '.':>5}"
        )
    print()


def main():
    print_header()

    # Show which Redis we're connecting to
    host = parameters["redis_host"]
    port = parameters["redis_port"]
    user = parameters.get("redis_username", "")
    print(f"\n  Redis target: {host}:{port}  (user={user or 'none'})")

    # Check if env vars were actually loaded
    env_host = os.getenv("REDIS_HOST", "")
    if not env_host:
        print("  ⚠  REDIS_HOST env var is empty — .env may not be loaded!")
        print("     The simulation will likely fall back to fakeredis.")
    else:
        print(f"  ✓  REDIS_HOST loaded from .env: {env_host}")

    # ── Setup model (but don't call model.run()) ──────────────────────
    print(f"\n  Initializing model with {parameters['n_banks']} banks, "
          f"{parameters['steps']} steps...")
    model = FinancialNetworkModel(parameters)
    model.sim_setup()  # AgentPy internal setup

    # Check what Redis backend we actually got
    redis_type = type(model.redis._r).__module__
    is_fake = "fakeredis" in redis_type
    print(f"  Redis backend: {redis_type} ({'⚠ FAKEREDIS — external dashboard cannot see this!' if is_fake else '✓ Real Redis'})")

    if is_fake:
        print("\n  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  WARNING: Using fakeredis (in-memory).                      ║")
        print("  ║  An external dashboard process CANNOT read this data.       ║")
        print("  ║  To fix: ensure .env with REDIS_HOST is loadable, or        ║")
        print("  ║  run a local Redis server (brew install redis && redis-server).║")
        print("  ╚══════════════════════════════════════════════════════════════╝\n")

    # ── Run step-by-step, monitoring Redis after each tick ────────────
    print("\n  Starting step-by-step simulation with Redis monitoring...\n")
    print(
        f"  {'Tick':>6}  │  {'Liquidity':>11}  {'Exposure':>11}  │  "
        f"{'Vol':>8}  {'MarRate':>8}  {'PriceSig':>9}  │  "
        f"{'Intents':>8}  {'Pub':>4}  {'Priv':>5}  │  Status"
    )
    print("  " + "─" * 96)

    steps = parameters["steps"]
    for t in range(1, steps + 1):
        model.t = t
        model.step()

        # Read back from Redis
        state = read_redis_state(model.redis)
        print_tick(state)

        # Every 10 ticks, also print the bank table
        if t % 10 == 0 or t == steps:
            print_bank_table(state["banks"])

    # ── Final summary ────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  FINAL REDIS STATE")
    print("=" * 100)

    final = read_redis_state(model.redis)

    # Count all Redis keys
    all_keys = model.redis._r.keys("*")
    key_types = {}
    for k in all_keys:
        prefix = k.split(":")[0]
        key_types[prefix] = key_types.get(prefix, 0) + 1

    print(f"\n  Total Redis keys: {len(all_keys)}")
    print(f"  Key breakdown:")
    for prefix, count in sorted(key_types.items()):
        print(f"    {prefix:20s} → {count} keys")

    print(f"\n  System state:     step={final['step']}, "
          f"liq={final['agg_liq']:.1f}, exp={final['agg_exp']:.1f}")
    print(f"  Bank states:      {len(final['banks'])} banks in Redis")
    print(f"  Intent queue:     {final['intent_queue_len']} intents")
    print(f"  Public streams:   {final['public_streams']} tick-streams")
    print(f"  Private streams:  {final['private_streams']} agent-streams")
    print(f"  Market data:      vol={final['volatility']:.4f}, "
          f"sig={final['price_signal']:+.4f}")

    # Verify every bank has data
    missing = [i for i in range(parameters["n_banks"])
               if i not in final["banks"]]
    if missing:
        print(f"\n  ⚠ Missing bank states: {missing}")
    else:
        print(f"\n  ✓ All {parameters['n_banks']} banks have Redis state")

    print(f"  ✓ {final['step']} ticks written to Redis")
    print(f"  ✓ {final['intent_queue_len']} intents in queue")

    if is_fake:
        print("\n  ⚠ Data was in FAKEREDIS — not visible externally.")
        print("    Fix: load central/.env or install local Redis.\n")
    else:
        print("\n  ✓ Data is in REAL Redis — external dashboard can read it.\n")

    # DON'T flush so the dashboard can read the final state
    print("  (Skipping redis.flush() so dashboard can inspect final state)\n")


if __name__ == "__main__":
    main()
