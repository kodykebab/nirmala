#!/usr/bin/env python3
"""
run_simulation.py — Entry point for the Bayesian bank-agent simulation.

Banks are autonomous Bayesian-RL agents that pull system snapshots from
Redis, consume market data (update_market_data) and margin calls
(issue_margin_call), update private beliefs, and emit action intents
matching 11 JSON schemas across two batches:
  Batch 1: route_otc_proposal, pay_margin_call, sell_asset_standard,
           reduce_exposure, hoard_liquidity, borrow
  Batch 2: PROVIDE_INTERBANK_CREDIT, REPAY_INTERBANK_LOAN,
           FIRE_SALE_ASSET, DECLARE_DEFAULT, DEPOSIT_DEFAULT_FUND
"""

import os

from dotenv import load_dotenv
load_dotenv()  # pick up any local .env (Neo4j, etc.)

from model import FinancialNetworkModel
from visualization import (
    plot_bank_status,
    plot_aggregate_liquidity,
    plot_belief_evolution,
    plot_action_distribution,
    plot_network_snapshot,
    plot_ccp_dashboard,
)


def main():
    # ── parameters ──────────────────────────────────────────────────────
    parameters = {
        "n_banks": 10,
        "network_type": "erdos_renyi",
        "er_prob": 0.25,
        "steps": 25000,

        # bank initial state ranges
        "init_liquidity_lo": 150,        # moderate starting runway
        "init_liquidity_hi": 350,
        "init_capital_lo": 200,
        "init_capital_hi": 500,
        "init_liquid_bond_lo": 80,
        "init_liquid_bond_hi": 200,
        "init_illiquid_lo": 20,
        "init_illiquid_hi": 60,

        # risk / stress params
        "stress_threshold": 15,
        "min_liquidity": 8,
        "step_operating_cost": 0.2,

        # CCP params (simulated)
        "margin_rate": 0.4,
        "margin_call_threshold": 0.7,
        "default_fund_rate": 0.01,      # was 0.05 — less upfront drain

        # CCP agent params (game-theoretic)
        "ccp_initial_default_fund": 200.0,
        "ccp_base_margin": 0.03,
        "ccp_margin_sensitivity": 0.005,
        "ccp_safe_multiplier": 10.0,
        "ccp_w1": 0.4,
        "ccp_w2": 0.3,
        "ccp_w3": 0.2,
        "ccp_w4": 0.1,

        # Redis connection (local only)
        "redis_use_fake": False,
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 0,

        # Neo4j Aura (loaded from .env)
        "neo4j_uri": os.getenv("NEO4J_URI", ""),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", ""),

        # market params (simulated exchange)
        "base_volatility": 0.12,
        "vol_shock_step": 15,
        "market_depth": 750.0,

        # exogenous shock
        "shock_step": 100,
        "shock_intensity": 0.18,         # meaningful shock
        "shock_fraction": 0.3,           # half the banks hit

        "seed": 99,
    }

    # ── run ──────────────────────────────────────────────────────────────
    model = FinancialNetworkModel(parameters)
    model.run()

    m = model.metrics

    # ── summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BANK-AGENT SIMULATION SUMMARY  (11-schema intents)")
    print("=" * 70)
    print(f"  Redis                : {parameters.get('redis_host', 'localhost')}"
          f":{parameters.get('redis_port', 6379)}")
    neo_uri = parameters.get("neo4j_uri", "")
    if neo_uri and model.neo4j:
        print(f"  Neo4j                : {neo_uri}")
        print(f"  Neo4j run_id         : {model.neo4j._run_id}")
    print(f"  Banks                : {parameters['n_banks']}")
    print(f"  Timesteps            : {parameters['steps']}")
    print(f"  Final defaults       : {m['defaults'][-1]}")
    print(f"  Final active         : {m['active'][-1]}")
    print(f"  Freeze events        : {sum(m['freeze_events'])}")
    print(f"  Final liquidity      : {m['liquidity'][-1]:.1f}")
    print(f"  Final exposure       : {m['exposure'][-1]:.1f}")
    print(f"  Avg counterparty PD  : {m['avg_pd'][-1]:.4f}")
    print(f"  Avg stress belief    : {m['avg_stress_belief'][-1]:.4f}")
    print(f"  Avg margin estimate  : {m['avg_margin_belief'][-1]:.2f}")
    print(f"  Avg volatility belief: {m['avg_volatility_belief'][-1]:.4f}")
    total_mc = sum(m["margin_calls_issued"])
    print(f"  Total margin calls   : {total_mc}")
    print(f"  Default fund total   : {m['default_fund'][-1]:.1f}")
    print(f"  Active IB loans      : {m['interbank_loans'][-1]}")

    # CCP metrics
    print(f"  CCP final utility    : {m['ccp_utility'][-1]:.4f}")
    print(f"  CCP final margin rate: {m['ccp_margin_rate'][-1]:.4f}")
    print(f"  CCP panic mode ticks : {sum(m['ccp_panic_mode'])}")
    print(f"  CCP default fund     : {m['ccp_default_fund'][-1]:.1f}")
    print(f"  CCP fire-sale volume : {m['ccp_fire_sale_volume'][-1]:.1f}")
    print("=" * 70 + "\n")

    # ── per-bank detail ──────────────────────────────────────────────────
    hdr = (
        f"{'Bank':>6} {'Liq':>8} {'Cap':>8} {'Exp':>8} {'Assets':>8} "
        f"{'DFund':>7} {'IB':>4} "
        f"{'AvgPD':>8} {'Vol':>7} "
        f"{'Last Action':>25} {'Status':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for b in model.banks:
        exp = sum(b.exposure_to_neighbors.values())
        total_assets = sum(b.assets.values())
        bs = b.belief_summary
        last = b.last_intent.action_type if b.last_intent else "—"
        status = ("DEFAULT" if b.defaulted
                  else ("STRESSED" if b.stressed else "OK"))
        n_ib = len(b.interbank_loans_given)
        print(
            f"  B{b.bank_index:<4} {b.liquidity:>8.1f} {b.capital:>8.1f} "
            f"{exp:>8.1f} {total_assets:>8.1f} "
            f"{b.default_fund_contribution:>7.1f} {n_ib:>4} "
            f"{bs['avg_counterparty_pd']:>8.4f} "
            f"{bs['market_volatility']:>7.4f} "
            f"{last:>25} {status:>10}"
        )
    print()

    # ── plots ────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)

    plot_bank_status(m, save_path=os.path.join(out, "bank_status.png"))
    plot_aggregate_liquidity(m, save_path=os.path.join(out, "liquidity.png"))
    plot_belief_evolution(m, save_path=os.path.join(out, "belief_evolution.png"))
    plot_action_distribution(m,
                             save_path=os.path.join(out, "action_distribution.png"))
    plot_network_snapshot(model,
                          save_path=os.path.join(out, "network_snapshot.png"))
    plot_ccp_dashboard(m, save_path=os.path.join(out, "ccp_dashboard.png"))
    print(f"  Plots saved to {out}/\n")


if __name__ == "__main__":
    main()
