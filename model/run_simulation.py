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

from model import FinancialNetworkModel
from visualization import (
    plot_bank_status,
    plot_aggregate_liquidity,
    plot_belief_evolution,
    plot_action_distribution,
    plot_network_snapshot,
)


def main():
    # ── parameters ──────────────────────────────────────────────────────
    parameters = {
        "n_banks": 10,
        "network_type": "erdos_renyi",
        "er_prob": 0.35,
        "steps": 50,

        # bank initial state ranges
        "init_liquidity_lo": 50,
        "init_liquidity_hi": 140,
        "init_capital_lo": 60,
        "init_capital_hi": 180,
        "init_liquid_bond_lo": 20,
        "init_liquid_bond_hi": 80,
        "init_illiquid_lo": 10,
        "init_illiquid_hi": 50,

        # risk / stress params
        "stress_threshold": 30,
        "min_liquidity": 25,
        "step_operating_cost": 2.0,

        # CCP params (simulated)
        "margin_rate": 0.10,
        "margin_call_threshold": 0.5,
        "default_fund_rate": 0.05,   # fraction of liq deposited per DEPOSIT_DEFAULT_FUND

        # CCP agent params (game-theoretic)
        "ccp_initial_default_fund": 100.0,   # starting centralised fund
        "ccp_base_margin": 0.05,             # base margin rate
        "ccp_margin_sensitivity": 0.01,      # volatility * this added to margin
        "ccp_safe_multiplier": 10.0,         # panic when exposure > fund * this
        "ccp_w1": 0.4,                       # utility weight: stability
        "ccp_w2": 0.3,                       # utility weight: fund preservation
        "ccp_w3": 0.2,                       # utility weight: cascade prevention
        "ccp_w4": 0.1,                       # utility weight: market stress

        # market params (simulated exchange)
        "base_volatility": 0.20,
        "vol_shock_step": 15,          # volatility spike at step 15

        # exogenous shock
        "shock_step": 10,
        "shock_intensity": 0.5,

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
    print(f"  Plots saved to {out}/\n")


if __name__ == "__main__":
    main()
