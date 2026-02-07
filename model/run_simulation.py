#!/usr/bin/env python3
"""
run_simulation.py — Entry point for the interbank financial-network simulation.

Creates an 8-bank Erdős–Rényi network, runs for 50 timesteps, prints a
summary table, and produces four diagnostic plots.
"""

from model import FinancialNetworkModel
from visualization import (
    plot_bank_status,
    plot_aggregate_liquidity,
    plot_network_snapshot,
    plot_exposure_and_margin,
)
import os


def main():
    # ── parameters ──────────────────────────────────────────────────────
    parameters = {
        "n_banks": 10,
        "network_type": "erdos_renyi",   # 'erdos_renyi' | 'scale_free' | 'small_world'
        "er_prob": 0.35,
        "steps": 40,
        "init_liquidity_lo": 50,
        "init_liquidity_hi": 140,
        "init_capital_lo": 60,
        "init_capital_hi": 180,
        "stress_threshold": 30,
        "min_liquidity": 25,
        "step_operating_cost": 2.0,
        "initial_margin_rate": 0.10,
        "shock_step": 12,              # timestep at which an external shock hits
        "shock_intensity": 0.5,        # fraction of liquidity lost by shocked banks
        "seed": 99,
    }

    # ── run ──────────────────────────────────────────────────────────────
    model = FinancialNetworkModel(parameters)
    results = model.run()

    # ── summary ──────────────────────────────────────────────────────────
    m = model.metrics
    print("\n" + "=" * 60)
    print("  SIMULATION SUMMARY")
    print("=" * 60)
    print(f"  Banks             : {parameters['n_banks']}")
    print(f"  Timesteps         : {parameters['steps']}")
    print(f"  Final defaults    : {m['defaults'][-1]}")
    print(f"  Final active      : {m['active'][-1]}")
    print(f"  Freeze events     : {sum(m['freeze_events'])}")
    print(f"  Final liquidity   : {m['liquidity'][-1]:.1f}")
    print(f"  Final exposure    : {m['exposure'][-1]:.1f}")
    print(f"  Final margin rate : {m['margin_rate'][-1]:.2%}")
    print("=" * 60 + "\n")

    # ── per-bank snapshot ────────────────────────────────────────────────
    print(f"{'Bank':>6} {'Liquidity':>10} {'Capital':>10} {'Exposure':>10} {'Status':>10}")
    print("-" * 50)
    for b in model.banks:
        exp = sum(b.exposure_to_neighbors.values())
        status = "DEFAULT" if b.defaulted else ("STRESSED" if b.stressed else "OK")
        print(f"  B{b.id:<4} {b.liquidity:>10.1f} {b.capital:>10.1f} {exp:>10.1f} {status:>10}")
    print()

    # ── plots ────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out, exist_ok=True)
    plot_bank_status(m, save_path=os.path.join(out, "bank_status.png"))
    plot_aggregate_liquidity(m, save_path=os.path.join(out, "liquidity.png"))
    plot_exposure_and_margin(m, save_path=os.path.join(out, "exposure_margin.png"))
    plot_network_snapshot(model, save_path=os.path.join(out, "network_snapshot.png"))
    print(f"Plots saved to {out}/")


if __name__ == "__main__":
    main()
