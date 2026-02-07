"""
visualization.py — Plotting utilities for the bank-agent simulation.

Plots:
  1. Active vs Defaulted banks over time
  2. Aggregate liquidity (with freeze-event shading)
  3. Bayesian belief evolution (avg PD, stress, margin call, volatility)
  4. Action distribution over time (stacked area — 6 action types)
  5. Network graph snapshot coloured by status
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1. Bank status
# ═══════════════════════════════════════════════════════════════════════════

def plot_bank_status(metrics: dict, save_path: str | None = None):
    steps = range(1, len(metrics["active"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, metrics["active"], label="Active", color="seagreen", lw=2)
    ax.plot(steps, metrics["defaults"], label="Defaulted", color="crimson", lw=2)
    ax.fill_between(steps, metrics["defaults"], alpha=0.15, color="crimson")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Banks")
    ax.set_title("Active vs Defaulted Banks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Aggregate liquidity
# ═══════════════════════════════════════════════════════════════════════════

def plot_aggregate_liquidity(metrics: dict, save_path: str | None = None):
    steps = range(1, len(metrics["liquidity"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, metrics["liquidity"], color="royalblue", lw=2,
            label="Total liquidity")
    for t, freeze in enumerate(metrics["freeze_events"]):
        if freeze:
            ax.axvspan(t + 0.5, t + 1.5, color="orange", alpha=0.25)
    freeze_patch = mpatches.Patch(color="orange", alpha=0.35,
                                  label="Freeze event")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(freeze_patch)
    ax.legend(handles=handles)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Aggregate Liquidity")
    ax.set_title("System Liquidity Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bayesian belief evolution (4 panels)
# ═══════════════════════════════════════════════════════════════════════════

def plot_belief_evolution(metrics: dict, save_path: str | None = None):
    steps = range(1, len(metrics["avg_pd"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)

    # A — avg counterparty PD
    ax = axes[0]
    ax.plot(steps, metrics["avg_pd"], color="crimson", lw=2)
    ax.set_title("Avg Counterparty PD")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("P(default)")
    ax.grid(True, alpha=0.3)

    # B — network stress belief
    ax = axes[1]
    ax.plot(steps, metrics["avg_stress_belief"], color="darkorange", lw=2)
    ax.set_title("Network Stress Belief")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Stress level")
    ax.grid(True, alpha=0.3)

    # C — expected margin call
    ax = axes[2]
    ax.plot(steps, metrics["avg_margin_belief"], color="steelblue", lw=2)
    ax.set_title("Expected Margin Call")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Amount")
    ax.grid(True, alpha=0.3)

    # D — market volatility belief
    ax = axes[3]
    ax.plot(steps, metrics["avg_volatility_belief"], color="purple", lw=2)
    ax.set_title("Market Volatility Belief")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Volatility")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Bayesian Belief Evolution (bank averages)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Action distribution (6 action types)
# ═══════════════════════════════════════════════════════════════════════════

def plot_action_distribution(metrics: dict, save_path: str | None = None):
    actions_ts = metrics["actions"]
    steps = range(1, len(actions_ts) + 1)

    # batch 1
    otc    = [a.get("route_otc_proposal", 0) for a in actions_ts]
    borrow = [a.get("borrow", 0) for a in actions_ts]
    reduce = [a.get("reduce_exposure", 0) for a in actions_ts]
    hoard  = [a.get("hoard_liquidity", 0) for a in actions_ts]
    pay_mc = [a.get("pay_margin_call", 0) for a in actions_ts]
    sell   = [a.get("sell_asset_standard", 0) for a in actions_ts]
    # batch 2
    ib_credit = [a.get("PROVIDE_INTERBANK_CREDIT", 0) for a in actions_ts]
    ib_repay  = [a.get("REPAY_INTERBANK_LOAN", 0) for a in actions_ts]
    fire_sale = [a.get("FIRE_SALE_ASSET", 0) for a in actions_ts]
    decl_def  = [a.get("DECLARE_DEFAULT", 0) for a in actions_ts]
    dep_fund  = [a.get("DEPOSIT_DEFAULT_FUND", 0) for a in actions_ts]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        steps,
        otc, borrow, reduce, hoard, pay_mc, sell,
        ib_credit, ib_repay, fire_sale, decl_def, dep_fund,
        labels=[
            "OTC Proposal", "Borrow", "Reduce Exp",
            "Hoard", "Pay Margin", "Sell Asset",
            "IB Credit", "Repay Loan", "Fire Sale",
            "Declare Def", "Deposit Fund",
        ],
        colors=[
            "#4caf50", "#2196f3", "#ff9800",
            "#9e9e9e", "#e91e63", "#9c27b0",
            "#00bcd4", "#8bc34a", "#f44336",
            "#000000", "#ffc107",
        ],
        alpha=0.85,
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of banks")
    ax.set_title("Action Distribution Over Time")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Network snapshot
# ═══════════════════════════════════════════════════════════════════════════

def plot_network_snapshot(model, save_path: str | None = None):
    G = model._nx_graph
    banks_list = list(model.banks)

    color_map, sizes, labels = [], [], {}
    for i, node in enumerate(G.nodes):
        b = banks_list[i] if i < len(banks_list) else None
        if b is None or b.defaulted:
            color_map.append("crimson")
            sizes.append(200)
            labels[node] = f"B{node}\n✗"
        elif b.stressed:
            color_map.append("orange")
            sizes.append(300 + b.liquidity * 3)
            labels[node] = f"B{node}\n{b.liquidity:.0f}"
        else:
            color_map.append("seagreen")
            sizes.append(300 + b.liquidity * 3)
            labels[node] = f"B{node}\n{b.liquidity:.0f}"

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map,
                           node_size=sizes, edgecolors="k", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                            font_weight="bold")
    legend_elements = [
        mpatches.Patch(facecolor="seagreen", edgecolor="k", label="Healthy"),
        mpatches.Patch(facecolor="orange", edgecolor="k", label="Stressed"),
        mpatches.Patch(facecolor="crimson", edgecolor="k", label="Defaulted"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title("Interbank Network — Final Snapshot")
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
