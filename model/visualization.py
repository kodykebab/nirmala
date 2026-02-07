"""
visualization.py — Plotting utilities for the financial-network simulation.

Produces three figures:
  1. Active vs Defaulted banks over time
  2. Aggregate system liquidity over time
  3. Network graph snapshot with stress / default coloring
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def plot_bank_status(metrics: dict, save_path: str | None = None):
    """Line chart: active (green) and defaulted (red) banks per timestep."""
    steps = range(1, len(metrics["active"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, metrics["active"], label="Active banks", color="seagreen", linewidth=2)
    ax.plot(steps, metrics["defaults"], label="Defaulted banks", color="crimson", linewidth=2)
    ax.fill_between(steps, metrics["defaults"], alpha=0.15, color="crimson")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of banks")
    ax.set_title("Active vs Defaulted Banks Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_aggregate_liquidity(metrics: dict, save_path: str | None = None):
    """Line chart of total system liquidity with freeze-event shading."""
    steps = range(1, len(metrics["liquidity"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, metrics["liquidity"], label="Total liquidity", color="royalblue", linewidth=2)

    # shade freeze events
    for t, freeze in enumerate(metrics["freeze_events"]):
        if freeze:
            ax.axvspan(t + 0.5, t + 1.5, color="orange", alpha=0.25)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Aggregate Liquidity")
    ax.set_title("System Liquidity Over Time")
    freeze_patch = mpatches.Patch(color="orange", alpha=0.35, label="Liquidity freeze event")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(freeze_patch)
    ax.legend(handles=handles)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_network_snapshot(model, save_path: str | None = None):
    """
    Draw the interbank network.
    Colors:  green = healthy,  orange = stressed,  red = defaulted
    Node size proportional to remaining liquidity.
    """
    G = model._nx_graph  # original NetworkX graph with integer node labels
    banks_list = list(model.banks)  # ordered same as graph nodes

    color_map = []
    sizes = []
    labels = {}
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
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map, node_size=sizes, edgecolors="k", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    # legend
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


def plot_exposure_and_margin(metrics: dict, save_path: str | None = None):
    """Dual-axis chart: total exposure (left) and CCP margin rate (right)."""
    steps = range(1, len(metrics["exposure"]) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 4))

    color_exp = "steelblue"
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Total Network Exposure", color=color_exp)
    ax1.plot(steps, metrics["exposure"], color=color_exp, linewidth=2, label="Exposure")
    ax1.tick_params(axis="y", labelcolor=color_exp)

    ax2 = ax1.twinx()
    color_mr = "darkorange"
    ax2.set_ylabel("CCP Margin Rate", color=color_mr)
    ax2.plot(steps, metrics["margin_rate"], color=color_mr, linewidth=2, linestyle="--", label="Margin rate")
    ax2.tick_params(axis="y", labelcolor=color_mr)

    ax1.set_title("Network Exposure & CCP Margin Rate")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
