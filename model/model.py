"""
FinancialNetworkModel — AgentPy model that wires BankAgents + CCPAgent
on a NetworkX interbank graph and tracks systemic metrics each timestep.
"""

import agentpy as ap
import networkx as nx
import numpy as np

from agents.BankAgent import BankAgent
from agents.CCPAgent import CCPAgent


class FinancialNetworkModel(ap.Model):
    """
    Main simulation model.

    Parameters (passed via ``p``):
        n_banks              : int   – number of bank agents
        network_type         : str   – 'erdos_renyi' | 'scale_free' | 'small_world'
        er_prob              : float – edge probability for Erdős–Rényi
        steps                : int   – simulation length
        init_liquidity_lo/hi : float – uniform range for initial liquidity
        init_capital_lo/hi   : float – uniform range for initial capital
        stress_threshold     : float – liquidity below this → stressed
        min_liquidity        : float – liquidity target used by risk calc
        step_operating_cost  : float – per-step running cost for each bank
        initial_margin_rate  : float – CCP starting margin rate
        seed                 : int   – random seed
    """

    def setup(self):
        """Create network, bank agents, and the CCP."""
        n = self.p.get("n_banks", 8)
        net_type = self.p.get("network_type", "erdos_renyi")

        # --- build NetworkX graph ---
        if net_type == "scale_free":
            G = nx.barabasi_albert_graph(n, 2, seed=self.p.get("seed", 42))
        elif net_type == "small_world":
            G = nx.watts_strogatz_graph(n, 4, 0.3, seed=self.p.get("seed", 42))
        else:  # erdos_renyi
            G = nx.erdos_renyi_graph(n, self.p.get("er_prob", 0.4), seed=self.p.get("seed", 42))

        # --- create AgentPy network ---
        self.network = ap.Network(self, G)

        # --- bank agents (one per node) ---
        self.banks = ap.AgentList(self, n, BankAgent)
        nodes = list(self.network.nodes)
        self.network.add_agents(self.banks, nodes)

        # now that agents sit on the graph, initialise neighbor data
        for bank in self.banks:
            bank.init_neighbor_data()

        # keep a copy of the original NX graph (with integer labels) for plotting
        self._nx_graph = G

        # --- CCP agent ---
        self.ccp = CCPAgent(self)

        # --- metric accumulators ---
        self._ts_defaults: list[int] = []
        self._ts_active: list[int] = []
        self._ts_liquidity: list[float] = []
        self._ts_exposure: list[float] = []
        self._ts_margin_rate: list[float] = []
        self._ts_freeze_events: list[int] = []

    # ---------------------------------------------------------------- step
    def step(self):
        """One simulation timestep."""
        # apply external shock if scheduled
        self._apply_shock()
        # banks act
        self.banks.step()
        # CCP acts (after banks)
        self.ccp.step()
        # record metrics
        self._record_metrics()

    # ---------------------------------------------------------------- shock
    def _apply_shock(self):
        """Apply a one-time exogenous liquidity shock at a configured timestep."""
        shock_step = self.p.get("shock_step", None)
        if shock_step is None or self.t != shock_step:
            return
        intensity = self.p.get("shock_intensity", 0.3)
        # shock hits ~60 % of banks at random
        for bank in self.banks:
            if bank.defaulted:
                continue
            if self.random.random() < 0.6:
                drain = bank.liquidity * intensity
                bank.liquidity -= drain
                bank.capital -= drain * 0.8
                bank.stressed = True

    # ---------------------------------------------------------------- metrics
    def _record_metrics(self):
        n_defaulted = sum(1 for b in self.banks if b.defaulted)
        n_active = len(self.banks) - n_defaulted
        total_liq = sum(b.liquidity for b in self.banks if not b.defaulted)
        total_exp = sum(
            sum(b.exposure_to_neighbors.values())
            for b in self.banks
            if not b.defaulted
        )
        # liquidity freeze: >50 % of active banks are stressed
        n_stressed = sum(1 for b in self.banks if b.stressed and not b.defaulted)
        freeze = 1 if (n_active > 0 and n_stressed / n_active > 0.5) else 0

        self._ts_defaults.append(n_defaulted)
        self._ts_active.append(n_active)
        self._ts_liquidity.append(total_liq)
        self._ts_exposure.append(total_exp)
        self._ts_margin_rate.append(self.ccp.margin_rate)
        self._ts_freeze_events.append(freeze)

        # also record into AgentPy's built-in reporter
        self.record("n_defaulted", n_defaulted)
        self.record("n_active", n_active)
        self.record("total_liquidity", total_liq)
        self.record("total_exposure", total_exp)
        self.record("margin_rate", self.ccp.margin_rate)
        self.record("liquidity_freeze", freeze)

    # ---------------------------------------------------------------- end
    def end(self):
        """Summary statistics reported at end of run."""
        self.report("total_defaults", self._ts_defaults[-1] if self._ts_defaults else 0)
        self.report("total_freeze_events", sum(self._ts_freeze_events))
        self.report("final_total_exposure", self._ts_exposure[-1] if self._ts_exposure else 0)

    # ---------------------------------------------------------------- accessors
    @property
    def metrics(self) -> dict:
        """Return a dict of time-series for external plotting."""
        return {
            "defaults": list(self._ts_defaults),
            "active": list(self._ts_active),
            "liquidity": list(self._ts_liquidity),
            "exposure": list(self._ts_exposure),
            "margin_rate": list(self._ts_margin_rate),
            "freeze_events": list(self._ts_freeze_events),
        }
