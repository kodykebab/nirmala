"""
CCPAgent — Central Counter-Party that enforces margin requirements.

Responsibilities:
  • Collect bilateral exposures from all banks each timestep
  • Compute aggregate risk and set margin requirements
  • Issue margin calls to undercapitalized banks
"""

import agentpy as ap
import numpy as np


class CCPAgent(ap.Agent):
    """A single CCP that oversees the interbank network."""

    # ------------------------------------------------------------------ setup
    def setup(self):
        p = self.model.p
        self.margin_rate = p.get("initial_margin_rate", 0.10)  # 10 %
        self.margin_calls_this_step: list[int] = []

    # ============================================================ STEP LOGIC
    def step(self):
        """Called every timestep after all banks have acted."""
        exposures = self._collect_exposures()
        self._update_margin_rate(exposures)
        self._enforce_margin_calls()

    # ------------------------------------------------- collect exposures
    def _collect_exposures(self) -> dict[int, float]:
        """Return {bank_id: total_exposure} for every live bank."""
        exposures: dict[int, float] = {}
        for bank in self.model.banks:
            if bank.defaulted:
                continue
            exposures[bank.id] = sum(bank.exposure_to_neighbors.values())
        return exposures

    # ------------------------------------------------- update margin rate
    def _update_margin_rate(self, exposures: dict[int, float]):
        """Raise margin requirements when aggregate risk grows."""
        if not exposures:
            return
        total_exposure = sum(exposures.values())
        total_capital = sum(b.capital for b in self.model.banks if not b.defaulted)

        if total_capital <= 0:
            self.margin_rate = 0.50  # extreme stress cap
            return

        leverage = total_exposure / max(total_capital, 1)
        # simple rule: margin_rate scales with system leverage
        self.margin_rate = np.clip(0.05 + 0.05 * leverage, 0.05, 0.50)

    # ------------------------------------------------- margin calls
    def _enforce_margin_calls(self):
        """Force banks whose capital < required margin to post more collateral."""
        self.margin_calls_this_step.clear()
        for bank in self.model.banks:
            if bank.defaulted:
                continue
            total_exp = sum(bank.exposure_to_neighbors.values())
            required_margin = total_exp * self.margin_rate
            if bank.capital < required_margin:
                self._issue_margin_call(bank, shortfall=required_margin - bank.capital)

    def _issue_margin_call(self, bank, shortfall: float):
        """
        The bank must cover the shortfall from liquidity.
        If it cannot, it records a missed payment (stress signal).
        """
        self.margin_calls_this_step.append(bank.id)
        if bank.liquidity >= shortfall:
            bank.liquidity -= shortfall
            bank.capital += shortfall  # posted as collateral
        else:
            # partial post
            bank.capital += bank.liquidity
            bank.liquidity = 0
            bank.missed_payment = True
