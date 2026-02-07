"""
CCPAgent — Central Counterparty agent with game-theoretic objective function.

The CCP is a strategic agent that:
  1. Maintains a centralised DEFAULT FUND (mutualized pool)
  2. Dynamically adjusts MARGIN RATES based on market volatility
  3. Issues MARGIN CALLS to banks whose exposure exceeds thresholds
  4. Activates PANIC MODE when total exposure breaches safe limits
  5. Mutualises DEFAULT LOSSES across all surviving member banks
  6. Maximises a UTILITY FUNCTION prioritising systemic stability

Utility function (game-theoretic objective):
    CCP_Utility = w1 * (1 - Panic_Mode)
                + w2 * (Default_Fund / Safe_Limit)
                - w3 * Num_Defaults
                - w4 * Fire_Sale_Volume

    Weights: w1=0.4  (stability)
             w2=0.3  (fund preservation)
             w3=0.2  (cascade prevention)
             w4=0.1  (market stress reduction)

Information asymmetry (game theory):
    CCP sees PRIVATE:  Total_Exposure, Member_Risk_Score, Default_Fund
    Banks see PUBLIC:   Current_Margin_Rate (published via Redis)
    Banks have PRIVATE: Bayesian beliefs (CCP cannot see)

Strategic tension:
    CCP wants high margins → safer but constrains bank activity
    Banks want low margins → more profit but higher systemic risk

Decision rules:
    - Current_Margin_Rate = base_margin + volatility * margin_sensitivity
    - Panic_Mode activates when Total_Exposure > Default_Fund * safe_multiplier
    - On bank default: Default_Fund -= uncovered_amount
    - Remaining loss mutualized across surviving members
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agents.BankAgent import ActionIntent

from agents.BankAgent import IntentFactory


# ═══════════════════════════════════════════════════════════════════════════
# CCP Agent
# ═══════════════════════════════════════════════════════════════════════════

class CCPAgent:
    """
    Central Counterparty agent — strategic game-theoretic actor.

    Unlike BankAgent (which is an ap.Agent on the network), the CCP is a
    singleton manager that sits between all banks, mediating risk.
    """

    def __init__(self, model, params: dict):
        self.model = model
        self.p = params

        # ── Default Fund (centralised pool) ─────────────────────────────
        self.default_fund: float = params.get("ccp_initial_default_fund", 100.0)

        # ── Margin rate state ───────────────────────────────────────────
        self.base_margin: float = params.get("ccp_base_margin", 0.05)
        self.margin_sensitivity: float = params.get("ccp_margin_sensitivity", 0.01)
        self.current_margin_rate: float = self.base_margin

        # ── Panic mode ──────────────────────────────────────────────────
        self.panic_mode: bool = False
        self.safe_multiplier: float = params.get("ccp_safe_multiplier", 10.0)

        # ── Utility weights ─────────────────────────────────────────────
        self.w1: float = params.get("ccp_w1", 0.4)   # stability
        self.w2: float = params.get("ccp_w2", 0.3)    # fund preservation
        self.w3: float = params.get("ccp_w3", 0.2)    # cascade prevention
        self.w4: float = params.get("ccp_w4", 0.1)    # market stress

        # ── Private CCP knowledge (information asymmetry) ──────────────
        self.total_exposure: float = 0.0
        self.member_risk_scores: dict[int, float] = {}
        self.fire_sale_volume: float = 0.0
        self.num_defaults_this_tick: int = 0

        # ── Margin call threshold ───────────────────────────────────────
        self.margin_call_threshold: float = params.get(
            "margin_call_threshold", 0.5
        )

        # ── Metric history ──────────────────────────────────────────────
        self.utility_history: list[float] = []
        self.margin_rate_history: list[float] = []
        self.panic_mode_history: list[bool] = []
        self.default_fund_history: list[float] = []
        self.fire_sale_history: list[float] = []

    # ═══════════════════════════════════════════════════════════ PROPERTIES

    @property
    def safe_limit(self) -> float:
        """Maximum acceptable total exposure = default_fund * multiplier."""
        return self.default_fund * self.safe_multiplier

    # ═══════════════════════════════════════════════════════════ STEP

    def step(self, tick: int) -> int:
        """
        Full CCP decision cycle each timestep. Returns number of margin
        calls issued.

        Order:
            1. Observe system state (private info gathering)
            2. Update margin rate dynamically (strategic decision)
            3. Check / update panic mode
            4. Compute member risk scores
            5. Issue margin calls
            6. Publish updated margin rate to Redis
            7. Compute and record CCP utility
        """
        # 1. Observe
        self._observe_system()

        # 2. Dynamic margin rate
        self._update_margin_rate()

        # 3. Panic mode check
        self._check_panic_mode()

        # 4. Member risk scores
        self._compute_member_risk_scores()

        # 5. Issue margin calls
        n_calls = self._issue_margin_calls(tick)

        # 6. Publish to Redis
        self._publish_margin_rate()

        # 7. Utility
        utility = self._compute_utility()
        self._record_metrics(utility)

        return n_calls

    # ═══════════════════════════════════════════════ 1. OBSERVE

    def _observe_system(self) -> None:
        """
        Gather PRIVATE information about the system.
        Banks cannot see this — information asymmetry.
        """
        banks = self.model.banks
        live = [b for b in banks if not b.defaulted]

        # Total network exposure (CCP private)
        self.total_exposure = sum(
            sum(b.exposure_to_neighbors.values()) for b in live
        )

        # Track fire-sale volume this tick (from last tick's actions)
        self.fire_sale_volume = 0.0
        for b in live:
            if (b.last_intent and
                    b.last_intent.action_type == "FIRE_SALE_ASSET"):
                qty = b.last_intent.payload.get("quantity", 0)
                self.fire_sale_volume += qty

        # Count defaults this tick
        self.num_defaults_this_tick = sum(1 for b in banks if b.defaulted)

    # ═══════════════════════════════════════════════ 2. DYNAMIC MARGIN

    def _update_margin_rate(self) -> None:
        """
        Strategically adjust margin rate based on volatility.

        Formula: Current_Margin_Rate = base_margin + volatility * sensitivity

        In panic mode, margin rate gets an additional premium.
        This is the CCP's primary strategic lever — higher margins make the
        system safer but constrain bank activity (strategic tension).
        """
        volatility = self.model._current_volatility

        self.current_margin_rate = (
            self.base_margin + volatility * self.margin_sensitivity
        )

        # Panic mode surcharge: +50% margin rate
        if self.panic_mode:
            self.current_margin_rate *= 1.5

        # Clamp to reasonable range
        self.current_margin_rate = max(0.02, min(0.30, self.current_margin_rate))

    # ═══════════════════════════════════════════════ 3. PANIC MODE

    def _check_panic_mode(self) -> None:
        """
        Panic_Mode = TRUE when Total_Exposure > Safe_Limit

        Safe_Limit = Default_Fund * safe_multiplier  (default 10x)

        When active:
            - Margin rates increase by 50% (applied in _update_margin_rate)
            - Margin call threshold tightens
            - CCP becomes more aggressive
        """
        was_panic = self.panic_mode
        self.panic_mode = self.total_exposure > self.safe_limit

        if self.panic_mode and not was_panic:
            # Entering panic — tighten thresholds
            self.margin_call_threshold = max(
                0.2, self.margin_call_threshold * 0.6
            )
        elif not self.panic_mode and was_panic:
            # Exiting panic — relax thresholds gradually
            self.margin_call_threshold = min(
                self.p.get("margin_call_threshold", 0.5),
                self.margin_call_threshold * 1.2,
            )

    # ═══════════════════════════════════════════════ 4. RISK SCORES

    def _compute_member_risk_scores(self) -> None:
        """
        Compute per-bank risk score (CCP private information).

        Risk_Score = w_exp * (exposure / capital) + w_liq * (1 - liq/threshold)
                   + w_stress * stressed

        Banks do NOT see their own risk score — only the public margin rate.
        """
        for bank in self.model.banks:
            if bank.defaulted:
                self.member_risk_scores[bank.bank_index] = 1.0
                continue

            total_exp = sum(bank.exposure_to_neighbors.values())
            cap = max(bank.capital, 1.0)
            liq = max(bank.liquidity, 0.0)
            stress_thresh = self.p.get("stress_threshold", 30)

            exp_ratio = total_exp / cap
            liq_score = max(0, 1.0 - liq / max(stress_thresh, 1))
            stress_flag = 1.0 if bank.stressed else 0.0

            score = (
                0.5 * min(exp_ratio, 3.0) / 3.0    # normalise to [0,1]
                + 0.3 * liq_score
                + 0.2 * stress_flag
            )

            self.member_risk_scores[bank.bank_index] = round(
                min(1.0, score), 4
            )

    # ═══════════════════════════════════════════════ 5. MARGIN CALLS

    def _issue_margin_calls(self, tick: int) -> int:
        """
        Issue margin calls to banks whose exposure/capital ratio exceeds
        the threshold. Uses the CCP's dynamic margin rate.

        In panic mode, threshold is tighter → more calls issued.
        """
        n_issued = 0

        for bank in self.model.banks:
            if bank.defaulted:
                continue

            total_exp = sum(bank.exposure_to_neighbors.values())
            ratio = total_exp / max(bank.capital, 1)

            if ratio > self.margin_call_threshold:
                margin_amount = total_exp * self.current_margin_rate

                # Risk-weighted adjustment: riskier banks get larger calls
                risk_score = self.member_risk_scores.get(
                    bank.bank_index, 0.5
                )
                margin_amount *= (1.0 + risk_score * 0.5)

                call = IntentFactory.issue_margin_call(
                    tick=tick,
                    agent_id="ccp_01",
                    target_agent_id=f"bank_{bank.bank_index:02d}",
                    margin_amount=round(margin_amount, 2),
                    deadline_tick=tick + (1 if self.panic_mode else 2),
                    reason=(
                        "panic_mode_breach" if self.panic_mode
                        else "exposure_ratio_breach"
                    ),
                )
                self.model.redis.publish_margin_call(
                    bank.bank_index, call.to_dict()
                )
                n_issued += 1

        return n_issued

    # ═══════════════════════════════════════════════ 6. PUBLISH

    def _publish_margin_rate(self) -> None:
        """
        Publish the current margin rate to Redis — this is the ONLY
        piece of CCP information that banks can see (public signal).
        """
        self.model.redis.publish_system_state({
            "margin_rate": round(self.current_margin_rate, 4),
        })

    # ═══════════════════════════════════════════════ 7. UTILITY

    def _compute_utility(self) -> float:
        """
        CCP objective function (game-theoretic goal):

        CCP_Utility = w1 * (1 - Panic_Mode)
                    + w2 * (Default_Fund / Safe_Limit)
                    - w3 * Num_Defaults
                    - w4 * Fire_Sale_Volume_Normalised

        Interpretation:
            - Maximise stability  (avoid panic mode)
            - Preserve default fund  (don't burn through mutualized capital)
            - Minimise defaults  (prevent cascades)
            - Reduce market stress  (limit contagion via fire sales)
        """
        panic_penalty = 1.0 if self.panic_mode else 0.0

        fund_ratio = (
            self.default_fund / max(self.safe_limit, 1.0)
        )
        fund_ratio = min(fund_ratio, 1.0)  # cap at 1.0

        n_defaults = self.num_defaults_this_tick
        n_banks = max(len(self.model.banks), 1)
        norm_defaults = n_defaults / n_banks  # normalise to [0, 1]

        # Normalise fire-sale volume relative to total system liquidity
        total_liq = sum(
            b.liquidity for b in self.model.banks if not b.defaulted
        )
        norm_fire_sale = (
            self.fire_sale_volume / max(total_liq, 1.0)
        )
        norm_fire_sale = min(norm_fire_sale, 1.0)

        utility = (
            self.w1 * (1.0 - panic_penalty)
            + self.w2 * fund_ratio
            - self.w3 * norm_defaults
            - self.w4 * norm_fire_sale
        )

        return round(utility, 4)

    # ═══════════════════════════════════════════════ DEFAULT HANDLING

    def handle_bank_default(self, defaulting_bank) -> None:
        """
        Called when a bank defaults.  Implements the CCP default waterfall:

        1. Calculate uncovered losses from the defaulting bank's exposures
        2. Default_Fund -= uncovered_amount
        3. If fund is insufficient, mutualise remaining loss across
           all surviving member banks proportionally

        This is a KEY game-theoretic mechanism: banks know their
        individual losses are partially mutualised, which affects their
        risk-taking incentives.
        """
        # Calculate total exposure that other banks had to the defaulting bank
        total_uncovered = 0.0
        for bank in self.model.banks:
            if bank.defaulted or bank.id == defaulting_bank.id:
                continue
            exposure = bank.exposure_to_neighbors.get(defaulting_bank.id, 0)
            if exposure > 0:
                loss = exposure * 0.6  # LGD = 60%
                total_uncovered += loss

        # Step 1: Absorb from default fund
        fund_absorption = min(self.default_fund, total_uncovered)
        self.default_fund -= fund_absorption
        remaining_loss = total_uncovered - fund_absorption

        # Step 2: Direct contagion — neighbors take bilateral losses
        # (This is already handled by BankAgent._default(), but the CCP
        #  reduces the blow by absorbing some via the default fund)

        # Step 3: Mutualise remaining shortfall across ALL surviving banks
        if remaining_loss > 0:
            survivors = [
                b for b in self.model.banks
                if not b.defaulted and b.id != defaulting_bank.id
            ]
            if survivors:
                per_bank_share = remaining_loss / len(survivors)
                for bank in survivors:
                    bank.capital -= per_bank_share * 0.5
                    bank.liquidity -= per_bank_share * 0.5

    def accept_default_fund_deposit(self, amount: float) -> None:
        """
        Accept a deposit from a bank into the centralised default fund.
        Called when a bank executes DEPOSIT_DEFAULT_FUND.
        """
        self.default_fund += amount

    # ═══════════════════════════════════════════════ METRICS

    def _record_metrics(self, utility: float) -> None:
        """Record CCP-level metrics for visualization."""
        self.utility_history.append(utility)
        self.margin_rate_history.append(self.current_margin_rate)
        self.panic_mode_history.append(self.panic_mode)
        self.default_fund_history.append(self.default_fund)
        self.fire_sale_history.append(self.fire_sale_volume)

    @property
    def metrics(self) -> dict:
        """Return all CCP metric time series."""
        return {
            "ccp_utility": list(self.utility_history),
            "ccp_margin_rate": list(self.margin_rate_history),
            "ccp_panic_mode": [int(p) for p in self.panic_mode_history],
            "ccp_default_fund": list(self.default_fund_history),
            "ccp_fire_sale_volume": list(self.fire_sale_history),
        }
