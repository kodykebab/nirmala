"""
BankAgent — an AgentPy agent representing a single bank in a financial network.

State variables : liquidity, capital, exposure_to_neighbors
Bayesian beliefs: Beta-Bernoulli model for each neighbor's default probability
Actions         : lend, borrow, reduce_exposure, hoard_liquidity
"""

import agentpy as ap
import numpy as np


class BankAgent(ap.Agent):
    """A bank that participates in an interbank financial network."""

    # ------------------------------------------------------------------ setup
    def setup(self):
        """Initialise state variables and Bayesian priors."""
        p = self.model.p  # model parameters

        # --- core state ---
        self.liquidity = np.random.uniform(
            p.get("init_liquidity_lo", 50),
            p.get("init_liquidity_hi", 150),
        )
        self.capital = np.random.uniform(
            p.get("init_capital_lo", 80),
            p.get("init_capital_hi", 200),
        )
        self.exposure_to_neighbors: dict[int, float] = {}  # neighbor agent.id → $

        # --- status flags ---
        self.defaulted = False
        self.stressed = False          # True when liquidity < stress_threshold
        self.missed_payment = False    # True when a margin/obligation is missed

        # --- Bayesian beliefs (Beta-Bernoulli per neighbor) ---
        # alpha, beta parameterise Beta(alpha, beta)
        self._belief_alpha: dict[int, float] = {}
        self._belief_beta: dict[int, float] = {}

    def init_neighbor_data(self):
        """
        Set up initial beliefs and exposures for each graph neighbor.
        Must be called AFTER all agents have been placed on the network.
        """
        for nbr in self.model.network.neighbors(self):
            # prior: Beta(1, 9) → mean default prob ≈ 0.1
            self._belief_alpha[nbr.id] = 1.0
            self._belief_beta[nbr.id] = 9.0
            # random initial bilateral exposure
            self.exposure_to_neighbors[nbr.id] = np.random.uniform(5, 30)

    # ============================================================ STEP LOGIC
    def step(self):
        """Called every timestep by the model."""
        if self.defaulted:
            return  # dead banks do nothing

        # 1. Read own state  (already in self.*)
        # 2. Read neighbor states & observe stress signals
        neighbor_signals = self._observe_neighbors()

        # 3. Bayesian belief update
        self._update_beliefs(neighbor_signals)

        # 4. Compute risk metrics
        expected_loss, liquidity_shortfall = self._compute_risk_metrics()

        # 5. Choose action via expected utility
        action = self._choose_action(expected_loss, liquidity_shortfall)

        # 6. Execute
        self._execute_action(action)

        # 7. Update stress flag
        stress_thresh = self.model.p.get("stress_threshold", 30)
        self.stressed = self.liquidity < stress_thresh

        # 8. Check for default
        if self.liquidity <= 0 or self.capital <= 0:
            self._default()

    # ------------------------------------------------------ observe neighbors
    def _observe_neighbors(self) -> dict[int, dict]:
        """Return observable signals for each network neighbor."""
        signals: dict[int, dict] = {}
        for nbr in self.model.network.neighbors(self):
            signals[nbr.id] = {
                "stressed": nbr.stressed,
                "missed_payment": nbr.missed_payment,
                "defaulted": nbr.defaulted,
                "liquidity": nbr.liquidity,
                "agent": nbr,
            }
        return signals

    # ------------------------------------------------------ Bayesian updating
    def _update_beliefs(self, signals: dict[int, dict]):
        """Beta-Bernoulli update: treat stressed / missed_payment as 'default-like' event."""
        for nbr_id, sig in signals.items():
            if nbr_id not in self._belief_alpha:
                self._belief_alpha[nbr_id] = 1.0
                self._belief_beta[nbr_id] = 9.0

            # observation: 1 = distress signal, 0 = healthy
            obs = 1.0 if (sig["stressed"] or sig["missed_payment"] or sig["defaulted"]) else 0.0
            self._belief_alpha[nbr_id] += obs
            self._belief_beta[nbr_id] += (1.0 - obs)

    def belief_default_prob(self, nbr_id: int) -> float:
        """Posterior mean default probability for *nbr_id*."""
        a = self._belief_alpha.get(nbr_id, 1.0)
        b = self._belief_beta.get(nbr_id, 9.0)
        return a / (a + b)

    # ------------------------------------------------------ risk metrics
    def _compute_risk_metrics(self) -> tuple[float, float]:
        """Return (expected_loss, liquidity_shortfall)."""
        expected_loss = 0.0
        for nbr_id, exp in self.exposure_to_neighbors.items():
            pd = self.belief_default_prob(nbr_id)
            lgd = 0.6  # loss-given-default
            expected_loss += pd * lgd * exp

        min_liquidity = self.model.p.get("min_liquidity", 20)
        liquidity_shortfall = max(0.0, min_liquidity - self.liquidity)
        return expected_loss, liquidity_shortfall

    # ------------------------------------------------------ action selection
    def _choose_action(self, expected_loss: float, liquidity_shortfall: float) -> str:
        """
        Pick from {lend, borrow, reduce_exposure, hoard_liquidity}
        using a simple expected-utility heuristic.
        """
        utilities: dict[str, float] = {}

        # --- hoard_liquidity: attractive when shortfall is high
        utilities["hoard_liquidity"] = 2.0 * liquidity_shortfall

        # --- reduce_exposure: attractive when expected loss is high
        utilities["reduce_exposure"] = 1.5 * expected_loss

        # --- borrow: attractive when liquidity is low but capital is okay
        capital_ratio = self.capital / max(self.liquidity, 1)
        utilities["borrow"] = max(0, (30 - self.liquidity) * 0.5) if capital_ratio > 1.0 else 0.0

        # --- lend: attractive when very liquid and risk is low
        utilities["lend"] = max(0, (self.liquidity - 80) * 0.4 - expected_loss)

        return max(utilities, key=utilities.get)  # type: ignore[arg-type]

    # ------------------------------------------------------ action execution
    def _execute_action(self, action: str):
        """Mutate state according to chosen action."""
        p = self.model.p
        step_cost = p.get("step_operating_cost", 2)
        self.liquidity -= step_cost  # every bank pays running costs

        self.missed_payment = False  # reset per-step flag

        if action == "lend":
            self._action_lend()
        elif action == "borrow":
            self._action_borrow()
        elif action == "reduce_exposure":
            self._action_reduce_exposure()
        elif action == "hoard_liquidity":
            self._action_hoard_liquidity()

    def _action_lend(self):
        """Lend a portion of excess liquidity to a random live neighbor."""
        amount = min(self.liquidity * 0.1, 15)
        if amount <= 0:
            return
        neighbors = self._live_neighbors()
        if not neighbors:
            return
        borrower = self.model.random.choice(neighbors)
        self.liquidity -= amount
        borrower.liquidity += amount
        self.exposure_to_neighbors[borrower.id] = (
            self.exposure_to_neighbors.get(borrower.id, 0) + amount
        )

    def _action_borrow(self):
        """Attempt to borrow from a random live neighbor."""
        neighbors = self._live_neighbors()
        if not neighbors:
            self.missed_payment = True
            return
        lender = self.model.random.choice(neighbors)
        amount = min(lender.liquidity * 0.1, 10)
        if amount <= 1:
            self.missed_payment = True
            return
        lender.liquidity -= amount
        self.liquidity += amount
        lender.exposure_to_neighbors[self.id] = (
            lender.exposure_to_neighbors.get(self.id, 0) + amount
        )

    def _action_reduce_exposure(self):
        """Reduce exposure to the riskiest neighbor."""
        if not self.exposure_to_neighbors:
            return
        riskiest = max(self.exposure_to_neighbors, key=lambda n: self.belief_default_prob(n))
        reduction = self.exposure_to_neighbors[riskiest] * 0.2
        self.exposure_to_neighbors[riskiest] -= reduction
        self.liquidity += reduction * 0.5

    def _action_hoard_liquidity(self):
        """Conserve cash — reduce all exposures slightly."""
        for nbr_id in list(self.exposure_to_neighbors):
            cut = self.exposure_to_neighbors[nbr_id] * 0.05
            self.exposure_to_neighbors[nbr_id] -= cut
            self.liquidity += cut * 0.3

    # ------------------------------------------------------ default
    def _default(self):
        """Mark the bank as defaulted and propagate losses to neighbors."""
        self.defaulted = True
        self.stressed = True
        agent_map = {a.id: a for a in self.model.banks}
        for nbr_id, exp in self.exposure_to_neighbors.items():
            nbr = agent_map.get(nbr_id)
            if nbr is not None and not nbr.defaulted:
                loss = exp * 0.6  # LGD
                nbr.capital -= loss
                nbr.liquidity -= loss * 0.3
        self.exposure_to_neighbors.clear()
        self.liquidity = 0
        self.capital = 0

    # ------------------------------------------------------ helpers
    def _live_neighbors(self) -> list["BankAgent"]:
        """Return non-defaulted neighbors from the AgentPy network."""
        return [n for n in self.model.network.neighbors(self) if not n.defaulted]
