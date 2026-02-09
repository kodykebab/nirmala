# Nirmala

Agent-based simulation of a financial clearing network with autonomous Bayesian bank agents, a game-theoretic Central Counterparty (CCP), and a real-time dashboard.

> *Named "Nirmala" (pure, spotless, clean, bright) because nothing says "clean code" like a financial simulation held together by Redis and prayers.*

### Simulation (`model/`)

- **BankAgent** — Bayesian-RL agent that maintains 4 private belief channels (counterparty risk, liquidity stress, margin call size, market volatility), picks actions via expected-utility maximisation, and emits structured intents across 11 action types.
- **CCPAgent** — Game-theoretic counterparty that dynamically adjusts margin rates, issues margin calls, manages a default fund with loss mutualisation, and optimises a multi-objective utility function.
- **State sync** — Redis for pub/sub snapshots, Neo4j for intent logging and graph persistence.

### Central Server (`central/`)

- FastAPI app receiving agent intents at `POST /intent`
- Builds and serves a live NetworkX graph (`GET /network/graph`, `GET /network/stats`)
- Publishes intents to Redis and logs them to Neo4j


## Getting Started

```bash
pip install -r requirements.txt
cd model
# make sure redis-server is running 🙏
python3 live_dashboard.py
python3 run_simulation.py

```


## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_banks` | 10 | Number of bank agents |
| `steps` | 25000 | Simulation timesteps |
| `network_type` | `erdos_renyi` | Graph topology (`scale_free`, `small_world`) |
| `margin_rate` | 0.4 | Initial CCP margin rate |
| `stress_threshold` | 15 | Liquidity level triggering stress |

## License

MIT