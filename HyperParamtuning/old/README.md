# Hierarchical MAPC DQN overnight experiment

## What this changes

The experiment tunes everything requested except the DQN discount factor.
`discount` remains hard-coded to `0.0` in `mapc_dqn_agent_factory_tuned.py`.

The main tunables are:
- learning rate
- replay buffer size
- replay batch size
- replay updates per environment step
- epsilon minimum
- epsilon decay, separately for each hierarchy level
- Q-network hidden-layer widths
- optional LayerNorm flag (kept off in the sweep)

## Recommended file placement

Copy:

- `q_network_tuned.py` -> into `mapc_cmab/agents/q_network.py`
- `mapc_dqn_agent_factory_tuned.py` -> into the project directory, or import/use it from your runner
- `tune_hierarchical_dqn.py` -> project root
- `run_overnight.sh` -> project root

The runner imports `mapc_dqn_agent_factory_tuned` directly, so you do **not** need to replace the existing factory just to run this experiment.

## Sweep design

The sweep now contains **64 configurations**: 8 optimizer/replay/network regimes crossed with 8 exploration profiles. This deliberately explores a broad region instead of betting on a handful of hand-picked settings.

Optimizer/replay/network dimensions include:
- learning rate: `3e-4`, `5e-4`, `7e-4`, `1e-3`
- replay buffer: `1k`, `2k`, `5k`, `10k`
- batch size: `32`, `64`
- replay updates per environment step: `1`, `2`
- network sizes: `(32,32)`, `(64,64)`, `(128,64)`, `(128,128)`

Exploration profiles vary decay independently across hierarchy levels and test epsilon floors of `0.01`, `0.02`, `0.05`, and `0.10`. The current `0.995` schedule is retained as a baseline.

The goal is not to claim one configuration is universally best before seeing data. The script measures reward level, tail variability, cross-run variability, and approximate time-to-90%-of-tail, then ranks configurations using a fixed composite score.

## Overnight workload

Default:
- Sweep: **64 configurations × 3 runs × 2,500 steps = 480,000 environment steps**.
- Final confirmation: **20 runs × 10,000 steps = 200,000 environment steps**.

Total: **680,000 environment steps**, plus the DQN update work inside each step.

The final 20×10k experiment is run only for the automatically selected winner, so the expensive confirmation stage is not repeated 64 times.

## Launch

From the project root:

```bash
export PYTHON_BIN=python
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USER='youraddress@gmail.com'
export SMTP_PASSWORD='your-app-password'
export EMAIL_TO='youraddress@gmail.com'
./run_overnight.sh
```

For a quick smoke test before leaving the lab:

```bash
SWEEP_RUNS=1 SWEEP_STEPS=200 FINAL_RUNS=1 FINAL_STEPS=300 ./run_overnight.sh
```

## Morning outputs

Look first at:

- `results/overnight/sweep_learning_curves_top12.png` (most useful human-readable graph)
- `results/overnight/sweep_learning_curves_all.png` (all 64 configurations)
- `results/overnight/sweep_ranking.txt`
- `results/overnight/sweep_metrics.csv`
- `results/overnight/final_config.json`
- `results/overnight/final_metrics.json`
- the throughput CI plot produced by `analyze_and_plot_throughputs`
- `results/overnight/experiment.log`

The raw final array is `final_throughputs.npy`.
