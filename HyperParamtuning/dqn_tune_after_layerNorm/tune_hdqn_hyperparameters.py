#!/usr/bin/env python3
"""
tune_hdqn_hyperparameters.py

Aggressive, resumable hyperparameter tuning for the hierarchical MAPC DQN.

What is tuned INDEPENDENTLY for each hierarchy level (L1-L4):
    - optimizer family
    - learning rate
    - replay-buffer size
    - replay batch size
    - replay steps
    - epsilon minimum
    - epsilon decay
    - AdamW weight decay (when AdamW is selected)

The search is deterministic random search by default. Every generated
configuration has a stable ID and every completed RUN is checkpointed, so
the experiment can resume after a crash without losing completed runs.

Outputs:
    <output-dir>/
        configs.csv
        summary.csv
        search_metadata.txt
        cache/
            <config_id>/
                run_000.npz
                run_001.npz
                ...
                complete.npz
        plots/
            top_configurations.png
            throughput_vs_stability.png
            best_learning_curve.png

Recommended workflow:
    1. Start with --n-configs 1000 --n-runs 2 for screening.
    2. Take the best ~20 configurations.
    3. Re-run finalists with 20-50 runs for statistical validation.

The default objective is deliberately aligned with the user's goal:
    high post-switch throughput + high pre-switch throughput
    + fast arrival at the ~1500 Mbps target region
    + stable plateau / low run-to-run variation.

No hidden DQN parameters are invented here; only parameters exposed by the
provided MapcDQNAgentFactory parameter dictionaries are tuned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import jax
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy import stats
from tqdm import tqdm

from mapc_cmab.agents.mapc_cmab_agent_factory import MapcDQNAgentFactory
from mapc_cmab.envs.scenario_impl import small_office_scenario


# ============================================================
# Scenario
# ============================================================

class MixScen:
    def __init__(
        self,
        scenario_factory,
        d_sta_1: int,
        d_sta_2: int,
        d_ap: float,
        max_steps: int,
    ):
        self.scen1 = scenario_factory(d_ap=d_ap, d_sta=d_sta_1)
        self.scen2 = scenario_factory(d_ap=d_ap, d_sta=d_sta_2)

        self.step = 0
        self.switch_steps = max_steps // 2

        self.data_rate_fn1 = jax.jit(self.scen1.data_rate_fn)
        self.data_rate_fn2 = jax.jit(self.scen2.data_rate_fn)

        self.data_rate_fn = self.data_rate_fn1
        self.associations = self.scen1.associations

        self.str_repr = (
            f"mix_scen_ap_{d_ap:g}_"
            f"dsta_{d_sta_1}_{d_sta_2}_s{max_steps}"
        )

    def __call__(self, key, link_ap_sta):
        self.step += 1

        if self.step == self.switch_steps:
            self.data_rate_fn = (
                self.data_rate_fn2
                if self.data_rate_fn is self.data_rate_fn1
                else self.data_rate_fn1
            )

        return self.data_rate_fn(key, link_ap_sta=link_ap_sta)

    def reset(self):
        self.data_rate_fn = self.data_rate_fn1
        self.step = 0


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Aggressive independent-per-level, resumable HDQN "
            "hyperparameter tuning."
        )
    )

    # Search
    p.add_argument("--n-configs", type=int, default=1000)
    p.add_argument("--search-seed", type=int, default=20261006)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing run/config caches and recompute.",
    )

    # Independent search ranges
    p.add_argument(
        "--optimizers",
        nargs="+",
        choices=["adam", "adamw"],
        default=["adam", "adamw"],
        help="Optimizer families sampled independently per level.",
    )

    p.add_argument(
        "--learning-rate-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[1e-4, 5e-3],
        help="Log-uniform learning-rate range.",
    )

    p.add_argument(
        "--buffer-sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000, 10000],
    )

    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128],
    )

    p.add_argument(
        "--replay-steps-values",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )

    p.add_argument(
        "--epsilon-min-values",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    )

    p.add_argument(
        "--epsilon-decay-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[0.98, 0.9999],
        help=(
            "Decay range. Sampling is log-uniform in (1-decay), "
            "giving more resolution close to 1."
        ),
    )

    p.add_argument(
        "--adamw-weight-decay-values",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3],
    )

    # Experiment
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--n-steps", type=int, default=10_000)

    # Optional cheap screening override
    p.add_argument("--screen-runs", type=int, default=None)
    p.add_argument("--screen-steps", type=int, default=None)

    # Scenario
    p.add_argument("--d-ap", type=float, default=10.0)
    p.add_argument("--d-sta-1", type=int, default=2)
    p.add_argument("--d-sta-2", type=int, default=4)
    p.add_argument("--n-links", type=int, default=3)
    p.add_argument("--n-tx-power-levels", type=int, default=4)

    # Analysis
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--confidence", type=float, default=0.99)

    # Objective target
    p.add_argument(
        "--target-throughput",
        type=float,
        default=1500.0,
        help=(
            "Mbps performance target used for learning-speed measurement. "
            "This is a near-optimal target, not a hard physical maximum."
        ),
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="./hdqn_hyperparameter_tuning",
    )
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--show", action="store_true")

    return p.parse_args()


# ============================================================
# Random distributions
# ============================================================

def log_uniform(rng, low: float, high: float) -> float:
    if not (low > 0 and high > low):
        raise ValueError(f"Invalid log-uniform range: {low}, {high}")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def decay_log_uniform(rng, low: float, high: float) -> float:
    """
    Sample decay by sampling q = 1-decay log-uniformly.
    This gives much better coverage near decay=1.
    """
    if not (0 < low < high < 1):
        raise ValueError(
            "epsilon decay range must satisfy 0 < LOW < HIGH < 1"
        )

    q_low = 1.0 - high
    q_high = 1.0 - low
    q = np.exp(rng.uniform(np.log(q_low), np.log(q_high)))
    return float(1.0 - q)


# ============================================================
# Configuration generation
# ============================================================

LEVELS = ("l1", "l2", "l3", "l4")


def make_optimizer(name: str, lr: float, weight_decay: float):
    if name == "adam":
        return optax.adam(lr)

    if name == "adamw":
        return optax.adamw(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {name}")


def sample_configuration(rng, args):
    """
    Every tunable value is sampled independently for L1/L2/L3/L4.
    """
    optimizers = tuple(
        str(rng.choice(args.optimizers))
        for _ in LEVELS
    )

    learning_rates = tuple(
        log_uniform(
            rng,
            args.learning_rate_range[0],
            args.learning_rate_range[1],
        )
        for _ in LEVELS
    )

    buffers = tuple(
        int(rng.choice(args.buffer_sizes))
        for _ in LEVELS
    )

    batches = tuple(
        int(rng.choice(args.batch_sizes))
        for _ in LEVELS
    )

    replay_steps = tuple(
        int(rng.choice(args.replay_steps_values))
        for _ in LEVELS
    )

    epsilon_min = tuple(
        float(rng.choice(args.epsilon_min_values))
        for _ in LEVELS
    )

    epsilon_decay = tuple(
        decay_log_uniform(
            rng,
            args.epsilon_decay_range[0],
            args.epsilon_decay_range[1],
        )
        for _ in LEVELS
    )

    weight_decay = tuple(
        float(rng.choice(args.adamw_weight_decay_values))
        for _ in LEVELS
    )

    # Avoid nonsensical replay settings.
    # If batch > buffer, replace batch with the largest allowed value
    # that fits, independently for that level.
    fixed_batches = []
    for b, buf in zip(batches, buffers):
        valid = [x for x in args.batch_sizes if x <= buf]
        fixed_batches.append(int(rng.choice(valid)))

    return {
        "optimizer": optimizers,
        "lr": learning_rates,
        "buffer": buffers,
        "batch": tuple(fixed_batches),
        "replay_steps": replay_steps,
        "eps_min": epsilon_min,
        "eps_decay": epsilon_decay,
        "weight_decay": weight_decay,
    }


def baseline_configuration():
    """
    Exact baseline from the supplied experiment.
    """
    return {
        "optimizer": ("adam", "adam", "adam", "adam"),
        "lr": (1e-3, 1e-3, 1e-3, 1e-3),
        "buffer": (1000, 1000, 1000, 1000),
        "batch": (32, 32, 32, 32),
        "replay_steps": (1, 1, 1, 1),
        "eps_min": (0.1, 0.1, 0.1, 0.1),
        "eps_decay": (0.995, 0.995, 0.999, 0.995),
        "weight_decay": (1e-4, 1e-4, 1e-4, 1e-4),
    }


def config_json(cfg):
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def config_id(cfg):
    """
    Return a Windows-safe, short, deterministic configuration ID.

    IMPORTANT:
    The complete hyperparameter configuration is stored in configs.csv.
    The cache directory name only needs to uniquely identify the
    configuration. Keeping it short avoids Windows component/path-length
    failures when four levels each have many hyperparameters.
    """
    digest = hashlib.sha1(
        config_json(cfg).encode("utf-8")
    ).hexdigest()[:16]

    return f"cfg_{digest}"


def generate_configurations(args):
    rng = np.random.default_rng(args.search_seed)

    configs = []
    seen = set()

    # Always evaluate the supplied baseline first.
    baseline = baseline_configuration()
    configs.append(baseline)
    seen.add(config_json(baseline))

    while len(configs) < args.n_configs:
        cfg = sample_configuration(rng, args)
        key = config_json(cfg)

        if key in seen:
            continue

        seen.add(key)
        configs.append(cfg)

    return configs


# ============================================================
# Agent construction
# ============================================================

def make_agent_params(cfg, level_index: int):
    return {
        "optimizer": make_optimizer(
            cfg["optimizer"][level_index],
            cfg["lr"][level_index],
            cfg["weight_decay"][level_index],
        ),
        "experience_replay_buffer_size": cfg["buffer"][level_index],
        "experience_replay_batch_size": cfg["batch"][level_index],
        "experience_replay_steps": cfg["replay_steps"][level_index],
        "epsilon_min": cfg["eps_min"][level_index],
        "epsilon_decay": cfg["eps_decay"][level_index],
    }


def create_agent_factory(scenario, args, cfg, seed):
    return MapcDQNAgentFactory(
        associations=scenario.associations,
        agent_params_lvl1=make_agent_params(cfg, 0),
        agent_params_lvl2=make_agent_params(cfg, 1),
        agent_params_lvl3=make_agent_params(cfg, 2),
        agent_params_lvl4=make_agent_params(cfg, 3),
        n_tx_power_levels=args.n_tx_power_levels,
        n_links=args.n_links,
        seed=seed,
    )


# ============================================================
# Experiment
# ============================================================

def run_single_experiment(scenario, args, cfg, run_number, key):
    factory = create_agent_factory(
        scenario=scenario,
        args=args,
        cfg=cfg,
        seed=args.seed + run_number,
    )

    agent = factory.create_hierarchical_DQN_cmapc_agent(logger=None)

    throughputs = np.zeros(args.n_steps, dtype=np.float32)
    previous_throughput = 0.0

    scenario.reset()

    for step in tqdm(
        range(1, args.n_steps),
        desc=f"run {run_number + 1}/{args.n_runs}",
        leave=False,
        dynamic_ncols=True,
        unit="step",
    ):
        key, step_key = jax.random.split(key)

        tx_config = agent.sample(reward=previous_throughput)

        data_rate = scenario(step_key, tx_config)
        throughputs[step] = float(data_rate)
        previous_throughput = float(data_rate)

    return throughputs


def run_config(args, scenario, cfg, run_keys, config_cache_dir):
    """
    Per-run checkpointing.

    If the process crashes after run 3/5, runs 0-2 remain on disk and
    only runs 3-4 are resumed.
    """
    # Keep cache directory names short and Windows-safe.
    # Full configuration details live in configs.csv.
    cid = config_id(cfg)
    cfg_dir = config_cache_dir / cid
    cfg_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    for run_number in range(args.n_runs):
        run_path = cfg_dir / f"run_{run_number:03d}.npz"

        if run_path.exists() and not args.force:
            with np.load(run_path) as z:
                tp = z["throughput"]

            if tp.shape == (args.n_steps,):
                all_runs.append(tp)
                continue

        tp = run_single_experiment(
            scenario=scenario,
            args=args,
            cfg=cfg,
            run_number=run_number,
            key=run_keys[run_number],
        )

        np.savez_compressed(
            run_path,
            throughput=tp,
            run_number=np.int64(run_number),
        )

        all_runs.append(tp)

    result = np.stack(all_runs, axis=0)

    np.savez_compressed(
        cfg_dir / "complete.npz",
        throughputs=result,
    )

    return result


# ============================================================
# Metrics
# ============================================================

def moving_average(x, window):
    if window <= 1:
        return x

    if x.shape[-1] < window:
        return x

    kernel = np.ones(window, dtype=np.float64) / window

    if x.ndim == 1:
        return np.convolve(x, kernel, mode="valid")

    return np.asarray([
        np.convolve(row, kernel, mode="valid")
        for row in x
    ])


def mean_ci(x, confidence):
    n = x.shape[0]
    mean = np.mean(x, axis=0)

    if n < 2:
        return mean, mean, mean

    std = np.std(x, axis=0, ddof=1)
    alpha = 1.0 - confidence
    crit = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
    half = crit * std / np.sqrt(n)

    return mean, mean - half, mean + half


def phase_metrics(tp, start, end, args):
    phase = tp[:, start:end]

    if phase.shape[1] == 0:
        raise ValueError("Empty phase.")

    window = min(args.window_size, phase.shape[1])
    sm = moving_average(phase, window)

    curve = np.mean(sm, axis=0)

    plateau_len = max(1, int(0.20 * len(curve)))
    plateau_curve = curve[-plateau_len:]

    plateau = float(np.mean(plateau_curve))
    stab_std = float(np.std(plateau_curve))

    run_plateaus = np.mean(sm[:, -plateau_len:], axis=1)
    run_std = (
        float(np.std(run_plateaus, ddof=1))
        if len(run_plateaus) > 1
        else 0.0
    )

    cv = stab_std / max(abs(plateau), 1e-12)
    run_cv = run_std / max(abs(plateau), 1e-12)

    # Time to 90% of the explicit near-optimal throughput target.
    target = 0.90 * args.target_throughput
    idx = np.flatnonzero(curve >= target)

    if len(idx):
        t90_target = int(idx[0] + window)
    else:
        t90_target = int(args.n_steps)

    # Time to 90% of this configuration's own plateau.
    own_target = 0.90 * plateau
    own_idx = np.flatnonzero(curve >= own_target)

    if len(own_idx):
        t90_own = int(own_idx[0] + window)
    else:
        t90_own = int(args.n_steps)

    return {
        "mean": float(np.mean(phase)),
        "plateau": plateau,
        "t90_target": t90_target,
        "t90_own": t90_own,
        "stab_std": stab_std,
        "cv": cv,
        "run_std": run_std,
        "run_cv": run_cv,
    }


def compute_metrics(results, args):
    switch = args.n_steps // 2
    metrics = {}

    for cfg_key, tp in results.items():
        p1 = phase_metrics(tp, 1, switch, args)
        p2 = phase_metrics(tp, switch, args.n_steps, args)

        overall = float(np.mean(tp[:, 1:]))

        # Stable throughput score:
        # reward high throughput while gently penalizing both kinds of variance.
        p2_stability = math.exp(
            -2.0 * p2["cv"] - 1.5 * p2["run_cv"]
        )
        p1_stability = math.exp(
            -2.0 * p1["cv"] - 1.5 * p1["run_cv"]
        )
        stability = 0.35 * p1_stability + 0.65 * p2_stability

        # Speed score relative to the explicit near-optimal throughput target.
        # 0 = never/very slow; approaches 1 as t90 becomes small.
        speed_scale = max(1.0, 0.10 * args.n_steps)
        speed1 = math.exp(-p1["t90_target"] / speed_scale)
        speed2 = math.exp(-p2["t90_target"] / speed_scale)
        speed = 0.35 * speed1 + 0.65 * speed2

        # Throughput scores. Allow configurations above 1200 Mbps to be
        # rewarded, capped at 1.25x the target so throughput still matters.
        p1_thr = min(p1["plateau"] / args.target_throughput, 1.25)
        p2_thr = min(p2["plateau"] / args.target_throughput, 1.25)

        # Post-switch performance matters most.
        objective = (
            0.20 * p1_thr
            + 0.45 * p2_thr
            + 0.20 * speed
            + 0.15 * stability
        )

        metrics[cfg_key] = {
            "overall": overall,
            "p1": p1,
            "p2": p2,
            "speed_score": speed,
            "stability_score": stability,
            "objective": objective,
        }

    return metrics


# ============================================================
# Summary
# ============================================================

CSV_FIELDS = [
    "config_id",
    "objective",
    "overall",
    "speed_score",
    "stability_score",
]

for level in range(1, 5):
    CSV_FIELDS.extend([
        f"l{level}_optimizer",
        f"l{level}_lr",
        f"l{level}_buffer",
        f"l{level}_batch",
        f"l{level}_replay_steps",
        f"l{level}_epsilon_min",
        f"l{level}_epsilon_decay",
        f"l{level}_weight_decay",
    ])

for phase in ("p1", "p2"):
    CSV_FIELDS.extend([
        f"{phase}_mean",
        f"{phase}_plateau",
        f"{phase}_t90_target",
        f"{phase}_t90_own",
        f"{phase}_stab_std",
        f"{phase}_cv",
        f"{phase}_run_std",
        f"{phase}_run_cv",
    ])


def write_summary(metrics, cfg_lookup, path):
    ranked = sorted(
        metrics.items(),
        key=lambda kv: kv[1]["objective"],
        reverse=True,
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for cfg_key, m in ranked:
            cfg = cfg_lookup[cfg_key]

            row = {
                "config_id": config_id(cfg),
                "objective": m["objective"],
                "overall": m["overall"],
                "speed_score": m["speed_score"],
                "stability_score": m["stability_score"],
            }

            for i in range(4):
                row[f"l{i+1}_optimizer"] = cfg["optimizer"][i]
                row[f"l{i+1}_lr"] = cfg["lr"][i]
                row[f"l{i+1}_buffer"] = cfg["buffer"][i]
                row[f"l{i+1}_batch"] = cfg["batch"][i]
                row[f"l{i+1}_replay_steps"] = cfg["replay_steps"][i]
                row[f"l{i+1}_epsilon_min"] = cfg["eps_min"][i]
                row[f"l{i+1}_epsilon_decay"] = cfg["eps_decay"][i]
                row[f"l{i+1}_weight_decay"] = cfg["weight_decay"][i]

            for phase in ("p1", "p2"):
                pm = m[phase]
                for key in (
                    "mean",
                    "plateau",
                    "t90_target",
                    "t90_own",
                    "stab_std",
                    "cv",
                    "run_std",
                    "run_cv",
                ):
                    row[f"{phase}_{key}"] = pm[key]

            writer.writerow(row)


def write_configs(configs, path):
    fields = [
        "index",
        "config_id",
        "configuration_json",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, cfg in enumerate(configs, 1):
            writer.writerow({
                "index": i,
                "config_id": config_id(cfg),
                "configuration_json": config_json(cfg),
            })


def print_ranking(metrics, cfg_lookup, top_k):
    ranked = sorted(
        metrics.items(),
        key=lambda kv: kv[1]["objective"],
        reverse=True,
    )[:top_k]

    print("\n" + "=" * 150)
    print("TOP HDQN CONFIGURATIONS")
    print("=" * 150)
    print(
        f"{'rank':>4} "
        f"{'obj':>7} "
        f"{'overall':>9} "
        f"{'P1':>9} "
        f"{'P2':>9} "
        f"{'P1 t90':>8} "
        f"{'P2 t90':>8} "
        f"{'P2 CV':>8} "
        f"{'P2 runCV':>9}"
    )

    for rank, (key, m) in enumerate(ranked, 1):
        print(
            f"{rank:>4} "
            f"{m['objective']:>7.4f} "
            f"{m['overall']:>9.1f} "
            f"{m['p1']['plateau']:>9.1f} "
            f"{m['p2']['plateau']:>9.1f} "
            f"{m['p1']['t90_target']:>8} "
            f"{m['p2']['t90_target']:>8} "
            f"{m['p2']['cv']:>8.4f} "
            f"{m['p2']['run_cv']:>9.4f}"
        )

    for rank, (key, m) in enumerate(ranked[:3], 1):
        cfg = cfg_lookup[key]

        print(f"\n#{rank}  {config_id(cfg)}")
        for i in range(4):
            print(
                f"  L{i+1}: "
                f"optimizer={cfg['optimizer'][i]}, "
                f"lr={cfg['lr'][i]:.6g}, "
                f"buffer={cfg['buffer'][i]}, "
                f"batch={cfg['batch'][i]}, "
                f"replay={cfg['replay_steps'][i]}, "
                f"eps_min={cfg['eps_min'][i]:.6g}, "
                f"eps_decay={cfg['eps_decay'][i]:.8f}, "
                f"wd={cfg['weight_decay'][i]:.6g}"
            )


# ============================================================
# Plotting
# ============================================================

def draw_curve(ax, tp, args, label=None):
    x_data = tp[:, 1:]
    window = min(args.window_size, x_data.shape[1])

    sm = moving_average(x_data, window)
    mean, lo, hi = mean_ci(sm, args.confidence)

    x = np.arange(window, x_data.shape[1] + 1)

    ax.plot(x, mean, lw=1.6, label=label)
    ax.fill_between(x, lo, hi, alpha=0.12)

    switch = args.n_steps // 2
    ax.axvline(
        switch,
        color="black",
        linestyle="--",
        alpha=0.55,
    )

    ax.axhline(
        args.target_throughput,
        color="gray",
        linestyle=":",
        alpha=0.7,
    )


def plot_top(results, metrics, cfg_lookup, args, plot_dir):
    ranked = sorted(
        metrics.items(),
        key=lambda kv: kv[1]["objective"],
        reverse=True,
    )[:args.top_k]

    fig, ax = plt.subplots(figsize=(15, 8))

    for key, m in ranked:
        cfg = cfg_lookup[key]
        draw_curve(
            ax,
            results[key],
            args,
            label=(
                f"obj={m['objective']:.3f}, "
                f"P2={m['p2']['plateau']:.0f}, "
                f"{config_id(cfg)[:30]}"
            ),
        )

    ax.set_title(
        f"Top {len(ranked)} HDQN configurations "
        f"(mean of {args.n_runs} runs, "
        f"{int(args.confidence * 100)}% CI)"
    )
    ax.set_xlabel("simulation step")
    ax.set_ylabel(
        f"throughput (moving average, window={args.window_size})"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()

    fig.savefig(
        plot_dir / "top_configurations.png",
        dpi=160,
    )
    plt.close(fig)


def plot_best(results, metrics, cfg_lookup, args, plot_dir):
    best_key = max(
        metrics,
        key=lambda k: metrics[k]["objective"],
    )
    best = metrics[best_key]
    cfg = cfg_lookup[best_key]

    fig, ax = plt.subplots(figsize=(14, 7))

    draw_curve(
        ax,
        results[best_key],
        args,
        label=(
            f"BEST | P2={best['p2']['plateau']:.1f} Mbps | "
            f"objective={best['objective']:.4f}"
        ),
    )

    ax.set_title("Best HDQN configuration")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("throughput (Mbps)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        plot_dir / "best_learning_curve.png",
        dpi=180,
    )
    plt.close(fig)

    (plot_dir / "best_configuration.txt").write_text(
        "\n".join(
            [
                config_id(cfg),
                "",
                json.dumps(cfg, indent=2),
                "",
                f"objective={best['objective']}",
                f"overall={best['overall']}",
                f"P1={best['p1']}",
                f"P2={best['p2']}",
            ]
        ),
        encoding="utf-8",
    )


def plot_scatter(metrics, plot_dir):
    values = list(metrics.values())

    x = np.asarray([
        m["p2"]["plateau"]
        for m in values
    ])

    y = np.asarray([
        m["p2"]["cv"]
        for m in values
    ])

    c = np.asarray([
        m["p2"]["t90_target"]
        for m in values
    ])

    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        x,
        y,
        c=c,
        s=20,
        alpha=0.65,
    )

    fig.colorbar(
        sc,
        ax=ax,
        label="Phase-2 t90 to 90% of target (steps)",
    )

    ax.set_xlabel("Phase-2 plateau throughput (Mbps)")
    ax.set_ylabel("Phase-2 coefficient of variation")
    ax.set_title(
        "HDQN throughput vs stability vs learning speed"
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        plot_dir / "throughput_vs_stability.png",
        dpi=160,
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.screen_runs is not None:
        args.n_runs = args.screen_runs

    if args.screen_steps is not None:
        args.n_steps = args.screen_steps

    if args.n_runs < 1 or args.n_steps < 2:
        raise ValueError("n-runs must be >=1 and n-steps >=2")

    # JAX persistent compilation cache.
    jax_cache = Path("./jax_cache")
    jax_cache.mkdir(parents=True, exist_ok=True)

    jax.config.update(
        "jax_compilation_cache_dir",
        str(jax_cache),
    )
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "all",
    )

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    plot_dir = output_dir / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    scenario = MixScen(
        small_office_scenario,
        args.d_sta_1,
        args.d_sta_2,
        args.d_ap,
        max_steps=args.n_steps,
    )

    configs = generate_configurations(args)

    write_configs(
        configs,
        output_dir / "configs.csv",
    )

    # Exactly the same environment randomness for every configuration.
    run_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        args.n_runs,
    )

    print("=" * 100)
    print("AGGRESSIVE HDQN HYPERPARAMETER TUNING")
    print("=" * 100)
    print(f"Configurations : {len(configs)}")
    print(f"Runs/config    : {args.n_runs}")
    print(f"Steps/run     : {args.n_steps}")
    print(
        f"Potential sims : "
        f"{len(configs) * args.n_runs}"
    )
    print(
        f"Scenario       : "
        f"d_ap={args.d_ap}, "
        f"d_sta={args.d_sta_1}->{args.d_sta_2}"
    )
    print(
        f"Target         : "
        f"{args.target_throughput} Mbps"
    )
    print(
        "Independent    : "
        "optimizer, lr, buffer, batch, replay, "
        "epsilon_min, epsilon_decay, weight_decay"
    )
    print(f"Output         : {output_dir}")
    print("=" * 100)

    results = {}
    cfg_lookup = {}

    # Metrics are recomputed and summary is rewritten after every completed
    # configuration, so useful progress survives a crash.
    config_progress = tqdm(
        enumerate(configs, 1),
        total=len(configs),
        desc="Configurations",
        unit="config",
        dynamic_ncols=True,
    )

    for index, cfg in config_progress:
        key = config_json(cfg)
        cfg_lookup[key] = cfg

        config_progress.set_postfix_str(
            config_id(cfg)[:24],
            refresh=False,
        )

        print(
            f"\n[{index}/{len(configs)}] "
            f"{config_id(cfg)}"
        )

        tp = run_config(
            args=args,
            scenario=scenario,
            cfg=cfg,
            run_keys=run_keys,
            config_cache_dir=cache_dir,
        )

        results[key] = tp

        # Incremental summary / plots.
        metrics = compute_metrics(results, args)
        write_summary(
            metrics,
            cfg_lookup,
            output_dir / "summary.csv",
        )

        if len(results) >= 2:
            plot_top(
                results,
                metrics,
                cfg_lookup,
                args,
                plot_dir,
            )
            plot_scatter(
                metrics,
                plot_dir,
            )

        print_ranking(
            metrics,
            cfg_lookup,
            min(args.top_k, len(metrics)),
        )

    # Final plots.
    metrics = compute_metrics(results, args)

    plot_top(
        results,
        metrics,
        cfg_lookup,
        args,
        plot_dir,
    )

    plot_best(
        results,
        metrics,
        cfg_lookup,
        args,
        plot_dir,
    )

    plot_scatter(
        metrics,
        plot_dir,
    )

    write_summary(
        metrics,
        cfg_lookup,
        output_dir / "summary.csv",
    )

    metadata = {
        "n_configs": len(configs),
        "n_runs": args.n_runs,
        "n_steps": args.n_steps,
        "seed": args.seed,
        "search_seed": args.search_seed,
        "d_ap": args.d_ap,
        "d_sta_1": args.d_sta_1,
        "d_sta_2": args.d_sta_2,
        "n_links": args.n_links,
        "n_tx_power_levels": args.n_tx_power_levels,
        "target_throughput": args.target_throughput,
        "confidence": args.confidence,
        "output_dir": str(output_dir),
    }

    (output_dir / "search_metadata.txt").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("TUNING COMPLETE")
    print("=" * 100)
    print(f"Summary : {output_dir / 'summary.csv'}")
    print(f"Configs : {output_dir / 'configs.csv'}")
    print(f"Cache   : {cache_dir}")
    print(f"Plots   : {plot_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
