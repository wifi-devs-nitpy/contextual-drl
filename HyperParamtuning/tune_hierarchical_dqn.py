"""Overnight hyperparameter sweep + final 20x10k run for hierarchical MAPC DQN.

Workflow:
  1. Run a broad 64-configuration sweep to compare learning speed/stability.
  2. Select the best configuration using a reproducible score.
  3. Run the selected configuration for 20 independent reps x 10,000 steps.
  4. Save raw arrays, summary CSVs, and learning curves.

The discount factor is FIXED at 0.0 in the tuned factory and is never exposed
as a command-line hyperparameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from mapc_cmab.envs.scenario_impl import residential_scenario
from mapc_cmab.loggers.action_reward_logger import Logger
from mapc_cmab.plots.throughput_analysis.throughput_ci import analyze_and_plot_throughputs

from mapc_dqn_agent_factory_tuned import MapcDQNAgentFactory


LOG = logging.getLogger("overnight_dqn")


@dataclass(frozen=True)
class Config:
    name: str
    learning_rate: float
    buffer_size: int
    batch_size: int
    replay_steps: int
    epsilon_min: float
    epsilon_find_groups: float
    epsilon_assign_stations: float
    epsilon_assign_links: float
    epsilon_tx_power: float
    hidden_dims: tuple[int, ...]
    layer_norm: bool = False

    def to_factory_config(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "experience_replay_buffer_size": self.buffer_size,
            "experience_replay_batch_size": self.batch_size,
            "experience_replay_steps": self.replay_steps,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": {
                "find_groups": self.epsilon_find_groups,
                "assign_stations": self.epsilon_assign_stations,
                "assign_links": self.epsilon_assign_links,
                "tx_power": self.epsilon_tx_power,
            },
            "hidden_dims": self.hidden_dims,
            "use_layer_norm": self.layer_norm,
        }


# 64 configurations: 8 optimizer/replay/network regimes x 8 exploration profiles.
# Discount is intentionally absent: it is fixed at 0.0 in the factory.
_OPT_REPLAY_NET = [
    ("lr3e-4_b2k_bs32_r1", 3e-4, 2000, 32, 1, (32, 32), False),
    ("lr3e-4_b5k_bs32_r1", 3e-4, 5000, 32, 1, (64, 64), False),
    ("lr5e-4_b2k_bs32_r1", 5e-4, 2000, 32, 1, (64, 64), False),
    ("lr5e-4_b5k_bs64_r1", 5e-4, 5000, 64, 1, (64, 64), False),
    ("lr7e-4_b5k_bs64_r1", 7e-4, 5000, 64, 1, (128, 64), False),
    ("lr1e-3_b1k_bs32_r1", 1e-3, 1000, 32, 1, (64, 64), False),
    ("lr3e-4_b5k_bs64_r2", 3e-4, 5000, 64, 2, (128, 64), False),
    ("lr5e-4_b10k_bs64_r2", 5e-4, 10000, 64, 2, (128, 128), False),
]

# Exploration profiles are deliberately diverse. The four entries map to:
# find_groups, assign_stations, assign_links, tx_power.
_EPS_PROFILES = [
    ("eps_fast",    0.9950, 0.9950, 0.9980, 0.9950, 0.05),
    ("eps_modfast", 0.9970, 0.9970, 0.9990, 0.9970, 0.05),
    ("eps_balanced",0.9990, 0.9990, 0.9990, 0.9990, 0.05),
    ("eps_slow",    0.9995, 0.9995, 0.9995, 0.9995, 0.05),
    ("eps_floor01", 0.9990, 0.9990, 0.9990, 0.9990, 0.01),
    ("eps_floor10", 0.9990, 0.9990, 0.9990, 0.9990, 0.10),
    ("eps_hier",    0.9980, 0.9970, 0.9995, 0.9980, 0.05),
    ("eps_hier2",   0.9990, 0.9980, 0.9997, 0.9990, 0.02),
]

SWEEP_CONFIGS = []
for opt_name, lr, buf, batch, replay_steps, hidden_dims, layer_norm in _OPT_REPLAY_NET:
    for eps_name, eg, es, el, et, eps_min in _EPS_PROFILES:
        SWEEP_CONFIGS.append(
            Config(
                name=f"{opt_name}__{eps_name}",
                learning_rate=lr,
                buffer_size=buf,
                batch_size=batch,
                replay_steps=replay_steps,
                epsilon_min=eps_min,
                epsilon_find_groups=eg,
                epsilon_assign_stations=es,
                epsilon_assign_links=el,
                epsilon_tx_power=et,
                hidden_dims=hidden_dims,
                layer_norm=layer_norm,
            )
        )

assert len(SWEEP_CONFIGS) == 64, len(SWEEP_CONFIGS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("sweep", "final", "all"), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep-runs", type=int, default=3)
    p.add_argument("--sweep-steps", type=int, default=2500)
    p.add_argument("--final-runs", type=int, default=20)
    p.add_argument("--final-steps", type=int, default=10000)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--step-duration", type=float, default=0.005)
    p.add_argument("--confidence", type=float, default=0.99)
    p.add_argument("--output-dir", type=Path, default=Path("./results/overnight"))
    p.add_argument("--cache-dir", type=Path, default=Path("./jax_cache"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def make_scenario(seed: int):
    return residential_scenario(
        x_apartments=5,
        y_apartments=3,
        n_sta_per_ap=4,
        size=10,
        seed=seed,
    )


def safe_mean(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    y = y[np.isfinite(y)]
    return float(y.mean()) if y.size else float("nan")


def safe_std(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=np.float64)
    y = y[np.isfinite(y)]
    return float(y.std(ddof=1)) if y.size > 1 else 0.0


def curve_metrics(curve: np.ndarray) -> dict[str, float]:
    """Metrics aimed at reward level + stability + sample efficiency."""
    y = np.asarray(curve, dtype=np.float64)
    tail_n = max(100, y.shape[-1] // 5)
    half_n = max(100, y.shape[-1] // 2)
    tail = y[-tail_n:]
    first_half = y[:half_n]

    tail_mean = safe_mean(tail)
    tail_std = safe_std(tail)
    cv = tail_std / max(abs(tail_mean), 1e-12)

    # A moving-average proxy for time-to-90%-of-tail. Lower is faster.
    smooth_n = max(25, min(200, y.shape[-1] // 20))
    kernel = np.ones(smooth_n) / smooth_n
    smooth = np.convolve(y, kernel, mode="valid")
    target = 0.90 * tail_mean
    hits = np.where(smooth >= target)[0]
    t90 = float(hits[0] + smooth_n) if hits.size else float(y.shape[-1])

    early_mean = safe_mean(first_half[-max(50, half_n // 5):])
    gain = early_mean / max(abs(tail_mean), 1e-12)

    return {
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "tail_cv": cv,
        "t90_steps": t90,
        "early_to_tail_ratio": gain,
    }


def aggregate_run_curves(throughputs: np.ndarray) -> dict[str, float]:
    mean_curve = np.mean(throughputs, axis=0)
    metrics = curve_metrics(mean_curve)
    per_run_tail = np.mean(throughputs[:, -max(100, throughputs.shape[1] // 5):], axis=1)
    metrics["run_tail_mean_std"] = safe_std(per_run_tail)
    metrics["run_tail_p10"] = float(np.percentile(per_run_tail, 10))
    metrics["run_tail_p90"] = float(np.percentile(per_run_tail, 90))
    return metrics


def run_single_experiment(config: Config, scenario, run_number: int, n_steps: int, seed: int) -> np.ndarray:
    # Separate factory per run avoids hidden coupling of factory-internal seeds.
    factory = MapcDQNAgentFactory(
        associations=scenario.associations,
        agent_params_lvl1=None,
        agent_params_lvl2=None,
        agent_params_lvl3=None,
        agent_params_lvl4=None,
        n_links=3,
        n_tx_power_levels=4,
        seed=seed,
        config=config.to_factory_config(),
    )
    logger = Logger(run_number=run_number, exp_name=f"{scenario.str_repr}_{config.name}")
    agent = factory.create_hierarchical_DQN_cmapc_agent(logger=logger)

    throughputs = np.zeros(n_steps, dtype=np.float32)
    previous_throughput = 0.0
    key = jax.random.PRNGKey(seed)

    start = time.perf_counter()
    for step in range(1, n_steps):
        key, step_key = jax.random.split(key)
        tx_config = agent.sample(reward=previous_throughput)
        data_rate, _ = scenario(step_key, tx_config)
        data_rate = float(data_rate)
        throughputs[step] = data_rate
        previous_throughput = data_rate

    elapsed = time.perf_counter() - start
    logger.save(directory=f"logs/{scenario.str_repr}")
    LOG.info(
        "config=%s run=%d finished in %.1fs | final-mean(last 5%%)=%.4f",
        config.name,
        run_number,
        elapsed,
        safe_mean(throughputs[-max(50, n_steps // 20):]),
    )
    return throughputs


def run_experiment_batch(config: Config, args: argparse.Namespace, n_runs: int, n_steps: int, phase: str) -> np.ndarray:
    LOG.info("Starting %s: %s | runs=%d steps=%d", phase, config.name, n_runs, n_steps)
    all_tp = np.zeros((n_runs, n_steps), dtype=np.float32)
    for run_number in range(n_runs):
        # Same scenario/PRNG seed across configs for a fair comparison.
        # The config only changes the learner, not the environment sample.
        run_seed = args.seed + run_number
        scenario = make_scenario(run_seed)
        all_tp[run_number] = run_single_experiment(
            config=config,
            scenario=scenario,
            run_number=run_number,
            n_steps=n_steps,
            seed=run_seed,
        )
    return all_tp


def save_config_json(config: Config, path: Path) -> None:
    payload = asdict(config)
    payload["hidden_dims"] = list(config.hidden_dims)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_sweep_curves(results: dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    for name, curves in results.items():
        mean = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        x = np.arange(mean.size)
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - std, mean + std, alpha=0.12)
    plt.xlabel("Simulation step")
    plt.ylabel("Throughput")
    plt.title("Hierarchical DQN hyperparameter sweep")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def rank_configs(metrics: dict[str, dict[str, float]]) -> list[str]:
    names = list(metrics)
    tail = np.array([metrics[n]["tail_mean"] for n in names], dtype=float)
    cv = np.array([metrics[n]["tail_cv"] for n in names], dtype=float)
    t90 = np.array([metrics[n]["t90_steps"] for n in names], dtype=float)

    def normalize(a, higher_better=True):
        lo, hi = np.nanmin(a), np.nanmax(a)
        if math.isclose(lo, hi):
            return np.full_like(a, 0.5)
        z = (a - lo) / (hi - lo)
        return z if higher_better else 1.0 - z

    # Reward level is the main objective, then stability, then learning speed.
    score = (
        0.60 * normalize(tail, True)
        + 0.25 * normalize(cv, False)
        + 0.15 * normalize(t90, False)
    )
    order = np.argsort(-score)
    for i in order:
        metrics[names[i]]["composite_score"] = float(score[i])
    return [names[i] for i in order]


def save_metrics_csv(metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    fieldnames = ["config"] + sorted({k for m in metrics.values() for k in m})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, data in metrics.items():
            writer.writerow({"config": name, **data})


def configure_jax(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    configure_jax(args.cache_dir)

    LOG.info("JAX backend: %s", jax.default_backend())
    LOG.info("Output directory: %s", args.output_dir.resolve())
    LOG.info("Discount factor is fixed at 0.0 and is not tuned.")

    results: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    if args.mode in ("sweep", "all"):
        for config in SWEEP_CONFIGS:
            curves = run_experiment_batch(
                config=config,
                args=args,
                n_runs=args.sweep_runs,
                n_steps=args.sweep_steps,
                phase="sweep",
            )
            results[config.name] = curves
            metrics[config.name] = aggregate_run_curves(curves)
            np.save(args.output_dir / f"sweep_{config.name}.npy", curves)
            save_config_json(config, args.output_dir / f"sweep_{config.name}.json")
            LOG.info("%s metrics: %s", config.name, metrics[config.name])

        save_metrics_csv(metrics, args.output_dir / "sweep_metrics.csv")

        ranking = rank_configs(metrics)
        plot_sweep_curves(
            results,
            args.output_dir / "sweep_learning_curves_all.png",
        )
        top_names = ranking[:12]
        plot_sweep_curves(
            {name: results[name] for name in top_names},
            args.output_dir / "sweep_learning_curves_top12.png",
        )
        (args.output_dir / "sweep_ranking.txt").write_text(
            "\n".join(f"{i + 1}. {name} score={metrics[name]['composite_score']:.4f}" for i, name in enumerate(ranking)),
            encoding="utf-8",
        )
        selected_name = ranking[0]
        selected = next(c for c in SWEEP_CONFIGS if c.name == selected_name)
        LOG.info("Selected config for final 20x10k run: %s", selected_name)
    else:
        selected = SWEEP_CONFIGS[0]  # deterministic baseline fallback for --mode final
        LOG.info("--mode final: using fallback config %s", selected.name)

    if args.mode in ("final", "all"):
        final_curves = run_experiment_batch(
            config=selected,
            args=args,
            n_runs=args.final_runs,
            n_steps=args.final_steps,
            phase="final",
        )
        np.save(args.output_dir / "final_throughputs.npy", final_curves)
        save_config_json(selected, args.output_dir / "final_config.json")

        final_metrics = aggregate_run_curves(final_curves)
        with (args.output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        LOG.info("Final metrics: %s", final_metrics)

        filename = f"hierarchical_dqn_{selected.name}_20runs_10ksteps"
        analyze_and_plot_throughputs(
            throughputs=final_curves,
            window_size=args.window_size,
            step_duration=args.step_duration,
            confidence=args.confidence,
            output_dir=str(args.output_dir),
            filename=filename,
            show=args.show,
        )
        LOG.info("Final analysis complete: %s", args.output_dir / filename)


if __name__ == "__main__":
    main()
