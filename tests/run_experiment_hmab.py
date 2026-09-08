import argparse
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm

from mapc_cmab.agents.hierarchical_mab_mapc_agent import HierarchicalMABMapcAgent
from reinforced_lib.agents.mab  import UCB
from mapc_cmab.agents.mapc_hmab_agent_factory import MapcMABAgentFactory
from mapc_cmab.envs.scenario_impl import residential_scenario
from mapc_cmab.loggers.action_reward_logger import Logger 
from mapc_cmab.plots.throughput_analysis.throughput_ci import analyze_and_plot_throughputs

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sequential Hierarchical DQN MAPC experiments."
    )

    parser.add_argument("--d-ap", type=float, default=10.0,
                        help="AP-to-AP distance used by the scenario.")
    parser.add_argument("--d-sta", type=float, default=2.0,
                        help="Station distance used by the scenario.")
    parser.add_argument("--n-runs", type=int, default=10,
                        help="Number of independent runs.")
    parser.add_argument("--n-steps", type=int, default=5000,
                        help="Number of simulation steps per run.")
    parser.add_argument("--n-links", type=int, default=3,
                        help="Number of available links.")
    parser.add_argument("--n-tx-power-levels", type=int, default=4,
                        help="Number of transmission-power levels.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Number of steps per throughput averaging window.")
    parser.add_argument("--step-duration", type=float, default=0.005,
                        help="Duration of one simulation step in seconds.")
    parser.add_argument("--confidence", type=float, default=0.99,
                        help="Confidence level for the Student-t interval.")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory for output files.")
    parser.add_argument("--filename", type=str, default=None,
                        help="Base name for output files. Auto-generated if omitted.")
    parser.add_argument("--show", action="store_true",
                        help="Display the throughput plot.")

    return parser.parse_args()


def create_agent_factory(scenario, args):
    return MapcMABAgentFactory(
            associations=scenario.associations,
            agent_type=UCB,
            agent_params_lvl1={
                "c": 95.0878460790544,
                "gamma": 0.8768231620396211
            },
            agent_params_lvl2={
                "c": 95.0878460790544,
                "gamma": 0.8768231620396211
            },
            agent_params_lvl3={
                "c": 2.08,
                "gamma": 0.98,
            },
            agent_params_lvl4={
                "c": 1.5,
                "gamma": 0.99
            },
            n_tx_power_levels=args.n_tx_power_levels,
            n_links=args.n_links,
        )


def run_single_experiment(agent_factory, scenario, run_number, n_steps, key):
    # logger = Logger(run_number=run_number, exp_name=f"{scenario.str_repr}_HMAB_UCB")
    agent = agent_factory.create_hierarchical_mapc_agent(logger=None)

    throughputs = np.zeros(n_steps, dtype=np.float32)
    previous_throughput = 0.0

    for step in tqdm(range(1, n_steps), desc=f"run_number: {run_number}", leave=True):
        key, step_key = jax.random.split(key)

        tx_config = agent.sample(reward=previous_throughput)

        data_rate, _ = scenario(step_key, tx_config)
        data_rate = float(data_rate)

        throughputs[step] = data_rate
        previous_throughput = data_rate

    # logger.save(directory=f"logs/{scenario.str_repr}")

    return throughputs


def run_experiments(scenario, args):
    agent_factory = create_agent_factory(scenario, args)
    run_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_runs)

    throughputs = np.zeros(
        (args.n_runs, args.n_steps),
        dtype=np.float32,
    )

    for run_number in tqdm(range(args.n_runs), desc="Running experiments"):
        throughputs[run_number] = run_single_experiment(
            agent_factory=agent_factory,
            scenario=scenario,
            run_number=run_number,
            n_steps=args.n_steps,
            key=run_keys[run_number],
        )

    return throughputs


def main():
    args = parse_args()

    # Set a local directory for the persistent compilation cache (FIXED NAME)
    jax.config.update("jax_compilation_cache_dir", "./jax_cache")
    # Enable all extra XLA caching features ("all")
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

    filename = args.filename
    if filename is None:
        filename = (
            "HMAB_UCB"
            f"d_ap_{args.d_ap:g}_"
            f"d_sta_{args.d_sta:g}_"
            f"runs_{args.n_runs}_"
            f"steps_{args.n_steps}"
        )

    # scenario = small_office_scenario(
    #     d_ap=args.d_ap,
    #     d_sta=args.d_sta,
    #     n_tx_power_levels=args.n_tx_power_levels
    # )

    scenario = residential_scenario(
        x_apartments=5, 
        y_apartments=3, 
        n_sta_per_ap=4, 
        size=10, 
        seed=args.seed 
    ) 

    scenario.plot_rs()

    throughputs = run_experiments(
        scenario=scenario,
        args=args,
    )

    analyze_and_plot_throughputs(
        throughputs=throughputs,
        window_size=args.window_size,
        step_duration=args.step_duration,
        confidence=args.confidence,
        output_dir=args.output_dir,
        filename=filename,
        show=args.show,
    )


if __name__ == "__main__":
    main()
