import argparse
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm
import optax 

from mapc_cmab.agents.hierarchical_mab_mapc_agent import HierarchicalMABMapcAgent
from mapc_cmab.agents.hierarchical_dqn import HierarchicalMapcDQNAgent
from reinforced_lib.agents.mab  import UCB
from mapc_cmab.agents.mapc_cmab_agent_factory import MapcDQNAgentFactory
from mapc_cmab.envs.scenario_impl import residential_scenario, small_office_scenario
from mapc_cmab.loggers.action_reward_logger import Logger 
from mapc_cmab.plots.throughput_analysis.throughput_ci import analyze_and_plot_throughputs


d_ap = 10
n_steps = 10_000

class MixScen:
    def __init__(self, scenario_factory, d_sta_1: int, d_sta_2: int,  d_ap=d_ap, max_steps: int = n_steps):
        self.scen1 = scenario_factory(d_ap=d_ap, d_sta=d_sta_1)
        self.scen2 = scenario_factory(d_ap=d_ap, d_sta=d_sta_2)
        self.step = 0
        self.switch_steps = max_steps // 2
        self.data_rate_fn1 = jax.jit(self.scen1.data_rate_fn)
        self.data_rate_fn2 = jax.jit(self.scen2.data_rate_fn)
        self.data_rate_fn = self.data_rate_fn1
        self.associations = self.scen1.associations
        self.str_repr = f"mix_scen_ap_{d_ap}_dsta_{d_sta_1}_{d_sta_2}_s{max_steps}"

    def __call__(self, key, link_ap_sta):
        self.step += 1
        if self.step == self.switch_steps:
            self.data_rate_fn = self.data_rate_fn2 if self.data_rate_fn is self.data_rate_fn1 else self.data_rate_fn1

        return self.data_rate_fn(key, link_ap_sta=link_ap_sta)

    def reset(self):
        self.data_rate_fn = self.data_rate_fn1
        self.step = 0


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
    parser.add_argument("--n-steps", type=int, default=10_000,
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



common_params = {
    "optimizer": optax.adam(1e-3),
    "experience_replay_buffer_size": 1000,
    "experience_replay_batch_size": 32,
    "experience_replay_steps": 1,
    "epsilon_min": 0.05,
}

agent_params_lvl1 = {
    **common_params,
    "epsilon_decay": 0.995,
}

agent_params_lvl2 = {
    **common_params,
    "epsilon_decay": 0.995,
}

agent_params_lvl3 = {
    **common_params,
    "epsilon_decay": 0.999,
}

agent_params_lvl4 = {
    **common_params,
    "epsilon_decay": 0.995,
}


def create_agent_factory(scenario, args):
    return MapcDQNAgentFactory(
        associations=scenario.associations,
        agent_params_lvl1=agent_params_lvl1,
        agent_params_lvl2=agent_params_lvl2,
        agent_params_lvl3=agent_params_lvl3,
        agent_params_lvl4=agent_params_lvl4,
        n_tx_power_levels=args.n_tx_power_levels,
        n_links=args.n_links,
        seed=args.seed,
    )



def run_single_experiment(agent_factory, scenario, run_number, n_steps, key):
    # logger = Logger(run_number=run_number, exp_name=f"{scenario.str_repr}_HMAB_UCB")
    agent = agent_factory.create_hierarchical_DQN_cmapc_agent(logger=None)
    throughputs = np.zeros(n_steps, dtype=np.float32)
    previous_throughput = 0.0

    scenario.reset()

    for step in tqdm(range(1, n_steps), desc=f"run_number: {run_number}", leave=True):
        key, step_key = jax.random.split(key)

        tx_config = agent.sample(reward=previous_throughput)

        data_rate = scenario(step_key, tx_config)
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

    
    
    # scenario = small_office_scenario(
    #     d_ap=args.d_ap,
    #     d_sta=args.d_sta,
    #     n_tx_power_levels=args.n_tx_power_levels
    # )

    # scenario = residential_scenario(
    #     x_apartments=5, 
    #     y_apartments=3, 
    #     n_sta_per_ap=4, 
    #     size=10, 
    #     seed=args.seed 
    # ) 

    scenario = MixScen(small_office_scenario, 2, 4, args.d_ap, max_steps=args.n_steps)

    filename = args.filename
    if filename is None:
        filename = (
            f"updated_dqn_inital_"
            f"{scenario.str_repr}"
            f"runs_{args.n_runs}_"
            f"steps_{args.n_steps}"
        )

    # scenario.plot_rs()

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
