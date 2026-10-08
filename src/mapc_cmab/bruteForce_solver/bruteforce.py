import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial
from itertools import chain, combinations, product
from typing import Iterable

import jax
import jax.numpy as jnp
from mapc_cmab.mapc_sim_mlo.sim import network_data_rate
from mapc_cmab.mapc_sim_mlo.constants import MAX_TX_POWER
from tqdm import tqdm

from mapc_cmab.envs.static_scenario import StaticScenario
from mapc_cmab.envs.test_scenarios import ALL_SCENARIOS


def iter_tx(associations: dict) -> Iterable:
    aps = set(associations)

    for active in chain.from_iterable(combinations(aps, r) for r in range(1, len(aps) + 1)):
        for stations in product(*((s for s in associations[a]) for a in active)):
            yield tuple(zip(active, stations))


def run_brute_force(scenario: StaticScenario, delta_tx_power: float = 2.0, tx_power_levels: int = 4) -> tuple:
    associations = scenario.get_associations()
    n_nodes = len(scenario.pos)

    data_rate_fn = jax.jit(partial(
        network_data_rate,
        pos=scenario.pos,
        mcs=None,
        sigma=0.0,
        walls=scenario.walls,
        channel_width=scenario.channel_width
    ))
    best_val, best_conf = 0, None

    for conf in iter_tx(associations):
        for power in product(range(tx_power_levels), repeat=len(conf)):
            tx_matrix = jnp.zeros((n_nodes, n_nodes), dtype=int)
            tx_power_matrix = jnp.zeros(n_nodes, dtype=int)

            for ap, sta in conf:
                tx_matrix = tx_matrix.at[ap, sta].set(1)

            for (ap, _), tx in zip(conf, power):
                tx_power_matrix = tx_power_matrix.at[ap].set(tx)

            thr = data_rate_fn(jax.random.PRNGKey(0), tx_matrix, tx_power=MAX_TX_POWER - delta_tx_power * tx_power_matrix)

            if thr > best_val:
                best_val, best_conf = thr, (conf, power)

    return best_conf, best_val.item()


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--output', type=str, default='all_results.json')
    args = args.parse_args()

    all_results = []

    for scenario in tqdm(ALL_SCENARIOS, desc='Scenarios'):
        scenario_results = []
        split_rate, split_conf = [], []

        for static, _ in scenario.split_scenario():
            conf, rate = run_brute_force(static)
            split_conf.append(conf)
            split_rate.append(rate)

        scenario_results.append({
            'agent': 'brute_force',
            'conf': split_conf,
            'runs': split_rate
        })

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
