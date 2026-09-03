import os
os.environ['JAX_ENABLE_X64'] = 'True'

from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from mapc_cmab.envs.scenario_impl import random_scenario
from mapc_cmab.plots.config import COLUMN_WIDTH, AGENT_COLORS


if __name__ == '__main__':
    scenario = random_scenario(seed=3, d_ap=50., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=10).scenario_sec

    aps = np.array(list(scenario.associations.keys()))
    ap_pos = np.array(scenario.pos[aps])
    stas = np.array(list(chain.from_iterable(scenario.associations.values())))
    sta_pos = np.array(scenario.pos[stas])

    sta_pos[0, 0] -= 2.0
    sta_pos[7, 1] -= 1.0

    plt.rcParams['axes.linewidth'] = 0.0
    plt.rcParams['figure.figsize'] = (0.8 * COLUMN_WIDTH, 0.8 * COLUMN_WIDTH)
    colors = ['#1f78b4', '#33a02c', '#ffbb22', '#e31a1c']

    plt.scatter(ap_pos[:, 0], ap_pos[:, 1], color=colors, marker='x', s=15)
    plt.scatter(sta_pos[:, 0], sta_pos[:, 1], color=np.repeat(colors, 4, axis=0), marker='o', s=2)

    for i, txt in enumerate([r'AP {}'.format(i + 1) for i in range(len(aps))]):
        plt.annotate(txt, (ap_pos[i, 0] - 4, ap_pos[i, 1] + 2), fontsize=8)

    ticks = np.arange(0, 51, 50 / 6)
    tick_names = [0, '', 25, '', 50, '', 75]

    plt.xticks(ticks, tick_names)
    plt.yticks(ticks, tick_names)
    plt.xlim(0, 50.5)
    plt.ylim(0, 50.5)
    plt.ylabel(r'Y [m]')
    plt.xlabel(r'X [m]')

    plt.tight_layout()
    plt.grid()
    plt.savefig('random_scenario.pdf', bbox_inches='tight')
    plt.show()
