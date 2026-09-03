from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from mapc_cmab.envs.scenario_impl import residential_scenario
from mapc_cmab.plots.config import COLUMN_WIDTH, get_cmap


if __name__ == '__main__':
    scenario = residential_scenario(seed=100, n_steps=10, x_apartments=3, y_apartments=3, n_sta_per_ap=4, size=10.0)
    offset = 2

    aps = np.asarray(list(scenario.associations.keys()))
    ap_pos = np.asarray(scenario.pos[aps]) + offset
    stas = np.asarray(list(chain.from_iterable(scenario.associations.values())))
    sta_pos = np.asarray(scenario.pos[stas]) + offset

    ap_pos[1, 0] += 0.5
    ap_pos[3, 0] += 0.5
    ap_pos[4, 1] += 0.5
    ap_pos[7, 1] -= 1.0
    sta_pos[2, 1] -= 0.5
    sta_pos[12, 1] += 1.0
    sta_pos[13, 0] += 0.5
    sta_pos[14, 1] += 0.5
    sta_pos[16, 0] -= 1.0
    sta_pos[23, 0] -= 0.5
    sta_pos[26, 1] += 1.0
    sta_pos[27, 1] += 0.5
    sta_pos[33, 0] += 1.5
    sta_pos[33, 1] += 1.5
    sta_pos[34, 0] += 1.5

    plt.rcParams['axes.linewidth'] = 0.0
    plt.rcParams['figure.figsize'] = (0.8 * COLUMN_WIDTH, 0.8 * COLUMN_WIDTH)
    color = get_cmap(4)[1]

    for i in range(offset, 31 + offset, 10):
        plt.plot([offset, 30 + offset], [i, i], color='black')
        plt.plot([i, i], [offset, 30 + offset], color='black')

    plt.scatter(ap_pos[:, 0], ap_pos[:, 1], color=color, marker='x', s=15)
    plt.scatter(sta_pos[:, 0], sta_pos[:, 1], color=color, marker='o', s=2)

    for i, txt in enumerate([r'AP {}'.format(i + 1) for i in range(len(aps))]):
        plt.annotate(txt, (ap_pos[i, 0] - 1.75, ap_pos[i, 1] + 1), fontsize=7)

    ticks = np.arange(0, 31, 5) + offset
    tick_names = [0, '', r'$\rho$', '', r'$2\rho$', '', r'$3\rho$']

    plt.xticks(ticks, tick_names)
    plt.yticks(ticks, tick_names)
    plt.xlim(0, 30.15 + offset)
    plt.ylim(0, 30.15 + offset)
    plt.ylabel(r'Y [m]')
    plt.xlabel(r'X [m]')

    plt.tight_layout()
    plt.grid()
    plt.savefig('residential_scenario.pdf', bbox_inches='tight')
    plt.show()
