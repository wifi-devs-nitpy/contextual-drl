import json

import numpy as np
import matplotlib.pyplot as plt
from mapc_cmab.mapc_sim_mlo.constants import TAU

from mapc_cmab.plots.config import COLUMN_WIDTH, COLUMN_HEIGHT, AGENT_COLORS
from mapc_cmab.plots.utils import confidence_interval


plt.rcParams.update({
    'figure.figsize': (3 * COLUMN_WIDTH, COLUMN_HEIGHT + 0.7),
    'legend.fontsize': 9
})

N_STEPS = [600, 1200, 1200]
AGGREGATE_STEPS = [20, 40, 40]
SWITCH_STEPS = [None, 600, 600]
TITLES = [r"(a) Multi-room $2 \times 2$, $\rho=10$ m", r"(b) Multi-room $2 \times 2$, $\rho=20$ m", r"(c) Open space"]


if __name__ == '__main__':
    with open('../mab/all_mab_h_results.json') as file:
        mab_h_results = json.load(file)[:3]

    with open('../mab/all_mab_f_results.json') as file:
        mab_f_results = json.load(file)[:3]

    dcf_results = []

    with open('dcf/exemplary/static_residential_100_2_2_4_10.0.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('dcf/exemplary/dynamic_static_residential_101_2_2_4_20.0_a.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('dcf/exemplary/dynamic_static_residential_101_2_2_4_20.0_b.json') as file:
        dcf_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('dcf/exemplary/dynamic_random_3_75.0_4_4.0_4_a.json') as file:
        dcf_results.append([json.load(file)['DataRate']['Mean']])

    with open('dcf/exemplary/dynamic_random_3_75.0_4_4.0_4_b.json') as file:
        dcf_results[-1].append(json.load(file)['DataRate']['Mean'])

    sr_results = []

    with open('sr/exemplary/static_residential_100_2_2_4_10.0.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/exemplary/dynamic_static_residential_101_2_2_4_20.0_a.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/exemplary/dynamic_static_residential_101_2_2_4_20.0_b.json') as file:
        sr_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('sr/exemplary/dynamic_random_3_75.0_4_4.0_4_a.json') as file:
        sr_results.append([json.load(file)['DataRate']['Mean']])

    with open('sr/exemplary/dynamic_random_3_75.0_4_4.0_4_b.json') as file:
        sr_results[-1].append(json.load(file)['DataRate']['Mean'])

    with open('../upper_bound/all_results.json') as file:
        optimal_results = json.load(file)[:3]

    fig, axes = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0.)

    for i, ax in enumerate(axes):
        xs = np.linspace(0, N_STEPS[i], N_STEPS[i] // AGGREGATE_STEPS[i]) * TAU

        if SWITCH_STEPS[i] is not None:
            ax.axvline(SWITCH_STEPS[i] * TAU, linestyle='--', color='gray')

            xs_first, xs_sec = xs[:SWITCH_STEPS[i] // AGGREGATE_STEPS[i]], xs[SWITCH_STEPS[i] // AGGREGATE_STEPS[i]:]
            x_mid = (xs_first[-1] + xs_sec[0]) / 2
            xs_first = np.concatenate((xs_first, [x_mid]))
            xs_sec = np.concatenate(([x_mid], xs_sec))

            ax.plot(xs_first, len(xs_first) * [dcf_results[i][0]], c=AGENT_COLORS['DCF'])
            ax.plot(xs_sec, len(xs_sec) * [dcf_results[i][1]], c=AGENT_COLORS['DCF'])
            ax.plot(xs_first, len(xs_first) * [sr_results[i][0]], c=AGENT_COLORS['SR'])
            ax.plot(xs_sec, len(xs_sec) * [sr_results[i][1]], c=AGENT_COLORS['SR'])
            ax.plot(xs_first, len(xs_first) * [optimal_results[i][0]['runs'][0]], c=AGENT_COLORS['T-Optimal'])
            ax.plot(xs_sec, len(xs_sec) * [optimal_results[i][0]['runs'][1]], c=AGENT_COLORS['T-Optimal'])
        else:
            ax.plot(xs, len(xs) * [dcf_results[i][0]], c=AGENT_COLORS['DCF'])
            ax.plot(xs, len(xs) * [sr_results[i][0]], c=AGENT_COLORS['SR'])
            ax.plot(xs, len(xs) * [optimal_results[i][0]['runs'][0]], c=AGENT_COLORS['T-Optimal'])

        n_rep = len(mab_h_results[i][0]['runs'])
        mab_h = np.asarray(mab_h_results[i][0]['runs']).reshape((n_rep, -1, AGGREGATE_STEPS[i])).mean(axis=-1)

        mean, ci_low, ci_high = confidence_interval(mab_h)
        ax.plot(xs, mean, c=AGENT_COLORS['H-MAB'], marker='o')
        ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=AGENT_COLORS['H-MAB'], linewidth=0.0)

        n_rep = len(mab_f_results[i][0]['runs'])
        mab_f = np.asarray(mab_f_results[i][0]['runs']).reshape((n_rep, -1, AGGREGATE_STEPS[i])).mean(axis=-1)

        mean, ci_low, ci_high = confidence_interval(mab_f)
        ax.plot(xs, mean, c=AGENT_COLORS['MAB'], marker='o')
        ax.fill_between(xs, ci_low, ci_high, alpha=0.3, color=AGENT_COLORS['MAB'], linewidth=0.0)

        ax.set_title(TITLES[i], y=-0.45, fontsize=12)
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_xlim((xs[0], xs[-1]))
        ax.set_ylim(bottom=0, top=600)
        ax.set_yticks(range(0, 601, 100))
        ax.tick_params(axis='both', which='both', labelsize=10)
        ax.grid()

        if i == 0:
            ax.set_ylabel('Effective data rate [Mb/s]', fontsize=12)
            ax.plot([], [], marker='o', c=AGENT_COLORS['MAB'], label='MAB')
            ax.plot([], [], marker='o', c=AGENT_COLORS['H-MAB'], label='H-MAB')
            ax.legend(loc='upper right', fontsize=8)

            ax2 = ax.twinx()
            ax2.axis('off')
            ax2.plot([], [], c=AGENT_COLORS['DCF'], label='DCF (avg)')
            ax2.plot([], [], c=AGENT_COLORS['SR'], label='SR (avg)')
            ax2.plot([], [], c=AGENT_COLORS['T-Optimal'], label='T-Optimal')
            ax2.legend(loc='upper left', title='Baselines', ncol=2, fontsize=8)
        else:
            ax.text(3.5, 495, 'Stations move\nto new positions', fontsize=8)
            ax.annotate('', xy=(3.4, 410), xytext=(4.2, 480), arrowprops=dict(arrowstyle='->', color='gray', lw=0.75))

    plt.tight_layout()
    plt.savefig(f'data-rates.pdf', bbox_inches='tight')
    plt.show()
