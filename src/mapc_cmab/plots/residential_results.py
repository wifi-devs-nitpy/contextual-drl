import json

import numpy as np
import matplotlib.pyplot as plt

from mapc_cmab.plots.config import AGENT_COLORS, PLOT_PARAMS


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    residential_scenario_start_idx = 3
    residential_scenario_end_idx = 18

    with open('../dcf/mean_dcf_results.json') as f:
        dcf_results = json.load(f)[residential_scenario_start_idx:residential_scenario_end_idx]
        dcf_results = np.asarray(list(dcf_results)).flatten()

    with open('../dcf/mean_sr_results.json') as f:
        sr_results = json.load(f)[residential_scenario_start_idx:residential_scenario_end_idx]
        sr_results = np.asarray(list(sr_results)).flatten()

    with open('../mab/mean_mab_h_results.json') as f:
        mab_h_results = json.load(f)[residential_scenario_start_idx:residential_scenario_end_idx]
        mab_h_results = np.asarray(list(mab_h_results)).flatten()

    with open('../mab/mean_mab_f_results.json') as f:
        mab_f_results = json.load(f)[residential_scenario_start_idx:residential_scenario_end_idx]
        mab_f_results = [x or [-100] for x in mab_f_results]
        mab_f_results = np.asarray(list(mab_f_results)).flatten()

    with open('../upper_bound/all_results.json') as f:
        optimal_results = json.load(f)[residential_scenario_start_idx:residential_scenario_end_idx]
        t_optimal_results = [o[0]['runs'] for o in optimal_results]
        t_optimal_results = np.asarray(list(t_optimal_results)).flatten()
        f_optimal_results = [o[1]['runs'] for o in optimal_results]
        f_optimal_results = np.asarray(list(f_optimal_results)).flatten()

    results = {
        'DCF': dcf_results,
        'SR': sr_results,
        'MAB': mab_f_results,
        'H-MAB': mab_h_results,
        'F-Optimal': f_optimal_results,
        'T-Optimal': t_optimal_results,
    }

    for distance, mod in zip([10, 20, 30], [0, 1, 2]):
        fig, ax = plt.subplots()

        mask = (np.arange(residential_scenario_end_idx - residential_scenario_start_idx) % 3) == mod
        xs = np.arange(5)
        barwidth = 0.12

        for i, column in enumerate(results.keys()):
            if column == "T-Optimal":
                ax.bar(xs, results[column][mask], color="gray", width=5 * barwidth, label=column, alpha=0.5)

        for i, column in enumerate(results.keys()):
            if column != "T-Optimal":
                ax.bar(xs + (i - 2) * barwidth, results[column][mask], color=AGENT_COLORS[column], width=barwidth, label=column)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["2x2", "2x3", "3x3", "3x4", "4x4"])
        ax.set_ylim(-30, ax.get_ylim()[1])
        ax.set_ylabel('Effective data rate [Mb/s]')
        ax.legend(loc='upper left', fontsize=6, ncols=2)

        handles, labels = ax.get_legend_handles_labels()
        order = [1, 3, 5, 2, 4, 0]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left', fontsize=6, ncol=2)

        plt.grid(axis='y')
        plt.savefig(f'results_residential_d{distance}.pdf', bbox_inches='tight')
        plt.show()