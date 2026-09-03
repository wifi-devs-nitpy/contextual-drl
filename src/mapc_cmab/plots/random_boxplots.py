import json
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mapc_cmab.plots.config import PLOT_PARAMS


if __name__ == "__main__":
    plt.rcParams.update(PLOT_PARAMS)
    random_scenario_idx = 18

    with open('../dcf/mean_dcf_results.json') as f:
        dcf_results = json.load(f)[random_scenario_idx:]
        dcf_results = chain.from_iterable(dcf_results)
        dcf_results = np.asarray(list(dcf_results))

    with open('../dcf/mean_sr_results.json') as f:
        sr_results = json.load(f)[random_scenario_idx:]
        sr_results = chain.from_iterable(sr_results)
        sr_results = np.asarray(list(sr_results)) / dcf_results * 100

    with open('../mab/mean_mab_h_results.json') as f:
        mab_h_results = json.load(f)[random_scenario_idx:]
        mab_h_results = chain.from_iterable(mab_h_results)
        mab_h_results = np.asarray(list(mab_h_results)) / dcf_results * 100

    with open('../mab/mean_mab_f_results.json') as f:
        mab_f_results = json.load(f)[random_scenario_idx:]
        mab_f_results = chain.from_iterable(mab_f_results)
        mab_f_results = np.asarray(list(mab_f_results)) / dcf_results * 100

    with open('../upper_bound/all_results.json') as f:
        optimal_results = json.load(f)[random_scenario_idx:]
        t_optimal_results = [o[0]['runs'] for o in optimal_results]
        t_optimal_results = chain.from_iterable(t_optimal_results)
        t_optimal_results = np.asarray(list(t_optimal_results)) / dcf_results * 100
        f_optimal_results = [o[1]['runs'] for o in optimal_results]
        f_optimal_results = chain.from_iterable(f_optimal_results)
        f_optimal_results = np.asarray(list(f_optimal_results)) / dcf_results * 100

    df = pd.DataFrame(
        np.stack([t_optimal_results, f_optimal_results, mab_h_results, mab_f_results, sr_results], axis=1),
        columns=['T-Optimal', 'F-Optimal', 'H-MAB', 'MAB', 'SR']
    )

    plt.axvline(100, color='k', linestyle='--', linewidth=0.5)
    sns.boxplot(
        data=pd.melt(df),
        y='variable', x='value',
        color='#305080',
        width=0.6,
        boxprops=dict(linewidth=0.),
        whiskerprops=dict(linewidth=0.5),
        medianprops=dict(linewidth=0.5, color='w'),
        capprops=dict(linewidth=0.5),
        flierprops=dict(marker='o', markersize=1, markeredgecolor='#305080'),
    )

    plt.ylabel('')
    plt.xlabel(r'Improvement over DCF [\%]')
    plt.xlim(0, 400)

    plt.grid(axis='x', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('random_boxplot.pdf', bbox_inches='tight')
    plt.show()
