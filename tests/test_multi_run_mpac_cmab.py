# %%
from mapc_cmab.agents.hierarchical_dqn  import HierarchicalMapcDQNAgent 
from mapc_cmab.agents.mapc_cmab_agent_factory import MapcDQNAgentFactory
from mapc_cmab.envs.scenario_impl import small_office_scenario 
import jax 
import jax.numpy as jnp 
from tqdm import tqdm 


# %%
d_ap = 10
d_sta = 2
scenario = small_office_scenario(d_ap=d_ap, d_sta=d_sta)

# %%
print(scenario.__dict__)
scenario.plot_rs()

# %%


# %%
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

# Set a local directory for the persistent compilation cache (FIXED NAME)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")

# Enable all extra XLA caching features ("all")
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

agent_factory = MapcDQNAgentFactory(
    associations=scenario.associations,
    agent_params_lvl1=None, 
    agent_params_lvl2=None, 
    agent_params_lvl3=None, 
    agent_params_lvl4=None, 
    n_tx_power_levels=4, 
    n_links=3, 
    seed=42
)


def run_agent_thread(
    run_number: int,
    key: jax.Array,
    n_steps: int,
):

    # Independent agent
    agent = (
        agent_factory.create_hierarchical_DQN_cmapc_agent()
    )

    throughputs = np.zeros(
        n_steps,
        dtype=np.float32
    )

    previous_throughput = 0.0

    for step in tqdm(range(1, n_steps), desc=f"run number: {run_number}"):

        key, run_key = jax.random.split(key)

        link_ap_sta = agent.sample(
            reward=previous_throughput
        )

        data_rate, reward = scenario(
            run_key,
            link_ap_sta,
        )

        data_rate = float(data_rate)

        throughputs[step] = data_rate
        previous_throughput = data_rate

    return run_number, throughputs

def compute_multiple_runs_threaded(
    n_runs: int,
    key: jax.Array,
    n_steps: int,
    max_workers: int | None = None,
):

    if max_workers is None:
        max_workers = min(
            n_runs,
            os.cpu_count() or 1
        )

    run_keys = jax.random.split(
        key,
        n_runs
    )

    results = [None] * n_runs

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                run_agent_thread,
                run_number,
                run_keys[run_number],
                n_steps,
            )
            for run_number in range(n_runs)
        ]

        for future in tqdm(
            as_completed(futures),
            total=n_runs,
            desc="Running repetitions",
        ):

            run_number, result = future.result()

            results[run_number] = result

    return jnp.asarray(
        np.stack(results)
    )

# ============================================================
# RUN
# ============================================================

total_steps = 5000
n_reps = 10

key = jax.random.PRNGKey(42)

throughputs = compute_multiple_runs_threaded(
    n_runs=n_reps,
    key=key,
    n_steps=5000,
    max_workers=10,
)

# %%
import matplotlib.pyplot as plt
import numpy as np 

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from pathlib import Path


def analyze_and_plot_throughputs(
    throughputs,
    window_size: int = 100,
    step_duration: float = 0.005,
    confidence: float = 0.99,
    title: str = "Average Throughput Over Time",
    label: str = "Average throughput",
    output_dir: str = "./results",
    filename: str = "throughput_results",
    save: bool = True,
    show: bool = True,
):
    """
    Analyze throughput measurements from multiple independent runs,
    compute windowed averages and Student-t confidence intervals,
    plot the results, and optionally save them.

    Parameters
    ----------
    throughputs : array-like
        Throughput measurements with shape:

            (n_runs, total_steps)

        Each row represents one independent experiment/run.

    window_size : int, default=100
        Number of consecutive simulation steps to average into
        one plotted measurement.

    step_duration : float, default=0.005
        Duration represented by one simulation step, in seconds.

    confidence : float, default=0.99
        Confidence level for the Student-t confidence interval.

    title : str
        Plot title.

    label : str
        Label for the mean throughput curve.

    output_dir : str
        Directory where results will be saved.

    filename : str
        Base filename without extension.

    save : bool, default=True
        Whether to save the figure and processed statistics.

    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    results : dict
        Dictionary containing:
            - windowed_throughputs
            - time
            - mean
            - std
            - standard_error
            - ci_low
            - ci_high
    """

    # =========================================================
    # Convert to NumPy
    # =========================================================
    throughputs = np.asarray(throughputs)

    # =========================================================
    # Validate input
    # =========================================================
    if throughputs.ndim != 2:
        raise ValueError(
            "throughputs must have shape "
            "(n_runs, total_steps)."
        )

    n_runs, total_steps = throughputs.shape

    if n_runs < 2:
        raise ValueError(
            "At least 2 independent runs are required "
            "to compute a confidence interval."
        )

    if total_steps % window_size != 0:
        raise ValueError(
            f"total_steps ({total_steps}) must be exactly "
            f"divisible by window_size ({window_size}).\n"
            "No data will be silently sliced or discarded."
        )

    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    # =========================================================
    # Windowed averaging
    #
    # (n_runs, total_steps)
    #
    #        ↓ reshape
    #
    # (n_runs, n_windows, window_size)
    #
    #        ↓ mean
    #
    # (n_runs, n_windows)
    # =========================================================
    n_windows = total_steps // window_size

    windowed_throughputs = (
        throughputs
        .reshape(
            n_runs,
            n_windows,
            window_size,
        )
        .mean(axis=-1)
    )

    # =========================================================
    # Statistics across independent runs
    # =========================================================
    mean_ts = windowed_throughputs.mean(axis=0)

    # Sample standard deviation
    std_ts = windowed_throughputs.std(
        axis=0,
        ddof=1,
    )

    standard_error = (
        std_ts /
        np.sqrt(n_runs)
    )

    # =========================================================
    # Student-t confidence interval
    # =========================================================
    alpha = 1 - confidence

    critical_value = t.ppf(
        1 - alpha / 2,
        df=n_runs - 1,
    )

    margin_of_error = (
        critical_value *
        standard_error
    )

    ci_low = (
        mean_ts -
        margin_of_error
    )

    ci_high = (
        mean_ts +
        margin_of_error
    )

    # =========================================================
    # Time axis
    # =========================================================
    window_duration = (
        window_size *
        step_duration
    )

    time = (
        np.arange(n_windows + 1) *
        window_duration
    )

    # Add initial zero to match your plotting convention
    mean_plot = np.concatenate(
        [[0.0], mean_ts]
    )

    ci_low_plot = np.concatenate(
        [[0.0], ci_low]
    )

    ci_high_plot = np.concatenate(
        [[0.0], ci_high]
    )

    # =========================================================
    # Create figure
    # =========================================================
    fig, ax = plt.subplots(
        figsize=(11, 7),
        dpi=600,
    )

    # Mean throughput
    ax.plot(
        time,
        mean_plot,
        marker="o",
        markersize=4.5,
        linewidth=2.4,
        label=label,
    )

    # Confidence interval
    ax.fill_between(
        time,
        ci_low_plot,
        ci_high_plot,
        alpha=0.16,
        label=f"{int(confidence * 100)}% CI",
        interpolate=True,
    )

    # =========================================================
    # Labels
    # =========================================================
    ax.set_title(
        title,
        fontsize=19,
        fontweight="normal",
        pad=16,
    )

    ax.set_xlabel(
        "Time (s)",
        fontsize=18,
        fontweight="normal",
        labelpad=12,
    )

    ax.set_ylabel(
        "Throughput [Mb/s]",
        fontsize=18,
        fontweight="normal",
        labelpad=12,
    )

    # =========================================================
    # Tick formatting
    # =========================================================
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        width=1.3,
        length=6,
        colors="black",
        pad=7,
    )

    # =========================================================
    # Grid
    # =========================================================
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.30,
    )

    # =========================================================
    # Spines
    # =========================================================
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    # =========================================================
    # Legend
    # =========================================================
    ax.legend(
        fontsize=13,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
        loc="best",
    )

    fig.tight_layout()

    # =========================================================
    # Save results
    # =========================================================
    if save:

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save figure
        figure_path = (
            output_path /
            f"{filename}.png"
        )

        fig.savefig(
            figure_path,
            dpi=600,
            bbox_inches="tight",
        )

        # Save processed numerical results
        data_path = (
            output_path /
            f"{filename}.npz"
        )

        np.savez(
            data_path,
            raw_throughputs=throughputs,
            windowed_throughputs=windowed_throughputs,
            time=time,
            mean=mean_plot,
            std=np.concatenate([[0.0], std_ts]),
            standard_error=np.concatenate(
                [[0.0], standard_error]
            ),
            ci_low=ci_low_plot,
            ci_high=ci_high_plot,
            confidence=confidence,
            critical_value=critical_value,
            n_runs=n_runs,
            total_steps=total_steps,
            window_size=window_size,
            step_duration=step_duration,
        )

        print(f"Figure saved to:\n{figure_path}")
        print(f"Statistics saved to:\n{data_path}")

    # =========================================================
    # Show plot
    # =========================================================
    if show:
        plt.show()

    else:
        plt.close(fig)

    # =========================================================
    # Return everything useful
    # =========================================================
    return {
        "windowed_throughputs": windowed_throughputs,
        "time": time,
        "mean": mean_plot,
        "std": np.concatenate([[0.0], std_ts]),
        "standard_error": np.concatenate(
            [[0.0], standard_error]
        ),
        "ci_low": ci_low_plot,
        "ci_high": ci_high_plot,
        "confidence": confidence,
        "critical_value": critical_value,
        "n_runs": n_runs,
    }

# %%
analyze_and_plot_throughputs(throughputs, filename="./d_10_with_ss_parallel_runs")


