import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Generic NPZ Comparison Plotter
# ============================================================

def plot_npz_comparison(
    data,
    mean_key="mean",
    ci_low_key="ci_low",
    ci_high_key="ci_high",
    step_duration=0.5,
    title="Average Throughput Comparison",
    xlabel="Time (s)",
    ylabel="Throughput (Mbps)",
    figsize=(11, 7),
    dpi=600,
    output_dir="./presentation_results",
    filename="comparison.png",
):
    """
    Plot multiple NPZ datasets on the same graph.

    Parameters
    ----------
    data : list
        List of [npz_object, label].

        Example:
            [
                [hmab_static, "ML-CSR-HMAB"],
                [hdqn_static, "ML-CSR-HDQN"],
            ]

    mean_key : str
        Key containing the mean values.

    ci_low_key : str
        Key containing the lower confidence interval.

    ci_high_key : str
        Key containing the upper confidence interval.

    step_duration : float
        Duration represented by one sample.

    title : str
        Plot title.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    figsize : tuple
        Figure size.

    dpi : int
        Figure resolution.

    output_dir : str
        Directory where the figure will be saved.

    filename : str
        Output filename.
    """

    # ========================================================
    # Create figure
    # ========================================================

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    # ========================================================
    # Plot every dataset
    # ========================================================

    for arr, label in data:

        # ----------------------------------------------------
        # Extract arrays from NPZ
        # ----------------------------------------------------

        mean = np.asarray(arr[mean_key])
        ci_low = np.asarray(arr[ci_low_key])
        ci_high = np.asarray(arr[ci_high_key])

        # ----------------------------------------------------
        # Make sure all arrays have the same length
        # ----------------------------------------------------

        if not (
            len(mean)
            == len(ci_low)
            == len(ci_high)
        ):
            raise ValueError(
                f"Length mismatch for '{label}': "
                f"mean={len(mean)}, "
                f"ci_low={len(ci_low)}, "
                f"ci_high={len(ci_high)}"
            )

        # ----------------------------------------------------
        # Generate time axis
        # ----------------------------------------------------

        time = np.arange(len(mean)) * step_duration

        # ----------------------------------------------------
        # Mean line
        # ----------------------------------------------------

        ax.plot(
            time,
            mean,
            marker="o",
            markersize=4.5,
            linewidth=2.4,
            label=label,
        )

        # ----------------------------------------------------
        # Confidence interval
        # ----------------------------------------------------

        ax.fill_between(
            time,
            ci_low,
            ci_high,
            alpha=0.16,
        )

    # ========================================================
    # Title
    # ========================================================

    ax.set_title(
        title,
        fontsize=19,
        fontweight="normal",
        pad=16,
    )

    # ========================================================
    # Axis labels
    # ========================================================

    ax.set_xlabel(
        xlabel,
        fontsize=18,
        fontweight="normal",
        labelpad=12,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=18,
        fontweight="normal",
        labelpad=12,
    )

    # ========================================================
    # Tick formatting
    # ========================================================

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        width=1.3,
        length=6,
        colors="black",
        pad=7,
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.8,
        alpha=0.30,
    )

    # ========================================================
    # Spines
    # ========================================================

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        fontsize=13,
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="black",
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout()

    # ========================================================
    # Save figure
    # ========================================================

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path = output_dir / filename

    fig.savefig(
        figure_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    print(f"Figure saved to:\n{figure_path}")

    # ========================================================
    # Display
    # ========================================================

    plt.show()

