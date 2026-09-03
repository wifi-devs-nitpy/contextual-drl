import string
from abc import ABC, abstractmethod
from itertools import chain
from itertools import product
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from chex import Array, Scalar
from mapc_cmab.mapc_sim_mlo.utils import default_path_loss

from mapc_cmab.plots.config import get_cmap


class Scenario(ABC):
    """
    Base class for scenarios.

    Parameters
    ----------
    associations: Dict
        Dictionary of associations between access points and stations.
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls: Optional[Array]
        Matrix counting the walls between each pair of nodes.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to the X and Y coordinates of
        the wall start and end points.
    channel_width: int
        Channel width in MHz.
    path_loss_fn: Callable
        A function that calculates the path loss between two nodes. The function signature should be
        `path_loss_fn(distance: Array, walls: Array) -> Array`, where `distance` is the matrix of distances
        between nodes and `walls` is the adjacency matrix of walls. By default, the simulator uses the
        residential TGax path loss model.
    str_repr: str
        String representation of the scenario.
    """

    def __init__(
            self,
            associations: Dict,
            pos: Array,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            channel_width: int = None,
            path_loss_fn: Callable = default_path_loss,
            str_repr: str = ""
    ) -> None:
        self.CCA_THRESHOLD = -82.0  # IEEE Std 802.11-2020 (Revision of IEEE Std 802.11-2016), 17.3.10.6: CCA requirements
        self.DEFAULT_CHANNEL_WIDTH = 20

        self.associations = associations
        self.pos = pos
        self.walls_pos = walls_pos if walls_pos is not None else []
        self.walls = walls if walls is not None else self._calculate_walls_matrix()
        self.channel_width = channel_width if channel_width is not None else self.DEFAULT_CHANNEL_WIDTH
        self.path_loss_fn = path_loss_fn

        self.str_repr = str_repr

    def __str__(self):
        return self.str_repr

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[Scalar, Scalar]:
        """
        Calculates the throughput and reward for the given transmission matrix.

        Returns
        -------
        tuple[Scalar, Scalar]
            Throughput and reward.
        """
        pass

    @abstractmethod
    def split_scenario(self) -> list[tuple['Scenario', float]]:
        """
        Splits the scenario into the sequence of static scenarios and the duration.

        Returns
        -------
        list[tuple[Scenario, float]]
            List of tuples, where each tuple contains the static scenario and the duration.
        """
        pass

    def reset(self) -> None:
        pass

    def _calculate_walls_matrix(self) -> Array:
        """
        Converts a list of wall positions to a matrix counting the walls between each pair of nodes.

        Returns
        -------
        Array
            Matrix counting the walls between each pair of nodes.
        """

        def det(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

        def intersect(a, b, c, d):
            return det(a, b, c) * det(a, b, d) < 0 and det(a, c, d) * det(b, c, d) < 0

        nodes = list(self.associations.keys()) + list(chain.from_iterable(self.associations.values()))
        walls = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

        for wall in self.walls_pos:
            for node_a, node_b in product(nodes, nodes):
                if node_a > node_b and intersect(self.pos[node_a], self.pos[node_b], wall[:2], wall[2:]):
                    walls[node_a, node_b] += 1
                    walls[node_b, node_a] += 1

        return walls

    def get_associations(self) -> Dict:
        return self.associations

    @staticmethod
    def _generate_letter_sequence(n):
        seq = []
        length = 1

        while len(seq) < n:
            for combination in product(string.ascii_uppercase, repeat=length):
                seq.append("".join(combination))

                if len(seq) == n:
                    return seq

            length += 1

        return seq

    def plot(self, pos: Array, filename: str = None, label_size: int = 10, show_circles: bool = True) -> None:
        """
        Plots the current state of the scenario.

        Parameters
        ----------
        pos : Array
            Two dimensional array of node positions.
        filename : str
            Filename to save the plot.
        label_size : int
            Size of the labels.
        show_circles : bool
            If True, circles are drawn around the access points indicating the position of the furthest station.
        """

        colors = get_cmap(len(self.associations))
        ap_labels = self._generate_letter_sequence(len(self.associations))

        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(self.associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i])
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap_labels[i]}', (pos[ap, 0], pos[ap, 1] + 2), color=colors[i], va='bottom', ha='center', size=label_size)

            if show_circles:
                radius = np.max(np.sqrt(np.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
                circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.2, fill=False, linewidth=0.5)
                ax.add_patch(circle)

        # Plot walls
        for wall in self.walls_pos:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='black', linewidth=1)

        ax.set_axisbelow(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.set_title('Location of nodes')
        ax.grid()

        if filename:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    def is_cca_single_tx(self, pos: Array, tx_power: Array) -> bool:
        """
        Checks if the scenario is a CSMA single transmission scenario, i.e., if there is only one transmission
        possible at a time due to the CCA threshold. **Note**: This function assumes that the scenario
        contains downlink transmissions only.

        Parameters
        ----------
        pos : Array
            Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
        tx_power : Array
            Transmission power of the nodes. Each entry corresponds to a node.

        Returns
        -------
        bool
            True if the scenario is a CSMA single transmission scenario, False otherwise.
        """

        ap_ids = np.array(list(self.associations.keys()))

        ap_pos = pos[ap_ids]
        ap_tx_power = tx_power[ap_ids]
        ap_walls = self.walls[ap_ids][:, ap_ids]

        distance = np.sqrt(np.sum((ap_pos[:, None, :] - ap_pos[None, ...]) ** 2, axis=-1))
        signal_power = ap_tx_power - self.path_loss_fn(distance, ap_walls)
        signal_power = np.where(np.isnan(signal_power), np.inf, signal_power)

        return np.all(signal_power > self.CCA_THRESHOLD)

    def tx_to_action(self, tx_matrix: Array, tx_power: Array, mcs: Array = None) -> dict:
        """
        Converts a transmission matrix to a list of transmissions. Assumes downlink.

        Parameters
        ----------
        tx_matrix: Array
            Transmission matrix.
        tx_power: Array
            Transmission power of the nodes.

        Returns
        -------
        dict
            A dict, where each entry is a transmission from an access point to a station. The key is the access
            point and the value is the station and the transmission power.
        """

        aps = list(self.associations.keys())
        action = {}

        for ap in aps:
            assert np.sum(tx_matrix[ap, :]) <= 1, 'Multiple transmissions at AP'
            for sta in self.associations[ap]:
                if tx_matrix[ap, sta]:
                    action[ap] = (sta, tx_power[ap].item())

        return action



    def plot_rs(
        self,
        pos: Array,
        filename: str = None,
        label_size: int = 12,
        show_circles: bool = True,
        figsize: tuple = (14, 10),
        dpi: int = 200,
        save_dpi: int = 400,
    ) -> None:
        """
        Plot the current state of the scenario using a high-resolution,
        publication-quality visualization.

        Parameters
        ----------
        pos : Array
            Two-dimensional array of node positions with shape (N, 2).

        filename : str, optional
            Filename to save the plot. If None, the plot is displayed.

        label_size : int, default=12
            Font size used for AP labels.

        show_circles : bool, default=True
            If True, circles are drawn around each access point indicating
            the distance to its furthest associated station.

        figsize : tuple, default=(14, 10)
            Figure size in inches.

        dpi : int, default=200
            Resolution used while rendering the figure interactively.

        save_dpi : int, default=400
            Resolution used when saving raster images such as PNG.
        """

        # ------------------------------------------------------------
        # Create a large, high-resolution figure
        # ------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=figsize,
            dpi=dpi,
            constrained_layout=True,
        )

        colors = get_cmap(len(self.associations))
        ap_labels = self._generate_letter_sequence(len(self.associations))

        # ------------------------------------------------------------
        # Plot APs and their associated stations
        # ------------------------------------------------------------
        for i, (ap, stations) in enumerate(self.associations.items()):

            color = colors[i]

            stations = np.asarray(stations)

            # --------------------------------------------------------
            # Plot stations
            # --------------------------------------------------------
            if len(stations) > 0:

                ax.scatter(
                    pos[stations, 0],
                    pos[stations, 1],
                    s=45,                    # marker size
                    marker='o',
                    color=color,
                    alpha=0.75,
                    edgecolors='white',
                    linewidths=0.4,
                    label=f'AP {ap_labels[i]} Stations'
                )

            # --------------------------------------------------------
            # Plot Access Point
            # --------------------------------------------------------
            ax.scatter(
                pos[ap, 0],
                pos[ap, 1],
                s=220,
                marker='X',
                color=color,
                edgecolors='black',
                linewidths=0.8,
                zorder=5,
            )

            # --------------------------------------------------------
            # Annotate AP
            # --------------------------------------------------------
            ax.annotate(
                f'AP {ap_labels[i]}',
                xy=(pos[ap, 0], pos[ap, 1]),
                xytext=(0, 12),
                textcoords='offset points',
                color=color,
                fontsize=label_size,
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=6,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='white',
                    edgecolor=color,
                    alpha=0.85,
                    linewidth=0.7,
                ),
            )

            # --------------------------------------------------------
            # Coverage / association circle
            # --------------------------------------------------------
            if show_circles and len(stations) > 0:

                distances = np.linalg.norm(
                    pos[stations, :] - pos[ap, :],
                    axis=1
                )

                radius = np.max(distances)

                circle = plt.Circle(
                    (pos[ap, 0], pos[ap, 1]),
                    radius * 1.2,
                    fill=False,
                    edgecolor=color,
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.65,
                    zorder=1,
                )

                ax.add_patch(circle)

        # ------------------------------------------------------------
        # Plot walls
        # ------------------------------------------------------------
        for wall in self.walls_pos:

            ax.plot(
                [wall[0], wall[2]],
                [wall[1], wall[3]],
                color='black',
                linewidth=2.0,
                solid_capstyle='round',
                zorder=4,
            )

        # ------------------------------------------------------------
        # Styling
        # ------------------------------------------------------------
        ax.set_axisbelow(True)

        ax.set_xlabel(
            'X Position [m]',
            fontsize=13,
            fontweight='bold'
        )

        ax.set_ylabel(
            'Y Position [m]',
            fontsize=13,
            fontweight='bold'
        )

        ax.set_title(
            'Network Topology and Access Point Associations',
            fontsize=16,
            fontweight='bold',
            pad=15,
        )

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=11,
        )

        ax.grid(
            True,
            linestyle='--',
            linewidth=0.6,
            alpha=0.5,
        )

        ax.set_aspect('equal', adjustable='box')

        # Add some breathing room around the scenario
        x_min = np.min(pos[:, 0])
        x_max = np.max(pos[:, 0])

        y_min = np.min(pos[:, 1])
        y_max = np.max(pos[:, 1])

        x_margin = max((x_max - x_min) * 0.08, 1.0)
        y_margin = max((y_max - y_min) * 0.08, 1.0)

        ax.set_xlim(
            x_min - x_margin,
            x_max + x_margin
        )

        ax.set_ylim(
            y_min - y_margin,
            y_max + y_margin
        )

        # ------------------------------------------------------------
        # Save or display
        # ------------------------------------------------------------
        if filename:

            fig.savefig(
                filename,
                dpi=save_dpi,
                bbox_inches='tight',
                facecolor='white',
            )

            plt.close(fig)

        else:

            plt.show()
            plt.close(fig)