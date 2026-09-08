"""Hyperparameterized MAPC hierarchical DQN agent factory.

This is a drop-in replacement for the factory in the experiment setup.  The
main difference is that optimizer/replay/epsilon/network parameters are passed
through a single config dictionary rather than being hard-coded in each level.

IMPORTANT: discount is deliberately hard-coded to 0.0.  The tuning code never
modifies it, per the experiment requirement.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import chain, combinations
from typing import Iterable

import numpy as np
import optax
from chex import Array
from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN

from mapc_cmab.agents.hierarchical_dqn import HierarchicalMapcDQNAgent
from mapc_cmab.agents.mapc_agent import MapcAgent
from q_network_tuned import (
    QNetwork_lv1,
    QNetwork_lv2,
    QNetwork_lv3,
    QNetwork_lv4,
)


DEFAULT_AGENT_CONFIG = {
    "learning_rate": 5e-4,
    "experience_replay_buffer_size": 3000,
    "experience_replay_batch_size": 64,
    "experience_replay_steps": 1,
    "epsilon_min": 0.05,
    "epsilon_decay": {
        "find_groups": 0.999,
        "assign_stations": 0.999,
        "assign_links": 0.999,
        "tx_power": 0.999,
    },
    "hidden_dims": (64, 64),
    "use_layer_norm": False,
}


class MapcDQNAgentFactory:
    """Create a hierarchical contextual-DQN MAPC agent from a config."""

    def __init__(
        self,
        associations: dict[int, list[int]],
        agent_params_lvl1: dict | None = None,
        agent_params_lvl2: dict | None = None,
        agent_params_lvl3: dict | None = None,
        agent_params_lvl4: dict | None = None,
        n_links: int = 3,
        n_tx_power_levels: int = 4,
        seed: int = 42,
        logger=None,
        config: dict | None = None,
    ):
        self.associations = {ap: np.asarray(stations) for ap, stations in associations.items()}
        self.agent_params_lvl1 = agent_params_lvl1
        self.agent_params_lvl2 = agent_params_lvl2
        self.agent_params_lvl3 = agent_params_lvl3
        self.agent_params_lvl4 = agent_params_lvl4
        self.n_tx_power_levels = n_tx_power_levels
        self.n_links = n_links
        self.seed = seed
        self.logger = logger
        self.config = self._merge_config(config or {})

        np.random.seed(self.seed)

        self.inv_associations = {
            sta: ap for ap in associations.keys() for sta in associations[ap]
        }
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.n_nodes = self.n_ap + self.n_sta
        self.stations_per_ap = len(self.associations[0])
        self.ap_to_idx = {ap: i for i, ap in enumerate(self.access_points)}
        self.link_comb_index_to_links = {
            idx: list(link_comb)
            for idx, link_comb in enumerate(self._powerset_without_emptyset(range(self.n_links)))
        }
        self.sta_index_mapping = {
            sta: index for index, sta in enumerate(self.stations)
        }

    @staticmethod
    def _merge_config(config: dict) -> dict:
        merged = {
            **DEFAULT_AGENT_CONFIG,
            **{k: v for k, v in config.items() if k != "epsilon_decay"},
        }
        merged["epsilon_decay"] = {
            **DEFAULT_AGENT_CONFIG["epsilon_decay"],
            **config.get("epsilon_decay", {}),
        }
        merged["hidden_dims"] = tuple(merged["hidden_dims"])
        return merged

    def _make_agent(self, *, q_network, obs_shape, action_size, epsilon_decay, params_override=None):
        p = {
            "q_network": q_network,
            "obs_space_shape": obs_shape,
            "act_space_size": action_size,
            "optimizer": optax.adam(self.config["learning_rate"]),
            "experience_replay_buffer_size": self.config["experience_replay_buffer_size"],
            "experience_replay_batch_size": self.config["experience_replay_batch_size"],
            "experience_replay_steps": self.config["experience_replay_steps"],
            # Intentionally NOT tunable.
            "discount": 0.0,
            "epsilon": 1.0,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": self.config["epsilon_min"],
        }
        if params_override:
            p.update(params_override)
            # Prevent an accidental override from changing the controlled variable.
            p["discount"] = 0.0
        return RLib(agent_type=DQN, agent_params=p, no_ext_mode=True)

    def _network(self, cls, n_actions: int):
        return cls(
            n_actions=n_actions,
            hidden_dims=self.config["hidden_dims"],
            use_layer_norm=self.config["use_layer_norm"],
        )

    def create_hierarchical_DQN_cmapc_agent(self, logger) -> MapcAgent:
        self.seed += 1
        np.random.seed(self.seed)

        decay = self.config["epsilon_decay"]

        action_size_lvl1 = 2 ** (self.n_ap - 1)
        find_groups_agent = self._make_agent(
            q_network=self._network(QNetwork_lv2, action_size_lvl1),
            obs_shape=(self.n_ap + self.stations_per_ap,),
            action_size=action_size_lvl1,
            epsilon_decay=decay["find_groups"],
            params_override=self.agent_params_lvl2,
        )

        assign_stations_agent = {
            ap: self._make_agent(
                q_network=self._network(QNetwork_lv1, len(self.associations[ap])),
                obs_shape=(self.n_ap,),
                action_size=len(self.associations[ap]),
                epsilon_decay=decay["assign_stations"],
                params_override=self.agent_params_lvl1,
            )
            for ap in self.access_points
        }

        action_size_lvl3 = 2 ** self.n_links - 1
        assign_links_agent = {
            ap: self._make_agent(
                q_network=self._network(QNetwork_lv3, action_size_lvl3),
                obs_shape=(self.n_ap * self.stations_per_ap,),
                action_size=action_size_lvl3,
                epsilon_decay=decay["assign_links"],
                params_override=self.agent_params_lvl3,
            )
            for ap in self.access_points
        }

        grids = np.meshgrid(self.stations, list(range(self.n_links)), indexing="ij")
        sta_link = np.stack([grid.ravel() for grid in grids], axis=-1)

        assign_tx_power_agent = {
            (int(sta), int(link)): self._make_agent(
                q_network=self._network(QNetwork_lv4, self.n_tx_power_levels),
                obs_shape=(self.n_sta * self.n_links,),
                action_size=self.n_tx_power_levels,
                epsilon_decay=decay["tx_power"],
                params_override=self.agent_params_lvl4,
            )
            for sta, link in sta_link
        }

        return HierarchicalMapcDQNAgent(
            associations=self.associations,
            find_groups_agent=find_groups_agent,
            assign_stations_agent=assign_stations_agent,
            assign_links_agent=assign_links_agent,
            assign_tx_power_agent=assign_tx_power_agent,
            encode_sharing_ap=self._encode_sharing_ap,
            encode_ap_group=self._encode_ap_group,
            encode_ap_stations_to_tx_vector=self._encode_ap_stations_to_tx_vector,
            encode_sta_links_vector=self._encode_sta_links_vector,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            link_comb_index_to_links=self.link_comb_index_to_links,
            sta_index_mapping=self.sta_index_mapping,
            n_links=self.n_links,
            n_tx_power_levels=self.n_tx_power_levels,
            logger=logger,
        )

    @staticmethod
    def _powerset(iterable: Iterable) -> Iterable:
        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    @staticmethod
    def _powerset_without_emptyset(iterable: Iterable) -> Iterable:
        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> tuple[int]:
        ap_set = set(self.access_points).difference({sharing_ap})
        return tuple(self._powerset(ap_set))[ap_group_action]

    def _sta_group_action_to_sta_group(self, sta_group_action: dict[int, int]) -> list[int]:
        return [self.associations[ap][sta_id] for ap, sta_id in sta_group_action.items()]

    def _encode_ap_group(self, sharing_ap, selected_ap_group) -> Array:
        ap_group = np.asarray(list(selected_ap_group) + [sharing_ap])
        return np.isin(np.asarray(self.access_points), ap_group).astype(np.int32)

    def _encode_ap_stations_to_tx_vector(self, ap_sta_dict):
        res = np.zeros(shape=(self.n_ap, self.stations_per_ap))
        rows = [self.ap_to_idx[ap] for ap in ap_sta_dict.keys()]
        cols = list(ap_sta_dict.values())
        res[rows, cols] = 1
        return res.reshape(-1)

    def _encode_sta_links_vector(self, sta_links: dict[int, int]):
        res = np.zeros((self.n_sta, self.n_links))
        rows = [
            self.sta_index_mapping[sta]
            for sta, idx in sta_links.items()
            for _ in self.link_comb_index_to_links[idx]
        ]
        cols = [
            link
            for idx in sta_links.values()
            for link in self.link_comb_index_to_links[idx]
        ]
        res[rows, cols] = 1
        return res.reshape(-1)

    def _encode_sharing_ap(self, sharing_ap, sharing_station) -> Array:
        arr1 = np.isin(
            np.asarray(self.access_points), np.asarray(sharing_ap)
        ).astype(np.int32)
        arr2 = np.zeros(self.stations_per_ap)
        rel_index = (self.associations[sharing_ap] == sharing_station).argmax()
        arr2[rel_index] = 1
        return np.concatenate([arr1, arr2], axis=0)
