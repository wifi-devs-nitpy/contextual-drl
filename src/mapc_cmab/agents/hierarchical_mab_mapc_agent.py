from collections import defaultdict
from itertools import chain
from typing import Callable
from copy import copy

import numpy as np
from chex import Array, Shape, Scalar
from reinforced_lib import RLib

from mapc_cmab.agents.mapc_agent import MapcAgent


class HierarchicalMABMapcAgent(MapcAgent):
    """
    The hierarchical MAB agent responsible for the selection of the AP and station pairs.
    The agent consists of three phases:

      1. Selection of the group of APs which are sharing the channel.
      2. Selection of the stations which are served simultaneously by the APs in the group.
      3. selection of the links used by a station
      4. Selection of the transmission power for each ap-sta link in the group.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    find_groups_agent : RLib
        The agent which selects the group of APs sharing the channel.
    find_groups_dict : dict[int, int]
        The dictionary which maps the station to the agent index.
    assign_stations_agents : dict[int, RLib]
        The agents which select the stations served by the APs.
    assign_stations_dict : dict[tuple[tuple, int], tuple[int, int]]
        The dictionary which maps the station and the group of APs to the agent index.
    select_tx_power_agent : dict[int, RLib]
        The agent which selects the transmission power for a particular link.
    select_tx_power_dict : dict[tuple[tuple, int, int], int]
        The dictionary which maps the group of APs, served station and link to the agent index.
    ap_group_action_to_ap_group : Callable
        The function which translates the action of the agent to the tuple of APs sharing the channel.
    sta_group_action_to_sta_group : Callable
        The function which translates the action of the agent to the list of served stations.
    link_action_to_links_group: Callable
        The function which translate the link action of the agent to the list of links that the station uses
    tx_matrix_shape : Shape
        The shape of the transmission matrix.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            find_groups_agent: RLib,
            find_groups_dict: dict[int, int],
            assign_stations_agents: dict[int, RLib],
            assign_stations_dict: dict[tuple[tuple, int], tuple[int, int]],

            assign_links_agent: RLib,
            assign_links_dict: dict[tuple[tuple, int, int], int],

            select_tx_power_agent: dict[int, RLib],
            select_tx_power_dict: dict[tuple[tuple, int, int], int],

            link_action_to_links_group: Callable,
            ap_group_action_to_ap_group: Callable,
            sta_group_action_to_sta_group: Callable,
            tx_matrix_shape: Shape,
            n_tx_power_levels: int,
            logger=None
    ) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))
        self.n_nodes = len(self.access_points) + len(list(chain.from_iterable(associations.values())))

        self.find_groups_agent = find_groups_agent
        self.find_groups_dict = find_groups_dict
        self.find_groups_last_step = defaultdict(int)
        self.assign_stations_agents = assign_stations_agents
        self.assign_stations_dict = assign_stations_dict
        self.assign_stations_last_step = defaultdict(int)

        self.assign_links_agent = assign_links_agent 
        self.assign_links_dict = assign_links_dict
        self.assign_links_last_step = defaultdict(int)
        
        self.select_tx_power_agent = select_tx_power_agent
        self.select_tx_power_dict = select_tx_power_dict
        self.select_tx_power_last_step = defaultdict(lambda: defaultdict(int))

        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.sta_group_action_to_sta_group = sta_group_action_to_sta_group
        self.link_action_to_links_group = link_action_to_links_group
        self.tx_matrix_shape = tx_matrix_shape
        self.n_tx_power_levels = n_tx_power_levels
        self.logger = logger 

        self.step = 0
        self.rewards = []

    def sample(self, reward: Scalar) -> tuple[Array, Array]:
        """
        Samples the agent to get the transmission matrix.

        Returns
        -------
        tuple
            The transmission matrix and the tx power vector.
        """

        self.step += 1
        self.rewards.append(reward)

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points).item()
        designated_station = np.random.choice(self.associations[sharing_ap]).item()

        # sharing_ap = 0
        # designated_station = 6


        # Sample the agent that finds groups of APs
        reward = self.rewards[self.find_groups_last_step[designated_station]]
        self.find_groups_last_step[designated_station] = self.step
        idx = self.find_groups_dict[designated_station]
        ap_group_action = self.find_groups_agent.sample(reward, agent_id=idx).item()

        ap_group = self.ap_group_action_to_ap_group(ap_group_action, sharing_ap)
        all_aps = tuple(sorted(ap_group + (sharing_ap,)))

        # Sample the agents which assign stations to APs
        sta_group_action = {}

        for ap in ap_group:
            reward = self.rewards[self.assign_stations_last_step[all_aps, ap]]
            self.assign_stations_last_step[all_aps, ap] = self.step
            n, idx = self.assign_stations_dict[all_aps, ap]
            sta_group_action[ap] = self.assign_stations_agents[n].sample(reward, agent_id=idx).item()

        sta_group = self.sta_group_action_to_sta_group(sta_group_action)
        all_aps_sort = np.argsort(np.asarray(tuple(ap_group) + (sharing_ap,)))
        all_stas = tuple(np.asarray(tuple(sta_group) + (designated_station,))[all_aps_sort].tolist())

        link_ap_sta = {
            0: {
                "ap_sta_pairs": [], 
                "tx_power_indices": np.zeros(self.n_nodes, dtype=np.int32), 
                "tx_matrix": np.zeros((self.n_nodes, self.n_nodes))
            }, 
            1: {
                "ap_sta_pairs": [], 
                "tx_power_indices": np.zeros(self.n_nodes, dtype=np.int32), 
                "tx_matrix": np.zeros((self.n_nodes, self.n_nodes))
            }, 
            2: {
                "ap_sta_pairs": [], 
                "tx_power_indices": np.zeros(self.n_nodes, dtype=np.int32), 
                "tx_matrix": np.zeros((self.n_nodes, self.n_nodes))
            }
        }
        # the above dictionary will be modified later, by not violating the DRY rule

        # sampling the links
        for i, (ap, sta) in enumerate(zip(all_aps, all_stas)):
            reward = self.rewards[self.assign_links_last_step[all_aps, sta]]
            self.assign_links_last_step[all_aps, sta] = self.step
            idx = self.assign_links_dict[all_aps, sta]
            link_action = self.assign_links_agent.sample(reward, agent_id=idx).item()

            #decode logic for the assign_links_agent_action_to_links
            links = self.link_action_to_links_group(link_action)

            # links is a tuple for that particular station 
            for link in links: 
                link_ap_sta[link]["ap_sta_pairs"].append((ap, sta))
                

        # # Sample the agent which assigns tx power
        # tx_power = np.zeros(self.n_nodes, dtype=np.int32)

        # for i, (ap, sta) in enumerate(zip(all_aps, all_stas)):
        #     reward = self.rewards[self.select_tx_power_last_step[all_aps, sta, link]]
        #     self.select_tx_power_last_step[all_aps, sta, link] = self.step
        #     idx = self.select_tx_power_dict[all_aps, sta]
        #     tx_power[ap] = self.select_tx_power_agent.sample(reward, agent_id=idx).item()

        for link in link_ap_sta.keys():
            for (ap, sta) in link_ap_sta[link]["ap_sta_pairs"]:
                reward_idx = self.select_tx_power_last_step[link][all_aps, sta, link]
                self.select_tx_power_last_step[link][all_aps, sta, link] = self.step
                reward = self.rewards[reward_idx]
                agent_idx = self.select_tx_power_dict[all_aps, sta, link]
                tx_power = self.select_tx_power_agent[link].sample(reward, agent_id=agent_idx).item()
                link_ap_sta[link]["tx_power_indices"][ap] = tx_power
                link_ap_sta[link]["tx_matrix"][ap, sta] = 1

        # # Create the transmission matrix based on the sampled pairs
        # tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        # tx_matrix[sharing_ap, designated_station] = 1
   

        # for ap, sta in zip(ap_group, sta_group):
        #     tx_matrix[ap, sta] = 1

        # return tx_matrix, tx_power

        tx_matrices = np.array(list(link_ap_sta[r]["tx_matrix"] for r in range(0, 3)), dtype=np.int16)
        tx_power_indices = np.array(list(link_ap_sta[r]["tx_power_indices"] for r in range(0, 3)), dtype=np.int16)
        # tx_power_indices = np.ones(shape=(3, self.n_nodes), dtype=np.int32) * 7

        if self.logger is not None: 
            # print(f"tx_matrix: {np.where(tx_matrices == 1)}")
            # print(f"tx_power_indices: {tx_power_indices}")       
            self.logger.log(
                step=self.step,
                tx_matrices=tx_matrices,
                tx_power_indices=tx_power_indices,
                reward=self.rewards[-1], # reward is 1 step delayed. 
            )
            return tx_matrices, tx_power_indices
        
        return (tx_matrices, tx_power_indices)
