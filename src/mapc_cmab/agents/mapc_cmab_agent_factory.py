from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations, product
from typing import Iterable, Iterator

import numpy as np
from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN

from mapc_cmab.agents.mapc_agent import MapcAgent

from mapc_cmab.agents.q_network import QNetwork_lv1, QNetwork_lv2, QNetwork_lv3, QNetwork_lv4
import optax 
from chex import Array 

class MapcDQNAgentFactory:
    """

    This is a factory that creates the Contextual DQN agents with the given agent type and parameters"
    
    Parameters 
    ----------
    assocations: dict[int, list[int]]
        The dictionary of associations between APs and Stations 
    agent_type : DQN
        The type of the agent 
    agent_params_lvl1 : dict
        The dictionary containing the params for the level1
    agent_params_lvl2 : dict
        The dictionary containing the params for the level2
    agent_params_lvl3 : dict
        The dictionary containing the params for the level3
    agent_params_lvl4 : dict
        The dictionary containing the params for the level4
    n_links: int
        The number of radio links available
    n_tx_power_levels : int
        The number of Transmission power levels 
    seed: int

    """

    def __init__(
            self, 
            associations: dict[int, list[int]],
            agent_type: DQN, 
            agent_params_lvl1: dict,
            agent_params_lvl2: dict, 
            agent_params_lvl3: dict, 
            agent_params_lvl4: dict,
            n_links: int = 3,
            n_tx_power_levels: int = 4, 
            seed: int = 42, 
    ):
        self.associations = deepcopy(associations)
        self.agent_type = agent_type
        self.agent_params_lvl1 = agent_params_lvl1
        self.agent_params_lvl2 = agent_params_lvl2
        self.agent_params_lvl3 = agent_params_lvl3
        self.agent_params_lvl4 = agent_params_lvl4
        self.tx_power_levels = n_tx_power_levels
        self.n_links = n_links
        self.seed = seed


        np.random.seed()

        self.inv_associations = {sta: ap for ap in associations.keys() for sta in associations[ap]}
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.n_nodes = self.n_ap + self.n_sta
        self.stations_per_ap = len(self.associations[0]) # assuming that the number of stations of each AP are equal
        self.ap_to_idx = {ap: i for i, ap in enumerate(self.access_points)} # index of a AP in list of all APs
        self.link_comb_index_to_links = {
            idx: list(link_comb)

            for idx, link_comb in enumerate(self._powerset_without_emptyset(range(self.n_links)))
        }

    def create_hierarchical_DQN_cmapc_agent(self) -> MapcAgent: 
        """
        Intialises the Hierarchical DQN Agent 

        Returns
        -------
        HierarchialMapcAgent
            The hierarchial MAPC Agent        
        
        """

        action_size_lvl1 = 2**(self.n_ap - 1)
        find_groups_agent = RLib(
            agent_type=DQN,

            agent_params = {
                "q_network": QNetwork_lv2(n_actions=action_size_lvl1), 

                "obs_space_shape": (2, 4), # sharing AP, and its station in encoded format  
                "act_space_size": action_size_lvl1, 

                "optimizer": optax.adam(1e-3), 

                "experience_replay_buffer_size": 1000,
                "experience_replay_batch_size": 32,
                "experience_replay_steps": 1,

                #contextual Bandit type
                "discount": 0.0, 

                "epsilon": 1.0, 
                "epsilon_decay": 0.995,
                "epsilon_min": 0.05,
            },
            no_ext_mode=True,
        )

        # we donot need indexing here, as it is only one agent at the level -1 
        # level-2 assign_stations agents, Each AP has an agent that assigns the station to it. 
        # input is sharing AP in encoded format 
        action_size_lvl2 = len(self.associations[0]) # assumming that the no.of stations with each is equal.
        assign_stations_agent = {
            ap: RLib(
                agent_type=DQN,

                agent_params = {
                    "q_network": QNetwork_lv1(n_actions=len(self.associations[ap])), 

                    "obs_space_shape": (self.n_ap, ),  
                    "act_space_size": len(self.associations[ap]), 

                    "optimizer": optax.adam(1e-3), 

                    "experience_replay_buffer_size": 1000,
                    "experience_replay_batch_size": 32,
                    "experience_replay_steps": 1,

                    #contextual Bandit type
                    "discount": 0.0, 

                    "epsilon": 1.0, 
                    "epsilon_decay": 0.995,
                    "epsilon_min": 0.05,
                },
                no_ext_mode=True,
            )
            for ap in self.access_points
        }

        # encoded is passed as vector as the observation to the level-2
        
        ## level-3 Link_selection_agent 
        ## context -> 1) AP group 2) stations assigned to them 
        ## context is encoded as tx matrix 
        ## aps, stas are index as coded into the their respective indices domain

        assign_links_agent = {
            ap: RLib(
                agent_type=DQN,

                agent_params = {
                    "q_network": QNetwork_lv3(n_actions=self.n_links), 

                    "obs_space_shape": (self.n_ap, self.stations_per_ap),  
                    "act_space_size": len(self.n_links), 

                    "optimizer": optax.adam(1e-3), 

                    "experience_replay_buffer_size": 1000,
                    "experience_replay_batch_size": 32,
                    "experience_replay_steps": 1,

                    #contextual Bandit type
                    "discount": 0.0, 

                    "epsilon": 1.0, 
                    "epsilon_decay": 0.9995,
                    "epsilon_min": 0.05,
                },
                no_ext_mode=True,
            )
            for ap in self.access_points
        }


        # form the stations, link cross product 
        grids = np.meshgrid(self.stations, list(range(self.n_links)), indexing="ij")
        sta_link = np.stack([grid.ravel() for grid in grids], axis=-1)

        # sta-link mapping , context 
        # state[s,l] = 1 if station s is using the link-l 
        assign_tx_power_agent = {
            (int(sta), int(link)): RLib(
                    agent_type=DQN,
    
                    agent_params = {
                        "q_network": QNetwork_lv1(n_actions=self.n_tx_power_levels), 
    
                        "obs_space_shape": (self.n_sta, self.n_links),  
                        "act_space_size": len(self.n_tx_power_levels), 
    
                        "optimizer": optax.adam(1e-3), 
    
                        "experience_replay_buffer_size": 1000,
                        "experience_replay_batch_size": 32,
                        "experience_replay_steps": 1,
    
                        #contextual Bandit type
                        "discount": 0.0, 
    
                        "epsilon": 1.0, 
                        "epsilon_decay": 0.995,
                        "epsilon_min": 0.05,
                    },
                    no_ext_mode=True,
                )
            for sta, link in sta_link
        }


    @staticmethod
    def _powerset(iterable: Iterable) -> Iterable:
        """
        Returns the powerset of the given iterable. For example, the powerset of [1, 2, 3] is:
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)].

        Parameters
        ----------
        iterable: Iterable
            The iterable to compute the powerset.

        Returns
        -------
        Iterable
            The powerset of the given iterable.
        """

        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
    @staticmethod
    def _powerset_without_emptyset(iterable: Iterable) -> Iterable:
        """
        Returns the powerset of the given iterable. For example, the powerset of [1, 2, 3] is:
        [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)].

        Parameters
        ----------
        iterable: Iterable
            The iterable to compute the powerset.

        Returns
        -------
        Iterable
            The powerset of the given iterable.
        """

        s = sorted(list(iterable))
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> tuple[int]:
        """
        Translates the action of the agent to the list of APs which are sharing the channel.

        Parameters
        ----------
        ap_group_action : int
            The action of agent responsible for the selection of the access points group.
        sharing_ap : int
            The designated access point which has won the DCF contention and is sharing the channel.

        Returns
        -------
        list[int]
            The list of all APs sharing the channel.
        """

        ap_set = set(self.access_points).difference({sharing_ap})
        return tuple(self._powerset(ap_set))[ap_group_action]


    def _sta_group_action_to_sta_group(self, sta_group_action: dict[int, int]) -> list[int]:
        """
        Translates the action of the agent to the list of stations which are served simultaneously.

        Parameters
        ----------
        sta_group_action : dict[int, int]
            The action of agent responsible for the selection of the stations group.

        Returns
        -------
        list[int]
            The list of stations which are served.
        """

        return [self.associations[ap][sta_id] for ap, sta_id in sta_group_action.items()]   

    def _encode_ap_group(self, sharing_ap, selected_ap_group) -> Array: 
        ap_group = np.asarray(list(selected_ap_group) + [sharing_ap])
        return (
            np.isin(
                np.asarray(self.access_points), 
                ap_group
            ).astype(np.int32)
        )

    def _encode_ap_stations_to_tx_vector(self, ap_sta_dict):
        res = np.zeros(shape=(self.n_ap, self.stations_per_ap))
        ap_to_idx = self.ap_to_idx
        rows = [ap_to_idx[ap] for ap in ap_sta_dict.keys()]
        cols = list(ap_sta_dict.values())
        res[rows, cols] = 1
        return res
    
    def _encode_sta_links_vector(self, sta_links: dict[int, list[int]]):
        res = np.zeros((self.n_sta, self.n_links))
        
        # Unpack into matching row/column indices for bulk assignment
        rows = [sta for sta, idx in sta_links.items() for _ in self.link_comb_index_to_links[idx]]
        cols = [link for idx in sta_links.values() for link in self.link_comb_index_to_links[idx]]
        
        res[rows, cols] = 1
        return res
    
    def _encode_sharing_ap(self, sharing_ap) -> Array:
        return np.isin(
            np.asarray(self.access_points), 
            np.asarray(sharing_ap)
        ).astype(np.int32)

    
    
    
