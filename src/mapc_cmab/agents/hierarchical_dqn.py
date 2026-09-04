from collections import defaultdict
from itertools import chain
from typing import Callable
from copy import copy

import numpy as np
from chex import Array, Shape, Scalar
from reinforced_lib import RLib

from mapc_cmab.agents.mapc_agent import MapcAgent

class HierarchicalMapcDQNAgent(MapcAgent):
    """
    The hierarchical MAB agent responsible for the selection of the AP and station pairs.
    The agent consists of three phases:

      1. Selection of the group of APs which are sharing the channel.
      2. Selection of the stations which are served simultaneously by the APs in the group.
      3. selection of the links used to serve a station
      4. Selection of the transmission power for each ap-sta link in the group.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    find_groups_agent : RLib
        The agent which selects the group of APs sharing the channel.
    assign_stations_agents : dict[int, RLib]
        The agents which select the stations served by the APs.
    select_tx_power_agent : dict[int, RLib]
        The agent which selects the transmission power for a particular link.

    ap_group_action_to_ap_group : Callable
        The function which translates the action of the agent to the tuple of APs sharing the channel.
    sta_group_action_to_sta_group : Callable
        The function which translates the action of the agent to the list of served stations.
    link_action_to_links_group: Callable
        The function which translate the link action of the agent to the list of links that the station uses
    encode_ap_group: Callable 
        The function which encodes the selected ap group for the level-2 agents as a context.  
    encode_ap_stations_to_tx_vector: Callable 
        The function which encodes the ap_stations in to mi
        n tx_vector for the level-3 agents as a context.
    encode_sta_links_vector: Callable
        The function that encodes the output of the thrid level agent as a context for the level-4 agent 
    tx_matrix_shape : Shape
        The shape of the transmission matrix.
    """

    def __init__(
            self, 
            associations: dict[int, list[int]], 
            find_groups_agent: RLib, 
            assign_stations_agent: dict[int, RLib], 
            assign_links_agent: dict[int, RLib], 
            assign_tx_power_agent: dict[tuple[int, int], RLib],
            encode_sharing_ap: Callable,
            encode_ap_group: Callable, 
            encode_ap_stations_to_tx_vector: Callable, 
            encode_sta_links_vector: Callable, 
            ap_group_action_to_ap_group: Callable,
            link_comb_index_to_links: dict[int, list],
            sta_index_mapping: dict[int, int],
            n_links: int,
            n_tx_power_levels: int
        ):

        self.associations = associations
        self.inv_associations = {sta: ap for ap in associations.keys() for sta in associations[ap]}
        self.find_groups_agent = find_groups_agent
        self.assign_stations_agent = assign_stations_agent
        self.assign_links_agent = assign_links_agent
        self.assign_tx_power_agent = assign_tx_power_agent 
        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.link_comb_index_to_links = link_comb_index_to_links
        self.sta_index_mapping = sta_index_mapping
        self.n_links = n_links

        self.encoded_sharing_ap = encode_sharing_ap
        self.encode_ap_group = encode_ap_group 
        self.encode_ap_stations_to_tx_vector = encode_ap_stations_to_tx_vector 
        self.encode_sta_links_vector = encode_sta_links_vector

        self.n_tx_power_levels = n_tx_power_levels 

        self.find_groups_agent_last_step = 0
        self.find_groups_agent_last_action = 0 

        self.assign_stations_agent_last_step = defaultdict(int)
        self.assign_stations_agent_last_action = defaultdict(int)

        self.assign_links_agent_last_step = defaultdict(int)
        self.assign_links_agent_last_action = defaultdict(int)

        self.assign_tx_power_agent_last_step = defaultdict(int)
        self.assign_tx_power_agent_last_action = defaultdict(int)
                        
        self.step = 0
        self.rewards = [] 

        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.asarray(list(associations.keys()))
        self.stations = np.asarray(list(chain.from_iterable(associations.values())))
        self.n_nodes = len(self.access_points) + len(list(chain.from_iterable(associations.values())))


    def sample(self, reward) -> tuple[Array, Array]: 
        """ 
        Samples the agent to the transmission matrix 

        Returns
        --------
        tuple 
            The transmission matrix and tx_power vector 
        """

        self.step += 1
        self.rewards.append(reward)

        # loop invariant 
        # everytime the reward is appended, it gets into the index == (step-1)
        # this means to update a specific agent with a reward I must know the last_step in which it took the action. 

        sharing_ap = np.random.choice(self.access_points).item()
        sharing_sta = np.random.choice(self.associations[sharing_ap]).item()

        context_lvl1 = self.encoded_sharing_ap(sharing_ap, sharing_sta)

        find_groups_agent_action = self.find_groups_agent.sample(
                                update_observations={
                                    'env_state': context_lvl1, 
                                    'action': self.find_groups_agent_last_action,
                                    'reward': self.rewards[self.find_groups_agent_last_step], 
                                    'terminal': False
                                }, 
                                sample_observations={
                                    "env_state": context_lvl1
                                }
                        ).item()

        self.find_groups_agent_last_action = find_groups_agent_action 
        self.find_groups_agent_last_step = self.step 
        
        selected_ap_group = self.ap_group_action_to_ap_group(ap_group_action=find_groups_agent_action, sharing_ap=sharing_ap)

        # encoding the context for the level-2 
        context_lvl2 = self.encode_ap_group(sharing_ap=sharing_ap, selected_ap_group=selected_ap_group)
        selected_aps = self.access_points[context_lvl2.astype(bool)]

        ap_sta_pairs = {
            int(ap): self.assign_stations_agent[ap].sample(
                update_observations={
                    'env_state': context_lvl2, 
                    'action': self.assign_stations_agent_last_action[ap],
                    'reward': self.rewards[self.assign_stations_agent_last_step[ap]], 
                    'terminal': False
                }, 
                sample_observations={
                    'env_state': context_lvl2
                }
            ).item()
            for ap in selected_ap_group
        } 
        #index of ap is actual node index of ap , index of sta is relative index of sta in associations[ap]
        #update the last step, and last action 
        for ap, sta_idx in ap_sta_pairs.items():
            self.assign_stations_agent_last_step[ap] = self.step 
            self.assign_stations_agent_last_action[ap] = sta_idx
        
        ap_sta_pairs[sharing_ap] = (self.associations[sharing_ap] == sharing_sta).argmax().item()

        # encoding ap_sta_pairs a context for the level-3 
        context_lvl3 = self.encode_ap_stations_to_tx_vector(ap_sta_pairs)

        ap_sta_links = {
                    int(ap): self.assign_links_agent[ap].sample(
                        update_observations={
                            'env_state': context_lvl3, 
                            'action': self.assign_links_agent_last_action[ap],
                            'reward': self.rewards[self.assign_links_agent_last_step[ap]], 
                            'terminal': False
                        }, 
                        sample_observations={
                            'env_state': context_lvl3
                        }
                    ).item()
                    for ap in selected_aps
                }
        
        #update the last step, and last action 
        for ap, link_idx in ap_sta_links.items():
            self.assign_links_agent_last_step[ap] = self.step 
            self.assign_links_agent_last_action[ap] = link_idx


        # converting ap_sta_links to sta_links 
        sta_link_indices = {}
        for ap, link_idx in ap_sta_links.items():
            sta_selected_rel_index = ap_sta_pairs[ap]
            sta_index = self.associations[ap][sta_selected_rel_index]
            sta_link_indices[sta_index] = link_idx 

        # encoding it into a context vector for the level-4 
        context_lvl4 = self.encode_sta_links_vector(sta_link_indices)

        sta_links = {
            sta: self.link_comb_index_to_links[link_index]
            for sta, link_index in sta_link_indices.items()
        }

        link_ap_sta = {
            link: {
                "tx_matrix": np.zeros((self.n_nodes, self.n_nodes)), 
                "tx_power_indices": np.zeros(self.n_nodes, dtype=np.int32) 
            }
            for link in range(self.n_links)
        }
        
        # sampling the tx_power for the 
        for sta, links in sta_links.items():
            for link in links: 
                tx_power_index = self.assign_tx_power_agent[sta, link].sample(
                    update_observations = {
                            'env_state': context_lvl4, 
                            'action': self.assign_tx_power_agent_last_action[sta, link],
                            'reward': self.rewards[self.assign_tx_power_agent_last_step[sta, link]], 
                            'terminal': False
                        }, 
                        sample_observations={
                            'env_state': context_lvl4
                        }
                ).item()
                link_ap_sta[link]["tx_matrix"][self.inv_associations[sta], sta] = 1
                link_ap_sta[link]["tx_power_indices"][self.inv_associations[sta]] = tx_power_index
                self.assign_tx_power_agent_last_action[sta, link] = tx_power_index
                self.assign_tx_power_agent_last_step[sta, link] = self.step 

        tx_matrices = np.array(list(link_ap_sta[r]["tx_matrix"] for r in range(0, self.n_links)), dtype=np.int16)
        tx_power_indices = np.array(list(link_ap_sta[r]["tx_power_indices"] for r in range(0, self.n_links)), dtype=np.int16)


        return (tx_matrices, tx_power_indices)