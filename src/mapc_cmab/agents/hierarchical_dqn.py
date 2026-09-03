from collections import defaultdict
from itertools import chain
from typing import Callable
from copy import copy

import numpy as np
from chex import Array, Shape, Scalar
from reinforced_lib import RLib

from mapc_cmab.agents.mapc_agent import MapcAgent

class HierarchicalMapcAgent(MapcAgent):
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
        The function which encodes the ap_stations in to min tx_vector for the level-3 agents as a context.
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
            encode_ap_group: Callable, 
            encode_ap_stations_to_tx_vector: Callable, 
            encode_sta_links_vector: Callable, 
            tx_matrix_shape: Shape, 
            tx_power_levels: int
        ):

        self.associations = self.associations
        self.find_groups_agent = find_groups_agent
        self.assign_stations_agent = assign_stations_agent
        self.assign_links_agent = assign_links_agent
        self.assign_tx_power_agent = assign_tx_power_agent 

        self.encode_ap_group = encode_ap_group 
        self.encode_ap_stations_to_tx_vector = encode_ap_stations_to_tx_vector 
        self.encode_sta_links_vector = encode_sta_links_vector

        self.tx_matrix_shape = tx_matrix_shape
        self.tx_power_levels = tx_power_levels 

        self.step = 0
        self.rewards = [] 

    def sample(self, reward) -> tuple[Array, Array]: 
        """ 
        Samples the agent to the transmission matrix 

        Returns
        --------
        tuple 
            The transmission matrix and tx_power vector 
        """

        self.step += 1
        self.reward.append(reward)

        # loop invariant 
        # everytime the reward is appended, it gets into the index == (step-1)
        # this means to update a specific agent with a reward I must know the last_step in which it took the action. 

        


