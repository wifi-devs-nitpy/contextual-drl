from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations, product
from typing import Iterable, Iterator

import numpy as np
from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN

from mapc_cmab.agents.mapc_agent import MapcAgent


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
    tx_power_levels : int
        The number of Transmission power levels 
    seed: int
    
    """
