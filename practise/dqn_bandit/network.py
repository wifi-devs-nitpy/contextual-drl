from flax import linen as nn
from chex import Array

from reinforced_lib import RLib
from reinforced_lib.agents.deep import DQN

import optax


class QNetwork(nn.Module):
    """Simple MLP returning one Q-value per action."""

    n_actions: int

    @nn.compact
    def __call__(self, x: Array) -> Array:

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        q_values = nn.Dense(self.n_actions)(x)

        return q_values


rl = RLib(
    agent_type=DQN,

    agent_params={
        "q_network": QNetwork(n_actions=5),

        "obs_space_shape": (2,),

        "act_space_size": 5,

        "optimizer": optax.adam(1e-3),

        "experience_replay_buffer_size": 10_000,

        "experience_replay_batch_size": 64,

        "experience_replay_steps": 1,

        "discount": 0.0,

        "epsilon": 1.0,

        "epsilon_decay": 0.995,

        "epsilon_min": 0.05,
    },

    no_ext_mode=True,
)