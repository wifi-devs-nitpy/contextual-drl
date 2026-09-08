"""Configurable Q-networks for the hierarchical MAPC DQN agents.

The original project uses four separate class names only to distinguish hierarchy
levels. The implementation below keeps those names but makes the architecture
configurable so the experiment runner can compare smaller/larger MLPs.
"""

from __future__ import annotations

from typing import Sequence

from flax import linen as nn
from chex import Array


class _ConfigurableQNetwork(nn.Module):
    n_actions: int = 5
    hidden_dims: Sequence[int] = (64, 64)
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for width in self.hidden_dims:
            x = nn.Dense(width)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return nn.Dense(self.n_actions)(x)


class QNetwork_lv1(_ConfigurableQNetwork):
    pass


class QNetwork_lv2(_ConfigurableQNetwork):
    pass


class QNetwork_lv3(_ConfigurableQNetwork):
    pass


class QNetwork_lv4(_ConfigurableQNetwork):
    pass
