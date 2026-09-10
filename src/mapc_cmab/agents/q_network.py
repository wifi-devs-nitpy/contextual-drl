from flax import linen as nn
from chex import Array


class QNetwork(nn.Module):
    n_actions: int
    hidden_dims: tuple[int, ...] = (64, 64)
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)

            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            x = nn.relu(x)

        return nn.Dense(self.n_actions)(x)