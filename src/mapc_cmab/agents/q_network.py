from flax import linen as nn 
import optax 
from chex import Array 


class QNetwork_lv1(nn.Module):
    n_actions: int = 5 

    @nn.compact
    def __call__(self, x: Array) -> Array: 

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        q_values = nn.Dense(self.n_actions)(x)
        return q_values

class QNetwork_lv2(nn.Module):
    n_actions: int = 5 

    @nn.compact
    def __call__(self, x: Array) -> Array: 

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        q_values = nn.Dense(self.n_actions)(x)
        return q_values

class QNetwork_lv3(nn.Module):
    n_actions: int = 5 

    @nn.compact
    def __call__(self, x: Array) -> Array: 

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        q_values = nn.Dense(self.n_actions)(x)
        return q_values

class QNetwork_lv4(nn.Module):
    n_actions: int = 5 

    @nn.compact
    def __call__(self, x: Array) -> Array: 

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        x = nn.Dense(64)(x)
        x = nn.relu(x)

        q_values = nn.Dense(self.n_actions)(x)
        return q_values

