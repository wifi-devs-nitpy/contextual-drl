from copy import deepcopy
from functools import partial
from typing import Callable, Iterable

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from chex import dataclass, Array, PRNGKey, Scalar, Shape
from flax import linen as nn

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.utils.experience_replay import (
    experience_replay,
    ExperienceReplay,
    ReplayBuffer,
)
from reinforced_lib.utils.jax_utils import forward, gradient_step, init


def ap_group_to_action(
    ap_group: Iterable[int],
    aps: Array,
) -> Array:
    """
    Convert actual AP node indices into the fixed-length binary action vector.

    Example:
        aps      = [0, 4, 8, 9, 10]
        ap_group = [0, 9, 10]

        -> [1, 0, 0, 1, 1]
    """
    aps = jnp.asarray(aps)
    group = jnp.asarray(list(ap_group))
    return jnp.isin(aps, group).astype(jnp.int32)


def action_to_ap_group(
    action: Array,
    aps: Array,
) -> Array:
    """
    Convert a binary action vector into actual AP node indices.

    Example:
        aps    = [0, 4, 8, 9, 10]
        action = [1, 0, 0, 1, 1]

        -> [0, 9, 10]
    """
    action = jnp.asarray(action)
    aps = jnp.asarray(aps)
    return aps[action.astype(bool)]


class BranchingQNetwork(nn.Module):
    """
    Flax branching Q-network.

    For N APs:
        input  -> (batch, *obs_shape)
        output -> (batch, N, 2)

    For branch i:
        0 -> do not transmit with sharing AP
        1 -> transmit with sharing AP
    """

    num_branches: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Shared value stream V(s).
        value = nn.Dense(1)(x)

        # Branch-specific advantage stream A_i(s, a_i).
        advantages = nn.Dense(
            self.num_branches * 2
        )(x)

        advantages = advantages.reshape(
            x.shape[0],
            self.num_branches,
            2,
        )

        # Dueling-style centering per branch.
        advantages = (
            advantages
            - advantages.mean(axis=-1, keepdims=True)
        )

        # (batch, num_branches, 2)
        return value[:, None, :] + advantages


@dataclass
class BranchingDQNState(AgentState):
    """State container following the supplied reinforced-lib DQN structure."""

    params: dict
    net_state: dict
    opt_state: optax.OptState

    replay_buffer: ReplayBuffer
    prev_env_state: Array
    epsilon: Scalar


class BranchingDQN(BaseAgent):
    """
    Branching DQN for the AP-group selection problem.

    Observation
    -----------
    A one-hot sharing-AP vector with length N_AP.

    Example:
        APS = [0, 4, 8, 9, 10]
        sharing AP = 9

        observation = [0, 0, 0, 1, 0]

    Action
    ------
    A binary vector with length N_AP.

        action[i] = 0 -> AP i is not selected
        action[i] = 1 -> AP i transmits with the sharing AP

    Hard constraint
    ---------------
    The sharing AP MUST be selected:

        action[sharing_position] == 1

    This is enforced in:
        - greedy action selection
        - epsilon exploration
        - Bellman next-action selection
    """

    def __init__(
        self,
        q_network: nn.Module,
        obs_space_shape: Shape,
        num_branches: int,
        optimizer: optax.GradientTransformation = None,
        experience_replay_buffer_size: int = 10000,
        experience_replay_batch_size: int = 64,
        experience_replay_steps: int = 5,
        discount: Scalar = 0.99,
        epsilon: Scalar = 1.0,
        epsilon_decay: Scalar = 0.999,
        epsilon_min: Scalar = 0.001,
    ) -> None:

        assert num_branches > 0
        assert (
            experience_replay_buffer_size
            > experience_replay_batch_size
            > 0
        )
        assert experience_replay_steps > 0
        assert 0.0 <= discount <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert 0.0 <= epsilon_decay <= 1.0
        assert 0.0 <= epsilon_min <= epsilon

        if optimizer is None:
            optimizer = optax.adam(1e-3)

        self.obs_space_shape = (
            tuple(obs_space_shape)
            if jnp.ndim(obs_space_shape) > 0
            else (obs_space_shape,)
        )
        self.num_branches = int(num_branches)

        # One complete binary action vector is stored per transition.
        er = experience_replay(
            experience_replay_buffer_size,
            experience_replay_batch_size,
            self.obs_space_shape,
            (self.num_branches,),
        )

        self.init = jax.jit(
            partial(
                self.init,
                obs_space_shape=self.obs_space_shape,
                q_network=q_network,
                optimizer=optimizer,
                er=er,
                epsilon=epsilon,
            )
        )

        self.update = jax.jit(
            partial(
                self.update,
                step_fn=partial(
                    gradient_step,
                    optimizer=optimizer,
                    loss_fn=partial(
                        self.loss_fn,
                        q_network=q_network,
                        discount=discount,
                    ),
                ),
                er=er,
                experience_replay_steps=experience_replay_steps,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        )

        self.sample = jax.jit(
            partial(
                self.sample,
                q_network=q_network,
            )
        )

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "obs_space_shape": gym.spaces.Sequence(
                gym.spaces.Box(1, jnp.inf, (1,), int)
            ),
            "num_branches": gym.spaces.Box(
                1, jnp.inf, (1,), int
            ),
            "experience_replay_buffer_size": gym.spaces.Box(
                1, jnp.inf, (1,), int
            ),
            "experience_replay_batch_size": gym.spaces.Box(
                1, jnp.inf, (1,), int
            ),
            "experience_replay_steps": gym.spaces.Box(
                1, jnp.inf, (1,), int
            ),
            "discount": gym.spaces.Box(
                0.0, 1.0, (1,), float
            ),
            "epsilon": gym.spaces.Box(
                0.0, 1.0, (1,), float
            ),
            "epsilon_decay": gym.spaces.Box(
                0.0, 1.0, (1,), float
            ),
            "epsilon_min": gym.spaces.Box(
                0.0, 1.0, (1,), float
            ),
        })

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "env_state": gym.spaces.Box(
                -jnp.inf,
                jnp.inf,
                self.obs_space_shape,
                float,
            ),
            "action": gym.spaces.MultiBinary(
                self.num_branches
            ),
            "reward": gym.spaces.Box(
                -jnp.inf,
                jnp.inf,
                (1,),
                float,
            ),
            "terminal": gym.spaces.MultiBinary(1),
        })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            "env_state": gym.spaces.Box(
                -jnp.inf,
                jnp.inf,
                self.obs_space_shape,
                float,
            )
        })

    @property
    def action_space(self) -> gym.spaces.MultiBinary:
        return gym.spaces.MultiBinary(
            self.num_branches
        )

    @staticmethod
    def init(
        key: PRNGKey,
        obs_space_shape: Shape,
        q_network: nn.Module,
        optimizer: optax.GradientTransformation,
        er: ExperienceReplay,
        epsilon: Scalar,
    ) -> BranchingDQNState:

        # Flax network receives a batch dimension.
        x_dummy = jnp.empty(
            (1,) + tuple(obs_space_shape)
        )

        params, net_state = init(
            q_network,
            key,
            x_dummy,
        )

        opt_state = optimizer.init(params)
        replay_buffer = er.init()

        return BranchingDQNState(
            params=params,
            net_state=net_state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=jnp.zeros(
                obs_space_shape
            ),
            epsilon=epsilon,
        )

    @staticmethod
    def _joint_q_from_branch_q(
        q_values: Array,
        actions: Array,
    ) -> Array:
        """
        Select Q(s,a_i) for each branch and aggregate into one scalar.

        q_values: (..., N, 2)
        actions:  (..., N)
        returns:  (..., 1)
        """
        branch_q = jnp.take_along_axis(
            q_values,
            actions.astype(jnp.int32)[..., None],
            axis=-1,
        ).squeeze(-1)

        return branch_q.mean(
            axis=-1,
            keepdims=True,
        )

    @staticmethod
    def _mask_sharing_ap(
        q_values: Array,
        observations: Array,
    ) -> Array:
        """
        Mask the invalid action "reject sharing AP".

        q_values:
            (batch, N, 2)

        observations:
            (batch, N)

        For each sample, observation's 1 identifies the sharing AP.
        """
        sharing_position = jnp.argmax(
            observations,
            axis=-1,
        )

        branch_ids = jnp.arange(
            q_values.shape[1]
        )[None, :]

        sharing_mask = (
            branch_ids
            == sharing_position[:, None]
        )

        # action 0 = reject; forbidden for sharing AP.
        return q_values.at[:, :, 0].set(
            jnp.where(
                sharing_mask,
                -jnp.inf,
                q_values[:, :, 0],
            )
        )

    @staticmethod
    def loss_fn(
        params: dict,
        key: PRNGKey,
        net_state: dict,
        params_target: dict,
        net_state_target: dict,
        batch: tuple,
        q_network: nn.Module,
        discount: Scalar,
    ) -> tuple[Scalar, dict]:

        states, actions, rewards, terminals, next_states = batch

        q_key, target_key = jax.random.split(key)

        # ----------------------------------------------------
        # Q(s,a)
        # ----------------------------------------------------
        q_values, net_state = forward(
            q_network,
            params,
            net_state,
            q_key,
            states,
        )

        q_current = BranchingDQN._joint_q_from_branch_q(
            q_values,
            actions,
        )

        # ----------------------------------------------------
        # Q_target(s', .)
        # ----------------------------------------------------
        q_next, _ = forward(
            q_network,
            params_target,
            net_state_target,
            target_key,
            next_states,
        )

        # Enforce the sharing-AP constraint before selecting
        # the greedy next action.
        q_next = BranchingDQN._mask_sharing_ap(
            q_next,
            next_states,
        )

        next_actions = jnp.argmax(
            q_next,
            axis=-1,
        )

        q_next_joint = BranchingDQN._joint_q_from_branch_q(
            q_next,
            next_actions,
        )

        target = rewards + (
            (1 - terminals)
            * discount
            * q_next_joint
        )

        target = jax.lax.stop_gradient(
            target
        )

        loss = optax.l2_loss(
            q_current,
            target,
        ).mean()

        return loss, net_state

    @staticmethod
    def update(
        state: BranchingDQNState,
        key: PRNGKey,
        env_state: Array,
        action: Array,
        reward: Scalar,
        terminal: bool,
        step_fn: Callable,
        er: ExperienceReplay,
        experience_replay_steps: int,
        epsilon_decay: Scalar,
        epsilon_min: Scalar,
    ) -> BranchingDQNState:

        replay_buffer = er.append(
            state.replay_buffer,
            state.prev_env_state,
            action,
            reward,
            terminal,
            env_state,
        )

        # Follow the supplied DQN implementation: the target copy is frozen
        # for the current update call.
        params_target = deepcopy(
            state.params
        )
        net_state_target = deepcopy(
            state.net_state
        )

        def for_loop_fn(
            _: int,
            carry: tuple,
        ) -> tuple:

            params, net_state, opt_state, key = carry

            batch_key, network_key, key = (
                jax.random.split(key, 3)
            )

            loss_params = (
                network_key,
                net_state,
                params_target,
                net_state_target,
                er.sample(
                    replay_buffer,
                    batch_key,
                ),
            )

            params, net_state, opt_state, _ = (
                step_fn(
                    params,
                    loss_params,
                    opt_state,
                )
            )

            return (
                params,
                net_state,
                opt_state,
                key,
            )

        params, net_state, opt_state, _ = (
            jax.lax.fori_loop(
                0,
                experience_replay_steps
                * er.is_ready(replay_buffer),
                for_loop_fn,
                (
                    state.params,
                    state.net_state,
                    state.opt_state,
                    key,
                ),
            )
        )

        return BranchingDQNState(
            params=params,
            net_state=net_state,
            opt_state=opt_state,
            replay_buffer=replay_buffer,
            prev_env_state=env_state,
            epsilon=jax.lax.max(
                state.epsilon * epsilon_decay,
                epsilon_min,
            ),
        )

    @staticmethod
    def sample(
        state: BranchingDQNState,
        key: PRNGKey,
        env_state: Array,
        q_network: nn.Module,
    ) -> Array:

        network_key, action_key, explore_key = (
            jax.random.split(key, 3)
        )

        # One environment observation -> one network batch item.
        q, _ = forward(
            q_network,
            state.params,
            state.net_state,
            network_key,
            env_state[None, ...],
        )

        # (1, N, 2) -> (N, 2)
        q = q[0]

        # One-hot sharing AP -> branch index.
        sharing_position = jnp.argmax(
            env_state
        )

        branch_ids = jnp.arange(
            q.shape[0]
        )

        sharing_mask = (
            branch_ids
            == sharing_position
        )

        # Sharing AP may not be rejected.
        q_masked = q.at[:, 0].set(
            jnp.where(
                sharing_mask,
                -jnp.inf,
                q[:, 0],
            )
        )

        greedy_action = jnp.argmax(
            q_masked,
            axis=-1,
        ).astype(jnp.int32)

        # Explore a complete binary action vector.
        random_action = jax.random.randint(
            action_key,
            shape=(q.shape[0],),
            minval=0,
            maxval=2,
            dtype=jnp.int32,
        )

        # Even exploratory actions must include the sharing AP.
        random_action = random_action.at[
            sharing_position
        ].set(1)

        explore = (
            jax.random.uniform(explore_key)
            < state.epsilon
        )

        action = jnp.where(
            explore,
            random_action,
            greedy_action,
        )

        # Final hard guarantee.
        action = action.at[
            sharing_position
        ].set(1)

        return action


__all__ = [
    "BranchingDQN",
    "BranchingDQNState",
    "BranchingQNetwork",
    "ap_group_to_action",
    "action_to_ap_group",
]
