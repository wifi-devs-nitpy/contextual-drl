from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    terminal: bool


class ContextualBandit:
    """
    Five-arm contextual bandit.

    The context is a 2-dimensional vector.

    Given context x and action a:
        reward ~ Normal(mean_reward(x, a), noise_std)

    Each interaction is one independent bandit step.
    """

    N_ARMS = 5
    STATE_DIM = 2

    def __init__(
        self,
        seed: int = 42,
        noise_std: float = 0.10,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

        self.state = np.zeros(self.STATE_DIM, dtype=np.float32)

    def reset(self) -> np.ndarray:
        """Generate the first context."""
        self.state = self._sample_context()
        return self.state.copy()

    def _sample_context(self) -> np.ndarray:
        """Generate a context in [0, 1]^2."""
        return self.rng.uniform(
            low=0.0,
            high=1.0,
            size=self.STATE_DIM,
        ).astype(np.float32)

    def expected_reward(
        self,
        state: np.ndarray,
        action: int,
    ) -> float:
        """
        Return the noiseless expected reward.
        """

        x1, x2 = state

        if action == 0:
            return 0.20 + 0.30 * x1

        if action == 1:
            return 0.10 + 0.80 * x2

        if action == 2:
            return 0.30 + 0.40 * x1 + 0.20 * x2

        if action == 3:
            return 0.85 - 0.60 * abs(x1 - x2)

        if action == 4:
            return 0.20 + 0.70 * (1.0 - x1)

        raise ValueError(f"Invalid action: {action}")

    def best_action(self, state: np.ndarray) -> int:
        """Return the optimal action for this context."""
        rewards = [
            self.expected_reward(state, action)
            for action in range(self.N_ARMS)
        ]

        return int(np.argmax(rewards))

    def step(self, action: int) -> StepResult:
        """
        Execute one contextual-bandit interaction.

        Unlike an ordinary sequential environment, there is no
        meaningful next-state dependency in the reward.
        """

        if not 0 <= action < self.N_ARMS:
            raise ValueError(
                f"action must be in [0, {self.N_ARMS - 1}]"
            )

        mean_reward = self.expected_reward(
            self.state,
            action,
        )

        reward = float(
            self.rng.normal(
                loc=mean_reward,
                scale=self.noise_std,
            )
        )

        # Every interaction gets a fresh context.
        next_state = self._sample_context()

        self.state = next_state

        return StepResult(
            state=next_state.copy(),
            reward=reward,
            terminal=False,
        )