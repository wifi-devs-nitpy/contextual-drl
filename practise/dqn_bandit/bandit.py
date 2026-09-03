import numpy as np 
N_ACTIONS = 5 

rng = np.random.default_rng()

def sample_state(rng: np.random.default_rng):
    return rng.uniform(0.0, 1.0, size=2).astype(np.float32)

def expected_reward(state, action):
    x1, x2 = state 

    rewards = np.array([
        0.20 + 0.30 * x1,
        0.10 + 0.80 * x2,
        0.30 + 0.40 * x1 + 0.20 * x2,
        0.85 - 0.60 * abs(x1 - x2),
        0.20 + 0.70 * (1.0 - x1),
    ])

    return float(rewards[action])

def get_reward(rng, state, action, noise_std=0.10):
    mean = expected_reward(state, action)

    return float(
        rng.normal(mean, noise_std)
    )

def optimal_action(state):
    rewards = [
        expected_reward(state, action) for action in range(N_ACTIONS)
    ]

    return int(np.argmax(rewards))

