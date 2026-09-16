"""
Tabular Q-learning agent.

Why tabular Q-learning instead of PPO/DQN for a 7-day hackathon:
  - Zero heavy dependencies (no torch/CUDA - which, as we found out, can eat
    your entire disk quota installing GPU libraries you don't need for a
    proof of concept).
  - Trains in seconds on a laptop, easy to inspect (the whole "model" is a
    dict you can print), and easy to explain to judges.
  - If you have time/compute left on Day 5-6, swap this for
    stable-baselines3 PPO on the exact same MicrofinanceEnv - nothing about
    the environment needs to change, only the training loop.

We discretize the 8-dim continuous observation into a small number of bins
per feature. This is the classic tabular-RL trick for handling continuous
state spaces without a neural network.
"""

import numpy as np
import pickle
from collections import defaultdict

from microfinance_env import MicrofinanceEnv, N_ACTIONS


# Bin edges chosen from domain reasoning, not arbitrary equal-width bins.
# Each entry: (feature_index, edges) - edges split the range into len(edges)+1 bins.
BIN_EDGES = [
    (0, [0.15, 0.35]),        # income_cv: low / medium / high volatility
    (1, [0.35, 0.5, 0.7]),    # dti: comfortable / watch / high / danger
    (2, [0.35, 0.7]),         # buffer_days_scaled: thin / moderate / healthy
    (3, [-0.15, 0.1]),        # seasonal_deviation: below / at / above expected
    (4, [-0.02, 0.02]),       # momentum: declining / flat / improving
    (5, [0.15, 0.4]),         # arrears_ratio: none-ish / some / high
    (6, [0.15, 0.45]),        # consecutive_missed (scaled): 0 / 1 / 2+
    (7, [0.3, 0.7]),          # months_remaining_frac: late / mid / early
]


def discretize(obs):
    """Map the continuous 8-dim observation to a tuple of small ints (a hashable state)."""
    bins = []
    for idx, edges in BIN_EDGES:
        bins.append(int(np.digitize(obs[idx], edges)))
    return tuple(bins)


class QLearningAgent:
    def __init__(self, n_actions=N_ACTIONS, alpha=0.15, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_episodes=3000):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def epsilon(self, episode):
        frac = min(episode / self.epsilon_decay_episodes, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, state, episode=None, greedy=False):
        if not greedy and episode is not None and np.random.random() < self.epsilon(episode):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), loaded)


def train(n_episodes=4000, max_steps=48, seed=0, verbose_every=500):
    env = MicrofinanceEnv(seed=seed)
    agent = QLearningAgent(epsilon_decay_episodes=int(n_episodes * 0.75))

    episode_returns = []
    outcome_log = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        state = discretize(obs)
        total_r = 0.0
        outcome = "truncated"

        for _ in range(max_steps):
            action = agent.act(state, episode=ep)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            next_state = discretize(next_obs)
            agent.update(state, action, reward, next_state, terminated or truncated)

            state = next_state
            total_r += reward
            if terminated or truncated:
                outcome = step_info.get("outcome", "truncated")
                break

        episode_returns.append(total_r)
        outcome_log.append(outcome)

        if verbose_every and (ep + 1) % verbose_every == 0:
            recent_returns = np.mean(episode_returns[-verbose_every:])
            recent_defaults = outcome_log[-verbose_every:].count("default") / verbose_every
            print(f"episode {ep+1:5d}  avg_return={recent_returns:6.2f}  "
                  f"default_rate={recent_defaults:.1%}  epsilon={agent.epsilon(ep):.2f}  "
                  f"states_visited={len(agent.q_table)}")

    return agent, episode_returns, outcome_log


if __name__ == "__main__":
    agent, returns, outcomes = train(n_episodes=8192000)
    agent.save("q_table.pkl")
    print("\nSaved trained agent to q_table.pkl")
    print(f"Final 500-episode default rate: {outcomes[-500:].count('default')/500:.1%}")
    print(f"Final 500-episode avg return:   {np.mean(returns[-500:]):.2f}")
