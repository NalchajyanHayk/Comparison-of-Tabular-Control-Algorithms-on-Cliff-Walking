from __future__ import annotations

import numpy as np


class EpsilonGreedyPolicy:
    @staticmethod
    def select_action(q_table: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
        n_actions = q_table.shape[1]
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))

        row = q_table[state]
        max_value = np.max(row)
        greedy_actions = np.flatnonzero(row == max_value)
        return int(rng.choice(greedy_actions))

    @staticmethod
    def greedy_action(q_table: np.ndarray, state: int) -> int:
        row = q_table[state]
        max_value = np.max(row)
        greedy_actions = np.flatnonzero(row == max_value)
        return int(np.random.choice(greedy_actions))

    @staticmethod
    def expected_value(q_table: np.ndarray, state: int, epsilon: float) -> float:
        n_actions = q_table.shape[1]
        row = q_table[state]
        max_value = np.max(row)
        greedy_actions = np.flatnonzero(row == max_value)

        probs = np.ones(n_actions) * (epsilon / n_actions)
        for action in greedy_actions:
            probs[action] += (1.0 - epsilon) / len(greedy_actions)

        return float(np.dot(probs, row))