from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from config import ExperimentConfig
from env_manager import EnvironmentManager


class BaseControlAlgorithm(ABC):
    def __init__(self, config: ExperimentConfig, seed: int):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.env = EnvironmentManager.create_env(config.env_name, seed=seed)

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def close(self) -> None:
        self.env.close()