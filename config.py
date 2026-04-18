from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str = "CliffWalking-v1"
    num_episodes: int = 800
    alpha: float = 0.5
    gamma: float = 1.0
    epsilon: float = 0.1
    n_step: int = 4
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    eval_episodes: int = 100
    smoothing_window: int = 20
    max_steps_per_episode: int = 500
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        return asdict(self)