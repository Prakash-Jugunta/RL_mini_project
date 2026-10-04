"""
DQN Configuration and Epsilon Decay Schedule.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DQNConfig:
    """
    Configuration dataclass for Masked Double DQN Agent.
    These are implementation defaults for Phase 10 validation & smoke testing only.
    """
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 10000
    learning_starts: int = 100
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_frequency: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1000

    gradient_clip_norm: float = 10.0

    seed: int = 42
    device: str = "cpu"

    max_candidates: int = 20
    candidate_feature_dim: int = 91
    objective_dim: int = 64
    context_dim: int = 30
    state_encoding_version: str = "phase6-public-prev-candidate-v2"


def get_epsilon_at_step(config: DQNConfig, step: int) -> float:
    """
    Calculates linear epsilon decay at a given environment step.
    """
    if step <= 0:
        return config.epsilon_start
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end

    decay_ratio = float(step) / float(config.epsilon_decay_steps)
    eps = config.epsilon_start - decay_ratio * (config.epsilon_start - config.epsilon_end)
    return float(max(config.epsilon_end, min(config.epsilon_start, eps)))
