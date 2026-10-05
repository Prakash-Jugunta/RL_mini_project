"""
PPO Agent Configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from rl.state.encoder import STATE_ENCODING_VERSION
from rl.reward import REWARD_VERSION
from rl.curriculum.config import CURRICULUM_VERSION


@dataclass
class PPOConfig:
    """
    Hyperparameters and configuration for Candidate-Aware PPO Agent.
    """
    objective_dim: int = 64
    context_dim: int = 30
    candidate_feature_dim: int = 91
    max_candidates: int = 20

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    rollout_steps: int = 256
    minibatch_size: int = 64
    update_epochs: int = 4

    device: str = "cpu"
    state_encoding_version: str = STATE_ENCODING_VERSION
    reward_version: str = REWARD_VERSION
    curriculum_version: str = CURRICULUM_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective_dim": self.objective_dim,
            "context_dim": self.context_dim,
            "candidate_feature_dim": self.candidate_feature_dim,
            "max_candidates": self.max_candidates,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "rollout_steps": self.rollout_steps,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "device": self.device,
            "state_encoding_version": self.state_encoding_version,
            "reward_version": self.reward_version,
            "curriculum_version": self.curriculum_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
