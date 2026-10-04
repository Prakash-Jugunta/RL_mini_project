"""
Dict-Observation Replay Buffer for Masked Double DQN.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any


class DictReplayBuffer:
    """
    Replay buffer designed for Gymnasium Dict observations:
      - objective:      (64,)
      - candidates:     (20, 91)
      - candidate_mask: (20,)
      - context:        (30,)

    Does NOT store private evaluator metadata (expected_role, semantic_role, etc.).
    """

    def __init__(
        self,
        capacity: int = 10000,
        objective_dim: int = 64,
        max_candidates: int = 20,
        candidate_feature_dim: int = 91,
        context_dim: int = 30,
        seed: Optional[int] = 42,
    ):
        self.capacity = capacity
        self.objective_dim = objective_dim
        self.max_candidates = max_candidates
        self.candidate_feature_dim = candidate_feature_dim
        self.context_dim = context_dim

        self.rng = np.random.default_rng(seed)

        # Preallocated circular storage buffers
        self.obs_objective = np.zeros((capacity, objective_dim), dtype=np.float32)
        self.obs_candidates = np.zeros((capacity, max_candidates, candidate_feature_dim), dtype=np.float32)
        self.obs_candidate_mask = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.obs_context = np.zeros((capacity, context_dim), dtype=np.float32)

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)

        self.next_obs_objective = np.zeros((capacity, objective_dim), dtype=np.float32)
        self.next_obs_candidates = np.zeros((capacity, max_candidates, candidate_feature_dim), dtype=np.float32)
        self.next_obs_candidate_mask = np.zeros((capacity, max_candidates), dtype=np.float32)
        self.next_obs_context = np.zeros((capacity, context_dim), dtype=np.float32)

        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)

        self.pos: int = 0
        self.size: int = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Adds a single transition to the replay buffer."""
        idx = self.pos

        self.obs_objective[idx] = obs["objective"]
        self.obs_candidates[idx] = obs["candidates"]
        self.obs_candidate_mask[idx] = obs["candidate_mask"]
        self.obs_context[idx] = obs["context"]

        self.actions[idx] = action
        self.rewards[idx] = reward

        self.next_obs_objective[idx] = next_obs["objective"]
        self.next_obs_candidates[idx] = next_obs["candidates"]
        self.next_obs_candidate_mask[idx] = next_obs["candidate_mask"]
        self.next_obs_context[idx] = next_obs["context"]

        self.terminated[idx] = terminated
        self.truncated[idx] = truncated

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions from the buffer.

        Returns
        -------
        obs_batch : Dict[str, torch.Tensor]
        actions : torch.Tensor (batch_size,)
        rewards : torch.Tensor (batch_size,)
        next_obs_batch : Dict[str, torch.Tensor]
        terminated : torch.Tensor (batch_size,) bool
        truncated : torch.Tensor (batch_size,) bool
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        indices = self.rng.choice(self.size, size=batch_size, replace=(batch_size > self.size))

        obs_batch = {
            "objective": torch.tensor(self.obs_objective[indices], dtype=torch.float32, device=device),
            "candidates": torch.tensor(self.obs_candidates[indices], dtype=torch.float32, device=device),
            "candidate_mask": torch.tensor(self.obs_candidate_mask[indices], dtype=torch.float32, device=device),
            "context": torch.tensor(self.obs_context[indices], dtype=torch.float32, device=device),
        }

        actions = torch.tensor(self.actions[indices], dtype=torch.int64, device=device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=device)

        next_obs_batch = {
            "objective": torch.tensor(self.next_obs_objective[indices], dtype=torch.float32, device=device),
            "candidates": torch.tensor(self.next_obs_candidates[indices], dtype=torch.float32, device=device),
            "candidate_mask": torch.tensor(self.next_obs_candidate_mask[indices], dtype=torch.float32, device=device),
            "context": torch.tensor(self.next_obs_context[indices], dtype=torch.float32, device=device),
        }

        terminated = torch.tensor(self.terminated[indices], dtype=torch.bool, device=device)
        truncated = torch.tensor(self.truncated[indices], dtype=torch.bool, device=device)

        return obs_batch, actions, rewards, next_obs_batch, terminated, truncated

    def __len__(self) -> int:
        return self.size
