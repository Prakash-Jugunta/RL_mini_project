"""
On-Policy Rollout Buffer for A2C Agent.
Stores transitions and computes advantage returns for synchronous Advantage Actor-Critic update.
"""

from typing import Dict, NamedTuple
import numpy as np
import torch


class A2CRolloutData(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class A2CRolloutBuffer:
    """
    On-Policy Rollout Buffer for A2C.
    """

    def __init__(
        self,
        capacity: int,
        objective_dim: int = 64,
        context_dim: int = 30,
        candidate_feature_dim: int = 91,
        max_candidates: int = 20,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.objective_dim = objective_dim
        self.context_dim = context_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.max_candidates = max_candidates
        self.device = device

        self.reset()

    def reset(self) -> None:
        self.objectives = np.zeros((self.capacity, self.objective_dim), dtype=np.float32)
        self.contexts = np.zeros((self.capacity, self.context_dim), dtype=np.float32)
        self.candidates = np.zeros((self.capacity, self.max_candidates, self.candidate_feature_dim), dtype=np.float32)
        self.candidate_masks = np.zeros((self.capacity, self.max_candidates), dtype=np.float32)

        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.truncations = np.zeros(self.capacity, dtype=np.bool_)

        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        value: float,
        done: bool,
        truncated: bool = False,
    ) -> None:
        if self.ptr >= self.capacity:
            raise IndexError("A2CRolloutBuffer is full! Reset before adding more transitions.")

        self.objectives[self.ptr] = obs["objective"]
        self.contexts[self.ptr] = obs["context"]
        self.candidates[self.ptr] = obs["candidates"]
        self.candidate_masks[self.ptr] = obs["candidate_mask"]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.truncations[self.ptr] = truncated

        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> None:
        """
        Computes n-step / GAE returns and advantages.
        Fixes episode boundary handling so advantage does NOT leak across episode boundaries.
        """
        n_samples = self.ptr
        last_gae_lam = 0.0

        for t in reversed(range(n_samples)):
            if t == n_samples - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_values = last_value
                delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
                last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
            else:
                term_mask = 1.0 - float(self.dones[t])
                cont_mask = 1.0 - float(self.dones[t] or self.truncations[t])
                next_values = self.values[t + 1]

                delta = self.rewards[t] + gamma * next_values * term_mask - self.values[t]
                last_gae_lam = delta + gamma * gae_lambda * cont_mask * last_gae_lam

            self.advantages[t] = last_gae_lam

        self.returns[:n_samples] = self.advantages[:n_samples] + self.values[:n_samples]

        if normalize_advantages and n_samples > 1:
            adv_slice = self.advantages[:n_samples]
            mean_adv = np.mean(adv_slice)
            std_adv = np.std(adv_slice) + 1e-8
            self.advantages[:n_samples] = (adv_slice - mean_adv) / std_adv

    def get_rollout_data(self) -> A2CRolloutData:
        """
        Returns stored transitions as PyTorch tensors for A2C update.
        """
        n_samples = self.ptr
        obs_batch = {
            "objective": torch.tensor(self.objectives[:n_samples], dtype=torch.float32, device=self.device),
            "context": torch.tensor(self.contexts[:n_samples], dtype=torch.float32, device=self.device),
            "candidates": torch.tensor(self.candidates[:n_samples], dtype=torch.float32, device=self.device),
            "candidate_mask": torch.tensor(self.candidate_masks[:n_samples], dtype=torch.float32, device=self.device),
        }

        return A2CRolloutData(
            observations=obs_batch,
            actions=torch.tensor(self.actions[:n_samples], dtype=torch.int64, device=self.device),
            returns=torch.tensor(self.returns[:n_samples], dtype=torch.float32, device=self.device),
            advantages=torch.tensor(self.advantages[:n_samples], dtype=torch.float32, device=self.device),
        )
