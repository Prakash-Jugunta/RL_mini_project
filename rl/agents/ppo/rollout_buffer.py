"""
On-Policy Rollout Buffer for PPO Agent.
Stores transitions, computes Generalized Advantage Estimation (GAE), and provides minibatches.
"""

from typing import Dict, Generator, List, NamedTuple, Tuple, Optional
import numpy as np
import torch


class PPOBatch(NamedTuple):
    observations: Dict[str, torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class PPORolloutBuffer:
    """
    On-Policy Rollout Buffer for PPO.
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
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.bool_)
        self.truncations = np.zeros(self.capacity, dtype=np.bool_)

        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        truncated: bool = False,
    ) -> None:
        if self.ptr >= self.capacity:
            raise IndexError("PPORolloutBuffer is full! Call compute_returns_and_advantages and reset before adding more transitions.")

        self.objectives[self.ptr] = obs["objective"]
        self.contexts[self.ptr] = obs["context"]
        self.candidates[self.ptr] = obs["candidates"]
        self.candidate_masks[self.ptr] = obs["candidate_mask"]

        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.truncations[self.ptr] = truncated

        self.ptr += 1
        if self.ptr == self.capacity:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> None:
        """
        Computes GAE advantages and TD returns.
        Fixes episode boundary handling so advantage does NOT leak across episode boundaries.

        For transition t:
          - term_mask = 1.0 - float(dones[t]) (controls V(s_{t+1}) bootstrap for true termination)
          - cont_mask = 1.0 - float(dones[t] or truncations[t]) (controls GAE accumulation across episodes)
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

    def get_batches(
        self,
        minibatch_size: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Generator[PPOBatch, None, None]:
        """
        Yields randomized minibatches for PPO optimization epochs.
        Optionally accepts numpy RandomState for reproducible shuffling.
        """
        n_samples = self.ptr
        indices = np.arange(n_samples)
        if rng is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        for start_idx in range(0, n_samples, minibatch_size):
            batch_indices = indices[start_idx : start_idx + minibatch_size]

            obs_batch = {
                "objective": torch.tensor(self.objectives[batch_indices], dtype=torch.float32, device=self.device),
                "context": torch.tensor(self.contexts[batch_indices], dtype=torch.float32, device=self.device),
                "candidates": torch.tensor(self.candidates[batch_indices], dtype=torch.float32, device=self.device),
                "candidate_mask": torch.tensor(self.candidate_masks[batch_indices], dtype=torch.float32, device=self.device),
            }

            yield PPOBatch(
                observations=obs_batch,
                actions=torch.tensor(self.actions[batch_indices], dtype=torch.int64, device=self.device),
                old_log_probs=torch.tensor(self.log_probs[batch_indices], dtype=torch.float32, device=self.device),
                old_values=torch.tensor(self.values[batch_indices], dtype=torch.float32, device=self.device),
                returns=torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=self.device),
                advantages=torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=self.device),
            )
