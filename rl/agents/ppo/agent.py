"""
PPO Agent Implementation.
Candidate-aware, permutation-equivariant actor-critic with clipped surrogate objective and GAE.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple, Optional

from rl.agents.ppo.config import PPOConfig
from rl.agents.common.actor_critic_network import (
    CandidateAwareActorCriticNetwork,
    apply_action_mask,
    NoValidActionError,
)
from rl.agents.ppo.rollout_buffer import PPORolloutBuffer


class PPOAgent:
    """
    Candidate-Aware PPO Agent.
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

        self.network = CandidateAwareActorCriticNetwork(
            objective_dim=self.config.objective_dim,
            context_dim=self.config.context_dim,
            candidate_feature_dim=self.config.candidate_feature_dim,
            max_candidates=self.config.max_candidates,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        self.rollout_buffer = PPORolloutBuffer(
            capacity=self.config.rollout_steps,
            objective_dim=self.config.objective_dim,
            context_dim=self.config.context_dim,
            candidate_feature_dim=self.config.candidate_feature_dim,
            max_candidates=self.config.max_candidates,
            device=str(self.device),
        )

        self.global_step = 0
        self.update_count = 0
        self.rng = np.random.RandomState(42)

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducible sampling and minibatch shuffling."""
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        eval_mode: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action from observation using policy distribution (or greedy argmax in eval_mode).

        Parameters
        ----------
        observation : Dict[str, np.ndarray]
            Environment observation dict.
        eval_mode : bool
            If True, select greedy deterministic candidate action argmax(masked_logits).
            If False, sample from Categorical distribution.

        Returns
        -------
        action : int
            Selected candidate index (0 to 19).
        value : float
            Predicted state value V(s).
        log_prob : float
            Log probability of selected action.
        """
        self.network.eval()
        with torch.no_grad():
            obs_tensor = {
                "objective": torch.tensor(observation["objective"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "context": torch.tensor(observation["context"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "candidates": torch.tensor(observation["candidates"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "candidate_mask": torch.tensor(observation["candidate_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            }

            actor_critic_out = self.network(obs_tensor)
            raw_logits = actor_critic_out.logits
            value = actor_critic_out.value.item()

            masked_logits = apply_action_mask(raw_logits, obs_tensor["candidate_mask"])
            dist = Categorical(logits=masked_logits)

            if eval_mode:
                action_tensor = torch.argmax(masked_logits, dim=-1)
            else:
                action_tensor = dist.sample()

            action = int(action_tensor.item())
            log_prob = float(dist.log_prob(action_tensor).item())

        return action, value, log_prob

    def update(self, last_value: float = 0.0, last_done: bool = True) -> Dict[str, float]:
        """
        Performs PPO clipped optimization over collected rollout buffer transitions.
        """
        if self.rollout_buffer.ptr == 0:
            return {}

        self.network.train()
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=True,
        )

        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []

        for _ in range(self.config.update_epochs):
            for batch in self.rollout_buffer.get_batches(self.config.minibatch_size, rng=self.rng):
                out = self.network(batch.observations)
                masked_logits = apply_action_mask(out.logits, batch.observations["candidate_mask"])
                dist = Categorical(logits=masked_logits)

                new_log_probs = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()

                log_ratios = new_log_probs - batch.old_log_probs
                ratios = torch.exp(log_ratios)

                # Clipped surrogate objective
                surr1 = ratios * batch.advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = out.value.squeeze(-1)
                value_loss = F.mse_loss(new_values, batch.returns)

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Logging metrics
                with torch.no_grad():
                    approx_kl = ((ratios - 1.0) - log_ratios).mean().item()
                    clip_frac = (torch.abs(ratios - 1.0) > self.config.clip_epsilon).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_frac)
                grad_norms.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))

        self.update_count += 1
        self.rollout_buffer.reset()

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "grad_norm": float(np.mean(grad_norms)),
        }
