"""
A2C Agent Implementation.
Candidate-aware synchronous Advantage Actor-Critic agent.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple, Optional

from rl.agents.a2c.config import A2CConfig
from rl.agents.common.actor_critic_network import (
    CandidateAwareActorCriticNetwork,
    apply_action_mask,
    NoValidActionError,
)
from rl.agents.a2c.rollout_buffer import A2CRolloutBuffer


class A2CAgent:
    """
    Candidate-Aware Synchronous Advantage Actor-Critic (A2C) Agent.
    """

    def __init__(self, config: Optional[A2CConfig] = None):
        self.config = config or A2CConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

        self.network = CandidateAwareActorCriticNetwork(
            objective_dim=self.config.objective_dim,
            context_dim=self.config.context_dim,
            candidate_feature_dim=self.config.candidate_feature_dim,
            max_candidates=self.config.max_candidates,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        self.rollout_buffer = A2CRolloutBuffer(
            capacity=self.config.rollout_steps,
            objective_dim=self.config.objective_dim,
            context_dim=self.config.context_dim,
            candidate_feature_dim=self.config.candidate_feature_dim,
            max_candidates=self.config.max_candidates,
            device=str(self.device),
        )

        self.global_step = 0
        self.update_count = 0

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
        Performs synchronous A2C update over collected rollout buffer transitions.
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

        rollout_data = self.rollout_buffer.get_rollout_data()
        obs = rollout_data.observations
        actions = rollout_data.actions
        returns = rollout_data.returns
        advantages = rollout_data.advantages

        out = self.network(obs)
        masked_logits = apply_action_mask(out.logits, obs["candidate_mask"])
        dist = Categorical(logits=masked_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(out.value.squeeze(-1), returns)

        total_loss = (
            actor_loss
            + self.config.value_loss_coef * critic_loss
            - self.config.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.update_count += 1
        self.rollout_buffer.reset()

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std().item()),
        }
