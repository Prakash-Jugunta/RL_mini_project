"""
Masked Double DQN Agent for UI Recovery Environment.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from rl.agents.dqn.config import DQNConfig, get_epsilon_at_step
from rl.agents.dqn.network import CandidateAwareQNetwork, apply_action_mask
from rl.agents.dqn.replay_buffer import DictReplayBuffer
from rl.agents.dqn.checkpoint import save_checkpoint, load_checkpoint

DQN_POLICY_VERSION = "phase10-dqn-state-v2"


class NoValidActionError(Exception):
    """Raised when an agent receives an observation where candidate_mask has zero valid candidates."""
    pass


class DQNAgent:
    """
    Candidate-Aware Masked Double DQN Agent.

    Contracts & Safety Guarantees:
      - Valid-only exploration: candidate_mask == 1 indices ONLY. Never selects padding.
      - Double DQN target: argmax_a Q_online(s', a) using NEXT-state mask, evaluated by Q_target(s', a*).
      - Termination / Truncation target: done = terminated or truncated -> target = reward (no bootstrap).
      - Huber Loss (smooth_l1_loss) with gradient norm clipping.
      - Completely leak-free: does not consume expected_role, semantic_role, or mutation metadata.
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        self.config = config or DQNConfig()
        self.version = DQN_POLICY_VERSION
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")

        # Explicit reproducible seeding
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        # Q-Networks
        self.online_network = CandidateAwareQNetwork(self.config).to(self.device)
        self.target_network = CandidateAwareQNetwork(self.config).to(self.device)

        # Initial hard update
        self.update_target_network()
        self.target_network.eval()
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate
        )

        self.replay_buffer = DictReplayBuffer(
            capacity=self.config.replay_capacity,
            objective_dim=self.config.objective_dim,
            max_candidates=self.config.max_candidates,
            candidate_feature_dim=self.config.candidate_feature_dim,
            context_dim=self.config.context_dim,
            seed=self.config.seed,
        )

        self.total_steps: int = 0
        self.learning_steps: int = 0

    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        explore: bool = True,
        **context: Any,
    ) -> int:
        """
        Selects an action following valid-only epsilon-greedy exploration.
        Never selects padding (mask == 0).
        """
        candidate_mask = observation["candidate_mask"]
        valid_indices = np.flatnonzero(candidate_mask == 1)

        if len(valid_indices) == 0:
            raise NoValidActionError("Observation contains zero valid candidate actions in candidate_mask.")

        eps = get_epsilon_at_step(self.config, self.total_steps) if explore else 0.0

        if explore and self.rng.random() < eps:
            action = int(self.rng.choice(valid_indices))
        else:
            obs_t = {
                "objective": torch.tensor(observation["objective"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "context": torch.tensor(observation["context"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "candidates": torch.tensor(observation["candidates"], dtype=torch.float32, device=self.device).unsqueeze(0),
            }
            mask_t = torch.tensor(candidate_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.online_network.eval()
            with torch.no_grad():
                raw_q = self.online_network(obs_t)
                masked_q = apply_action_mask(raw_q, mask_t)
                action = int(torch.argmax(masked_q, dim=1).item())

        # Defensive sanity check
        if candidate_mask[action] != 1:
            raise RuntimeError(f"DQN selected an invalid padded action index {action} (mask is 0).")

        return action

    def store_transition(
        self,
        obs: Dict[str, np.ndarray],
        action: int,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Stores a transition in the replay buffer and increments step counter."""
        self.replay_buffer.add(obs, action, reward, next_obs, terminated, truncated)
        self.total_steps += 1

    def can_learn(self) -> bool:
        """Returns True if the replay buffer contains enough transitions to begin learning."""
        return len(self.replay_buffer) >= self.config.learning_starts

    def learn(self) -> Dict[str, float]:
        """
        Performs one Double DQN learning step over a sampled batch.
        """
        if not self.can_learn():
            return {}

        self.learning_steps += 1
        self.online_network.train()

        obs, actions, rewards, next_obs, terminated, truncated = self.replay_buffer.sample(
            batch_size=self.config.batch_size,
            device=str(self.device),
        )

        # Target calculation (Double DQN)
        # Bootstrapping contract: done = terminated or truncated -> no bootstrap
        done = terminated | truncated

        with torch.no_grad():
            next_q_online = self.online_network(next_obs)
            masked_next_q_online = apply_action_mask(next_q_online, next_obs["candidate_mask"])
            next_actions = torch.argmax(masked_next_q_online, dim=1)  # (B,)

            next_q_target = self.target_network(next_obs)  # (B, 20)
            next_q_val = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)

            target_q = rewards + (1.0 - done.float()) * self.config.gamma * next_q_val

        # Online network Q-value for chosen action
        q_online = self.online_network(obs)  # (B, 20)
        q_acted = q_online.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Numerical stability checks
        if torch.isnan(q_acted).any() or torch.isinf(q_acted).any():
            raise RuntimeError("CRITICAL STABILITY ERROR: NaN or Inf detected in online Q-values.")

        # Loss calculation (Huber / Smooth L1 Loss)
        loss = F.smooth_l1_loss(q_acted, target_q.detach())

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("CRITICAL STABILITY ERROR: NaN or Inf detected in Double DQN loss.")

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm
        )
        self.optimizer.step()

        # Hard target network update
        if self.learning_steps % self.config.target_update_frequency == 0:
            self.update_target_network()

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_acted.mean().item()),
            "grad_norm": float(grad_norm.item()),
            "epsilon": get_epsilon_at_step(self.config, self.total_steps),
        }

    def update_target_network(self) -> None:
        """Copies online network weights to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_checkpoint(self, filepath: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
        save_checkpoint(filepath, self, extra_info=extra_info)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        return load_checkpoint(filepath, self)

    def set_eval_mode(self) -> None:
        self.online_network.eval()

    def set_train_mode(self) -> None:
        self.online_network.train()
