"""
Candidate-Aware Permutation-Equivariant Q-Network Architecture.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from typing import Dict, Union, Optional

from rl.agents.dqn.config import DQNConfig


class CandidateAwareQNetwork(nn.Module):
    """
    Candidate-Aware Q-Network.
    Calculates Q(s, candidate_i) for each candidate slot i in [0, max_candidates-1].

    Network Architecture:
      - Objective Encoder: 64 -> 128 -> 64
      - Context Encoder:   30 -> 64 -> 32 (public previous-candidate features)
      - Candidate Encoder: 91 -> 128 -> 64 (SHARED across all 20 candidate slots)
      - Concatenated Representation: 64 + 32 + 64 = 160 per candidate slot
      - Q Scorer:          160 -> 128 -> 64 -> 1 (SHARED across all candidate slots)

    Output shape: (B, 20) raw Q-values.
    """

    def __init__(self, config: Optional[DQNConfig] = None):
        super().__init__()
        cfg = config or DQNConfig()

        self.objective_dim = cfg.objective_dim
        self.context_dim = cfg.context_dim
        self.candidate_feature_dim = cfg.candidate_feature_dim
        self.max_candidates = cfg.max_candidates

        # 1. Objective Encoder: (B, 64) -> (B, 64)
        self.objective_encoder = nn.Sequential(
            nn.Linear(self.objective_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 2. Context Encoder: (B, 30) -> (B, 32)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 3. Candidate Encoder (Shared across candidate slots): (B, 20, 91) -> (B, 20, 64)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 4. Q-Value Scorer (Shared across candidate slots): (B, 20, 160) -> (B, 20, 1)
        # 64 (obj) + 32 (ctx) + 64 (cand) = 160
        self.q_scorer = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass computing raw Q-values for all candidate slots.

        Parameters
        ----------
        observation : Dict[str, torch.Tensor]
            Dictionary containing:
            - 'objective': (B, 64)
            - 'context': (B, 30)
            - 'candidates': (B, 20, 91)

        Returns
        -------
        raw_q_values : torch.Tensor
            Tensor of shape (B, 20) containing raw Q-values.
        """
        objective = observation["objective"]
        context = observation["context"]
        candidates = observation["candidates"]

        batch_size = objective.shape[0]

        # 1. Encode objective and context
        obj_emb = self.objective_encoder(objective)  # (B, 64)
        ctx_emb = self.context_encoder(context)      # (B, 32)

        # 2. Encode candidate features (shared across slots)
        cand_emb = self.candidate_encoder(candidates)  # (B, 20, 64)

        # 3. Expand objective and context embeddings across candidate dimension
        obj_emb_exp = obj_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 64)  # (B, 20, 64)
        ctx_emb_exp = ctx_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 32)  # (B, 20, 32)

        # 4. Concatenate representations: (B, 20, 64 + 32 + 64 = 160)
        slot_features = torch.cat([obj_emb_exp, ctx_emb_exp, cand_emb], dim=-1)

        # 5. Score Q-values per slot
        q_values_slot = self.q_scorer(slot_features)  # (B, 20, 1)

        # Squeeze last dimension to produce (B, 20)
        raw_q_values = q_values_slot.squeeze(-1)
        return raw_q_values


def apply_action_mask(
    q_values: torch.Tensor,
    candidate_mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Applies candidate validity mask to raw Q-values.
    Padded/invalid candidates (mask <= 0) receive mask_value (-1e9).

    Parameters
    ----------
    q_values : torch.Tensor
        Tensor of shape (B, 20).
    candidate_mask : torch.Tensor
        Tensor of shape (B, 20) with 1 for valid, 0 for padded.

    Returns
    -------
    masked_q_values : torch.Tensor
        Tensor of shape (B, 20) with invalid positions set to mask_value.
    """
    mask_bool = candidate_mask <= 0
    return q_values.masked_fill(mask_bool, mask_value)
