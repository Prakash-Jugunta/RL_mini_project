"""
Candidate-Aware Permutation-Equivariant Actor-Critic Network Architecture.
Shared by PPO and A2C agents.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoValidActionError(ValueError):
    """Raised when candidate_mask contains zero valid candidate actions."""
    pass


class ActorCriticOutput(NamedTuple):
    logits: torch.Tensor   # (B, 20) raw unmasked logits
    value: torch.Tensor    # (B, 1) state value estimate


class CandidateAwareActorCriticNetwork(nn.Module):
    """
    Candidate-Aware Actor-Critic Network for PPO and A2C.

    Actor Architecture (Permutation Equivariant):
      - Objective Encoder: 64 -> 128 -> 64
      - Context Encoder:   30 -> 64 -> 32
      - Shared Candidate Encoder: 91 -> 128 -> 64 (applied to each slot)
      - Slot Concat: 64 + 32 + 64 = 160 per candidate slot
      - Actor Scorer: 160 -> 128 -> 64 -> 1 (applied per candidate slot)
      - Output: (B, 20) candidate logits

    Critic Architecture (Permutation Invariant):
      - Masked Mean Pooling over valid candidate embeddings (B, 64)
      - State Concat: 64 (obj) + 32 (ctx) + 64 (pooled cand) = 160
      - Value Scorer: 160 -> 128 -> 64 -> 1
      - Output: (B, 1) scalar state value V(s)
    """

    def __init__(
        self,
        objective_dim: int = 64,
        context_dim: int = 30,
        candidate_feature_dim: int = 91,
        max_candidates: int = 20,
    ):
        super().__init__()

        self.objective_dim = objective_dim
        self.context_dim = context_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.max_candidates = max_candidates

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

        # 3. Shared Candidate Encoder: (B, 20, 91) -> (B, 20, 64)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 4. Actor Scorer (Shared across candidate slots): (B, 20, 160) -> (B, 20, 1)
        self.actor_scorer = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 5. Critic Value Scorer: (B, 160) -> (B, 1)
        # 64 (obj) + 32 (ctx) + 64 (pooled cand) = 160
        self.critic_scorer = nn.Sequential(
            nn.Linear(160, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_parameter_count_breakdown(self) -> Dict[str, int]:
        """
        Programmatically computes parameter counts, distinguishing shared encoders from actor/critic heads.
        """
        shared_obj = sum(p.numel() for p in self.objective_encoder.parameters() if p.requires_grad)
        shared_ctx = sum(p.numel() for p in self.context_encoder.parameters() if p.requires_grad)
        shared_cand = sum(p.numel() for p in self.candidate_encoder.parameters() if p.requires_grad)
        actor_head = sum(p.numel() for p in self.actor_scorer.parameters() if p.requires_grad)
        critic_head = sum(p.numel() for p in self.critic_scorer.parameters() if p.requires_grad)

        shared_total = shared_obj + shared_ctx + shared_cand
        unique_total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "shared_encoders": shared_total,
            "objective_encoder": shared_obj,
            "context_encoder": shared_ctx,
            "candidate_encoder": shared_cand,
            "actor_head": actor_head,
            "critic_head": critic_head,
            "total_unique_trainable_parameters": unique_total,
        }

    def _encode_submodules(
        self, observation: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared encoding submodules.
        Returns:
          obj_emb: (B, 64)
          ctx_emb: (B, 32)
          cand_emb: (B, 20, 64)
        """
        objective = observation["objective"]
        context = observation["context"]
        candidates = observation["candidates"]

        obj_emb = self.objective_encoder(objective)   # (B, 64)
        ctx_emb = self.context_encoder(context)       # (B, 32)
        cand_emb = self.candidate_encoder(candidates) # (B, 20, 64)

        return obj_emb, ctx_emb, cand_emb

    def forward_actor(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass computing raw candidate logits.
        Returns:
          raw_logits: (B, 20)
        """
        obj_emb, ctx_emb, cand_emb = self._encode_submodules(observation)
        batch_size = obj_emb.shape[0]

        obj_emb_exp = obj_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 64)
        ctx_emb_exp = ctx_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 32)

        # Slot features: (B, 20, 160)
        slot_features = torch.cat([obj_emb_exp, ctx_emb_exp, cand_emb], dim=-1)

        # Score actor logits: (B, 20, 1) -> (B, 20)
        logits = self.actor_scorer(slot_features).squeeze(-1)
        return logits

    def forward_critic(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass computing state value V(s).
        Uses masked mean pooling over valid candidate embeddings for permutation invariance.
        Returns:
          value: (B, 1)
        """
        obj_emb, ctx_emb, cand_emb = self._encode_submodules(observation)
        batch_size = obj_emb.shape[0]

        mask = observation.get("candidate_mask", None)
        if mask is None:
            mask = torch.ones((batch_size, self.max_candidates), device=cand_emb.device, dtype=cand_emb.dtype)
        else:
            mask = mask.to(dtype=cand_emb.dtype)

        # Mask shape: (B, 20, 1)
        mask_exp = mask.unsqueeze(-1)

        # Masked candidate embeddings
        masked_cand = cand_emb * mask_exp  # (B, 20, 64)

        # Sum along candidates dimension: (B, 64)
        sum_cand = masked_cand.sum(dim=1)

        # Valid candidate counts per batch item: (B, 1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Masked mean pooling: (B, 64)
        pooled_cand = sum_cand / valid_counts

        # Concat state features: 64 + 32 + 64 = 160
        state_features = torch.cat([obj_emb, ctx_emb, pooled_cand], dim=-1)

        # Compute scalar state value V(s): (B, 1)
        value = self.critic_scorer(state_features)
        return value

    def forward(self, observation: Dict[str, torch.Tensor]) -> ActorCriticOutput:
        """
        Simultaneous forward pass for both actor logits and critic value.
        """
        obj_emb, ctx_emb, cand_emb = self._encode_submodules(observation)
        batch_size = obj_emb.shape[0]

        # 1. Actor logits
        obj_emb_exp = obj_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 64)
        ctx_emb_exp = ctx_emb.unsqueeze(1).expand(batch_size, self.max_candidates, 32)
        slot_features = torch.cat([obj_emb_exp, ctx_emb_exp, cand_emb], dim=-1)
        logits = self.actor_scorer(slot_features).squeeze(-1)

        # 2. Critic value
        mask = observation.get("candidate_mask", None)
        if mask is None:
            mask = torch.ones((batch_size, self.max_candidates), device=cand_emb.device, dtype=cand_emb.dtype)
        else:
            mask = mask.to(dtype=cand_emb.dtype)

        mask_exp = mask.unsqueeze(-1)
        masked_cand = cand_emb * mask_exp
        sum_cand = masked_cand.sum(dim=1)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_cand = sum_cand / valid_counts

        state_features = torch.cat([obj_emb, ctx_emb, pooled_cand], dim=-1)
        value = self.critic_scorer(state_features)

        return ActorCriticOutput(logits=logits, value=value)


def apply_action_mask(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """
    Applies candidate validity mask to raw actor logits.
    Padded/invalid candidates (mask <= 0) receive mask_value (-1e9).
    Raises NoValidActionError if any batch item has zero valid actions.

    Parameters
    ----------
    logits : torch.Tensor
        Raw unmasked actor logits (B, 20).
    candidate_mask : torch.Tensor
        Candidate mask (B, 20) with 1 for valid, 0 for padded.
    mask_value : float
        Large negative float value to effectively set invalid logit probabilities to 0.

    Returns
    -------
    masked_logits : torch.Tensor
        Tensor of shape (B, 20).
    """
    valid_counts = (candidate_mask > 0).sum(dim=-1)
    if (valid_counts <= 0).any():
        raise NoValidActionError("No valid candidate actions available in candidate_mask!")

    mask_bool = candidate_mask <= 0
    return logits.masked_fill(mask_bool, mask_value)
