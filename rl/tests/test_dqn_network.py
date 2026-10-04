"""
Unit tests for CandidateAwareQNetwork architecture and Permutation Equivariance.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import numpy as np

from rl.agents.dqn.config import DQNConfig
from rl.agents.dqn.network import CandidateAwareQNetwork, apply_action_mask


def test_network_output_shape_and_finiteness():
    config = DQNConfig()
    net = CandidateAwareQNetwork(config)
    net.eval()

    for batch_size in [1, 4, 16]:
        obs = {
            "objective": torch.randn(batch_size, 64),
            "context": torch.randn(batch_size, 30),
            "candidates": torch.randn(batch_size, 20, 91),
        }
        with torch.no_grad():
            q_values = net(obs)

        assert q_values.shape == (batch_size, 20)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()


def test_candidate_permutation_equivariance():
    """
    CRITICAL ARCHITECTURAL TEST:
    Permuting candidate slot ordering in input MUST permute Q-value output identically.
    This proves the network scores candidate content rather than slot identity.
    """
    config = DQNConfig()
    net = CandidateAwareQNetwork(config)
    net.eval()

    torch.manual_seed(42)
    obs_orig = {
        "objective": torch.randn(1, 64),
        "context": torch.randn(1, 30),
        "candidates": torch.randn(1, 20, 91),
    }

    # Define a random permutation of indices 0..19
    perm = torch.randperm(20)

    # Permute candidate slot dimension (dim 1)
    obs_perm = {
        "objective": obs_orig["objective"].clone(),
        "context": obs_orig["context"].clone(),
        "candidates": obs_orig["candidates"][:, perm, :].clone(),
    }

    with torch.no_grad():
        q_orig = net(obs_orig)  # (1, 20)
        q_perm = net(obs_perm)  # (1, 20)

    # Expected: q_perm should equal q_orig[:, perm]
    q_orig_permuted = q_orig[:, perm]

    diff = torch.abs(q_perm - q_orig_permuted).max().item()
    assert diff < 1e-5, f"Candidate permutation equivariance failed! Max diff = {diff}"
