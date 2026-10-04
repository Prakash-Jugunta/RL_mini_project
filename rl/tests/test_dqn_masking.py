"""
Unit tests for Valid-Only Action Masking and Exploration.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np
import torch

from rl.agents.dqn.config import DQNConfig
from rl.agents.dqn.network import apply_action_mask
from rl.agents.dqn.agent import DQNAgent, NoValidActionError


def test_apply_action_mask_excludes_padding():
    q_values = torch.tensor([[1.2, 2.5, 1000.0, 3.8] + [0.0]*16])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0] + [0.0]*16])

    masked_q = apply_action_mask(q_values, mask)

    # Index 2 has mask=0, so its Q-value must be -1e9
    assert masked_q[0, 2].item() == -1e9

    # Greedy action on masked Q must pick index 3 (3.8 > 2.5 > 1.2)
    best_action = int(torch.argmax(masked_q, dim=1).item())
    assert best_action == 3, f"Greedy action selected padded/suboptimal index {best_action}"


def test_agent_exploration_never_selects_padding():
    config = DQNConfig(epsilon_start=1.0, epsilon_end=1.0, seed=42)
    agent = DQNAgent(config)

    # Mask where only indices 2 and 5 are valid
    mask = np.zeros(20, dtype=np.float32)
    mask[2] = 1.0
    mask[5] = 1.0

    dummy_obs = {
        "objective": np.zeros(64, dtype=np.float32),
        "context": np.zeros(30, dtype=np.float32),
        "candidates": np.zeros((20, 91), dtype=np.float32),
        "candidate_mask": mask,
    }

    selected_counts = {2: 0, 5: 0}
    n_samples = 500

    for _ in range(n_samples):
        act = agent.select_action(dummy_obs, explore=True)
        assert mask[act] == 1.0, f"Random exploration selected padded action index {act}!"
        selected_counts[act] += 1

    # Verify both valid actions were sampled roughly uniformly
    for idx, count in selected_counts.items():
        freq = count / n_samples
        assert 0.35 <= freq <= 0.65, f"Index {idx} frequency {freq:.3f} outside expected uniform range"


def test_empty_mask_raises_no_valid_action_error():
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)

    empty_obs = {
        "objective": np.zeros(64, dtype=np.float32),
        "context": np.zeros(30, dtype=np.float32),
        "candidates": np.zeros((20, 91), dtype=np.float32),
        "candidate_mask": np.zeros(20, dtype=np.float32),
    }

    with pytest.raises(NoValidActionError):
        agent.select_action(empty_obs, explore=True)

    with pytest.raises(NoValidActionError):
        agent.select_action(empty_obs, explore=False)
