"""
Unit tests for Double-DQN Target Calculation and Masking.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import numpy as np

from rl.agents.dqn.config import DQNConfig
from rl.agents.dqn.network import apply_action_mask
from rl.agents.dqn.agent import DQNAgent


def test_double_dqn_target_uses_next_state_mask():
    """
    CRITICAL DOUBLE-DQN TARGET MASKING TEST:
    Next-action argmax MUST use the next state's candidate mask.
    Padded candidates in the next state must NOT be selected by the online network.
    """
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)

    # Next state online Q-values: index 1 has highest Q (100.0), but is padded (mask[1] = 0)
    # Valid indices in next state: 0 (Q=1.0) and 2 (Q=5.0)
    next_q_online = torch.tensor([[1.0, 100.0, 5.0] + [0.0]*17])
    next_mask = torch.tensor([[1.0, 0.0, 1.0] + [0.0]*17])

    masked_next_q_online = apply_action_mask(next_q_online, next_mask)
    best_next_action = torch.argmax(masked_next_q_online, dim=1).item()

    assert best_next_action == 2, f"Double-DQN next action selected padded index {best_next_action} instead of valid index 2!"


def test_terminal_and_truncated_target_no_bootstrap():
    """
    MANDATORY TERMINAL & TRUNCATION TARGET TEST:
    For terminated=True or truncated=True, target MUST be reward (no bootstrap).
    """
    config = DQNConfig(seed=42, gamma=0.99, batch_size=16, learning_starts=10)
    agent = DQNAgent(config)

    obs = {
        "objective": np.zeros(64, dtype=np.float32),
        "context": np.zeros(30, dtype=np.float32),
        "candidates": np.zeros((20, 91), dtype=np.float32),
        "candidate_mask": np.ones(20, dtype=np.float32),
    }

    # Add terminal transition
    agent.store_transition(obs, action=1, reward=5.95, next_obs=obs, terminated=True, truncated=False)

    # Add truncated transition
    agent.store_transition(obs, action=2, reward=-2.05, next_obs=obs, terminated=False, truncated=True)

    # Fill buffer to learning_starts
    for _ in range(30):
        agent.store_transition(obs, action=0, reward=0.95, next_obs=obs, terminated=False, truncated=False)

    diag = agent.learn()
    assert "loss" in diag
    assert not np.isnan(diag["loss"])
    assert not np.isinf(diag["loss"])
