"""
Unit tests for DQNAgent lifecycle, isolation, checkpointing, and counterfactual safety.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np
import torch

from rl.agents.dqn.config import DQNConfig
from rl.agents.dqn.agent import DQNAgent


def make_dummy_obs(val: float = 1.0) -> dict:
    return {
        "objective": np.full(64, val, dtype=np.float32),
        "context": np.full(30, val, dtype=np.float32),
        "candidates": np.full((20, 91), val, dtype=np.float32),
        "candidate_mask": np.ones(20, dtype=np.float32),
    }


def test_target_network_isolation():
    config = DQNConfig(seed=42, learning_starts=5, target_update_frequency=100)
    agent = DQNAgent(config)

    obs = make_dummy_obs()

    for _ in range(10):
        agent.store_transition(obs, action=0, reward=1.0, next_obs=obs, terminated=False, truncated=False)

    # Initial target weights match online weights
    online_p_before = [p.clone() for p in agent.online_network.parameters()]
    target_p_before = [p.clone() for p in agent.target_network.parameters()]

    # Perform one learn step
    agent.learn()

    online_p_after = list(agent.online_network.parameters())
    target_p_after = list(agent.target_network.parameters())

    # Online network parameters must change
    online_changed = any(not torch.equal(b, a) for b, a in zip(online_p_before, online_p_after))
    assert online_changed, "Online network parameters did not update after learn()!"

    # Target network parameters must remain UNCHANGED (isolation)
    target_unchanged = all(torch.equal(b, a) for b, a in zip(target_p_before, target_p_after))
    assert target_unchanged, "Target network parameters changed before target_update_frequency!"

    # Explicit update
    agent.update_target_network()
    target_p_updated = list(agent.target_network.parameters())
    target_matches_online = all(torch.equal(o, t) for o, t in zip(online_p_after, target_p_updated))
    assert target_matches_online, "Target network weights do not match online network after explicit update!"


def test_checkpoint_round_trip(tmp_path):
    config = DQNConfig(seed=42)
    agent_a = DQNAgent(config)

    obs = make_dummy_obs()
    obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items() if k != "candidate_mask"}

    agent_a.online_network.eval()
    with torch.no_grad():
        q_a = agent_a.online_network(obs_t)

    checkpoint_path = os.path.join(tmp_path, "dqn_test.pt")
    agent_a.save_checkpoint(checkpoint_path)

    agent_b = DQNAgent(config)
    agent_b.load_checkpoint(checkpoint_path)
    agent_b.online_network.eval()

    with torch.no_grad():
        q_b = agent_b.online_network(obs_t)

    diff = torch.abs(q_a - q_b).max().item()
    assert diff < 1e-6, f"Checkpoint round trip mismatch! Max diff = {diff}"


def test_private_and_mutation_metadata_counterfactual_safety():
    """
    LEAKAGE & COUNTERFACTUAL AUDIT:
    Verifies that changing evaluator expected_role or mutation metadata
    does not alter agent Q-values or selected action.
    """
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)
    agent.set_eval_mode()

    obs = make_dummy_obs()

    act1 = agent.select_action(obs, explore=False, expected_role="username-input", mutation_level=1, mutation_seed=11)
    act2 = agent.select_action(obs, explore=False, expected_role="password-input", mutation_level=6, mutation_seed=99)

    assert act1 == act2, "Agent action selection changed based on private metadata context!"
