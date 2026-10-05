"""
Unit tests for A2C Agent (rl.agents.a2c).
"""

import os
import random
import tempfile
import numpy as np
import pytest
import torch

from rl.agents.a2c import A2CAgent, A2CConfig, save_checkpoint, load_checkpoint
from rl.agents.a2c.rollout_buffer import A2CRolloutBuffer
from rl.state.encoder import STATE_ENCODING_VERSION
from rl.reward import REWARD_VERSION
from rl.curriculum.config import CURRICULUM_VERSION


def create_mock_obs():
    return {
        "objective": np.random.randn(64).astype(np.float32),
        "context": np.random.randn(30).astype(np.float32),
        "candidates": np.random.randn(20, 91).astype(np.float32),
        "candidate_mask": np.array([1.0]*4 + [0.0]*16, dtype=np.float32),
    }


def test_a2c_gae_does_not_cross_episode_boundaries():
    """
    VERIFY STEP 1 & 2:
    GAE advantage does NOT accumulate from next episode into a preceding terminated/truncated episode.
    """
    buf = A2CRolloutBuffer(capacity=10)

    # Episode 1: 3 transitions, ends with done=True (terminated)
    for _ in range(2):
        buf.add(create_mock_obs(), action=0, reward=1.0, value=0.5, done=False, truncated=False)
    buf.add(create_mock_obs(), action=0, reward=5.0, value=0.5, done=True, truncated=False)

    # Episode 2: 3 transitions, ends with truncated=True
    for _ in range(2):
        buf.add(create_mock_obs(), action=0, reward=1.0, value=0.5, done=False, truncated=False)
    buf.add(create_mock_obs(), action=0, reward=-3.0, value=0.5, done=False, truncated=True)

    buf.compute_returns_and_advantages(last_value=1.0, last_done=False, gamma=0.99, gae_lambda=0.95, normalize_advantages=False)

    # Transition 2 (end of Ep 1, done=True): delta_2 = 5.0 - 0.5 = 4.5. Advantage MUST be exactly 4.5
    assert buf.advantages[2] == pytest.approx(4.5, abs=1e-5)

    # Transition 5 (end of Ep 2, truncated=True, last_done=False): delta_5 = -3.0 + 0.99*1.0 - 0.5 = -2.51. Advantage MUST be exactly -2.51
    assert buf.advantages[5] == pytest.approx(-2.51, abs=1e-5)


def test_a2c_partial_rollout_flushing():
    """Verify partial A2C rollout is updated and cleared cleanly."""
    agent = A2CAgent(A2CConfig(rollout_steps=64))

    for _ in range(10):  # Add only 10 transitions (< capacity 64)
        obs = create_mock_obs()
        act, val, logp = agent.select_action(obs)
        agent.rollout_buffer.add(obs, act, reward=1.0, value=val, done=False, truncated=False)

    assert agent.rollout_buffer.ptr == 10
    diag = agent.update(last_value=0.5, last_done=True)

    assert "actor_loss" in diag
    assert agent.rollout_buffer.ptr == 0  # Buffer cleared after update


def test_a2c_rng_seeding_reproducibility():
    """Verify training seed produces reproducible initial weights and stochastic action sampling."""
    from rl.training.train_a2c import set_global_seeds

    set_global_seeds(2026)
    agent1 = A2CAgent()
    obs = create_mock_obs()
    act1, val1, logp1 = agent1.select_action(obs, eval_mode=False)

    set_global_seeds(2026)
    agent2 = A2CAgent()
    act2, val2, logp2 = agent2.select_action(obs, eval_mode=False)

    assert act1 == act2
    assert val1 == pytest.approx(val2, abs=1e-5)
    assert logp1 == pytest.approx(logp2, abs=1e-5)


def test_a2c_config_and_checkpoint_metadata_versions():
    """Verify config and checkpoint metadata use actual project version strings."""
    config = A2CConfig()
    assert config.state_encoding_version == STATE_ENCODING_VERSION
    assert config.reward_version == REWARD_VERSION
    assert config.curriculum_version == CURRICULUM_VERSION

    agent = A2CAgent(config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = os.path.join(tmp_dir, "ckpt.pt")
        save_checkpoint(chk_path, agent)

        meta = load_checkpoint(chk_path, agent)
        assert meta["state_encoding_version"] == STATE_ENCODING_VERSION
        assert meta["reward_version"] == REWARD_VERSION
