"""
Unit tests for PPO Agent (rl.agents.ppo).
"""

import os
import random
import tempfile
import numpy as np
import pytest
import torch

from rl.agents.ppo import PPOAgent, PPOConfig, save_checkpoint, load_checkpoint
from rl.agents.ppo.rollout_buffer import PPORolloutBuffer
from rl.state.encoder import STATE_ENCODING_VERSION
from rl.reward import REWARD_VERSION
from rl.curriculum.config import CURRICULUM_VERSION


def create_mock_obs():
    return {
        "objective": np.random.randn(64).astype(np.float32),
        "context": np.random.randn(30).astype(np.float32),
        "candidates": np.random.randn(20, 91).astype(np.float32),
        "candidate_mask": np.array([1.0]*5 + [0.0]*15, dtype=np.float32),
    }


def test_ppo_gae_does_not_cross_episode_boundaries():
    """
    VERIFY STEP 1 & 2:
    GAE advantage does NOT accumulate from next episode into a preceding terminated/truncated episode.
    """
    buf = PPORolloutBuffer(capacity=10)
    
    # Episode 1: 3 transitions, ends with done=True (terminated)
    for _ in range(2):
        buf.add(create_mock_obs(), action=0, reward=1.0, value=0.5, log_prob=-0.5, done=False, truncated=False)
    buf.add(create_mock_obs(), action=0, reward=5.0, value=0.5, log_prob=-0.5, done=True, truncated=False)

    # Episode 2: 3 transitions, ends with truncated=True
    for _ in range(2):
        buf.add(create_mock_obs(), action=0, reward=1.0, value=0.5, log_prob=-0.5, done=False, truncated=False)
    buf.add(create_mock_obs(), action=0, reward=-3.0, value=0.5, log_prob=-0.5, done=False, truncated=True)

    buf.compute_returns_and_advantages(last_value=1.0, last_done=False, gamma=0.99, gae_lambda=0.95, normalize_advantages=False)

    # Transition 2 (end of Ep 1, done=True): delta_2 = 5.0 - 0.5 = 4.5. Advantage MUST be exactly 4.5 (no future GAE term)
    assert buf.advantages[2] == pytest.approx(4.5, abs=1e-5)

    # Transition 5 (end of Ep 2, truncated=True, last_done=False): delta_5 = -3.0 + 0.99*1.0 - 0.5 = -2.51. Advantage MUST be exactly -2.51
    assert buf.advantages[5] == pytest.approx(-2.51, abs=1e-5)


def test_ppo_partial_rollout_flushing():
    """Verify partial PPO rollout is updated and cleared cleanly."""
    agent = PPOAgent(PPOConfig(rollout_steps=64, minibatch_size=8, update_epochs=2))

    for _ in range(10):  # Add only 10 transitions (< capacity 64)
        obs = create_mock_obs()
        act, val, logp = agent.select_action(obs)
        agent.rollout_buffer.add(obs, act, reward=1.0, value=val, log_prob=logp, done=False, truncated=False)

    assert agent.rollout_buffer.ptr == 10
    diag = agent.update(last_value=0.5, last_done=True)

    assert "policy_loss" in diag
    assert agent.rollout_buffer.ptr == 0  # Buffer cleared after update


def test_ppo_rng_seeding_reproducibility():
    """Verify training seed produces reproducible initial weights and stochastic action sampling."""
    from rl.training.train_ppo import set_global_seeds

    set_global_seeds(2026)
    agent1 = PPOAgent()
    agent1.set_seed(2026)
    obs = create_mock_obs()
    act1, val1, logp1 = agent1.select_action(obs, eval_mode=False)

    set_global_seeds(2026)
    agent2 = PPOAgent()
    agent2.set_seed(2026)
    act2, val2, logp2 = agent2.select_action(obs, eval_mode=False)

    assert act1 == act2
    assert val1 == pytest.approx(val2, abs=1e-5)
    assert logp1 == pytest.approx(logp2, abs=1e-5)

    set_global_seeds(9999)
    agent3 = PPOAgent()
    agent3.set_seed(9999)
    act3, _, _ = agent3.select_action(obs, eval_mode=False)
    # Different seed gives different initialization or sampling
    weights1 = next(agent1.network.parameters())
    weights3 = next(agent3.network.parameters())
    assert not torch.equal(weights1, weights3)


def test_ppo_config_and_checkpoint_metadata_versions():
    """Verify config and checkpoint metadata use actual project version strings."""
    config = PPOConfig()
    assert config.state_encoding_version == STATE_ENCODING_VERSION
    assert config.reward_version == REWARD_VERSION
    assert config.curriculum_version == CURRICULUM_VERSION

    agent = PPOAgent(config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = os.path.join(tmp_dir, "ckpt.pt")
        save_checkpoint(chk_path, agent)

        meta = load_checkpoint(chk_path, agent)
        assert meta["state_encoding_version"] == STATE_ENCODING_VERSION
        assert meta["reward_version"] == REWARD_VERSION
