"""
Unit tests for DictReplayBuffer.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np

from rl.agents.dqn.replay_buffer import DictReplayBuffer


def make_dummy_obs(val: float = 1.0) -> dict:
    return {
        "objective": np.full(64, val, dtype=np.float32),
        "context": np.full(30, val, dtype=np.float32),
        "candidates": np.full((20, 91), val, dtype=np.float32),
        "candidate_mask": np.ones(20, dtype=np.float32),
    }


def test_replay_buffer_capacity_and_rollover():
    buf = DictReplayBuffer(capacity=5, seed=42)
    assert len(buf) == 0

    for i in range(10):
        obs = make_dummy_obs(float(i))
        buf.add(obs, action=i % 4, reward=float(i), next_obs=obs, terminated=False, truncated=False)

    assert len(buf) == 5, f"Expected size 5 after rollover, got {len(buf)}"


def test_replay_buffer_sampling_shapes_and_types():
    buf = DictReplayBuffer(capacity=100, seed=42)
    for i in range(20):
        obs = make_dummy_obs(float(i))
        next_obs = make_dummy_obs(float(i + 1))
        buf.add(obs, action=i % 5, reward=1.0, next_obs=next_obs, terminated=(i == 19), truncated=False)

    obs_b, act_b, rew_b, next_obs_b, term_b, trunc_b = buf.sample(batch_size=8, device="cpu")

    assert obs_b["objective"].shape == (8, 64)
    assert obs_b["context"].shape == (8, 30)
    assert obs_b["candidates"].shape == (8, 20, 91)
    assert obs_b["candidate_mask"].shape == (8, 20)

    assert act_b.shape == (8,)
    assert rew_b.shape == (8,)

    assert next_obs_b["objective"].shape == (8, 64)
    assert term_b.shape == (8,)
    assert trunc_b.shape == (8,)


def test_replay_buffer_private_metadata_absence_audit():
    """
    LEAKAGE AUDIT:
    Verifies that the replay buffer storage schema strictly excludes private evaluator metadata.
    """
    buf = DictReplayBuffer(capacity=10, seed=42)
    prohibited_keys = ["expected_role", "semantic_role", "data_semantic_role", "private_meta", "mutation_level", "mutation_seed"]

    for attr_name in dir(buf):
        for key in prohibited_keys:
            assert key not in attr_name, f"Replay buffer storage schema contains prohibited field: {attr_name}"
