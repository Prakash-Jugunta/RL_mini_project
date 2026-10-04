"""
Phase 11 — Environment Integration & Single-Episode Runner Tests

Validates UIRecoveryEnv reset initialization from EpisodeSpec, candidate ordering diversity
across environment seeds, and single-episode runner behavior without multi-episode training.
"""

import numpy as np
import pytest

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.training.episode_spec import EpisodeSpec
from rl.training.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from rl.training.episode_runner import reset_from_episode_spec, run_single_episode


def test_environment_reset_from_episode_spec():
    """Verifies that UIRecoveryEnv cleanly resets from an EpisodeSpec and returns valid Dict observation."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    spec = EpisodeSpec(
        episode_id=42,
        workflow="SEARCH",
        mutation_level="L2",
        mutation_seed=33,
        environment_seed=777,
    )

    obs, info = reset_from_episode_spec(env, spec, mode="train")

    assert env.observation_space.contains(obs)
    assert info["workflow_id"] == "SEARCH"
    assert info["mutation_level"] == 2
    assert info["mutation_seed"] == 33
    assert obs["candidate_mask"].shape == (20,)


def test_candidate_ordering_diversity():
    """
    Asserts that varying environment_seed for identical public mutation configs
    produces distinct candidate ordering variations.
    """
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    specs = [
        EpisodeSpec(
            episode_id=i,
            workflow="LOGIN",
            mutation_level="L1",
            mutation_seed=11,
            environment_seed=1000 + i * 500,
        )
        for i in range(5)
    ]

    candidate_matrix_hashes = []
    for spec in specs:
        obs, _ = reset_from_episode_spec(env, spec, mode="train")
        cands = obs["candidates"]
        cands_bytes = cands.tobytes()
        candidate_matrix_hashes.append(cands_bytes)

    # Across 5 distinct environment seeds, candidate order matrices must not be 100% identical
    unique_matrices = len(set(candidate_matrix_hashes))
    assert unique_matrices > 1


def test_single_episode_runner():
    """Executes a single episode via run_single_episode and validates EpisodeResult diagnostic output."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    spec = EpisodeSpec(
        episode_id=0,
        workflow="LOGIN",
        mutation_level="L0",
        mutation_seed=11,
        environment_seed=100,
    )

    def mock_first_valid_policy(obs, info):
        mask = obs["candidate_mask"]
        valid_indices = np.flatnonzero(mask == 1)
        return int(valid_indices[0]) if len(valid_indices) > 0 else 0

    result = run_single_episode(env, mock_first_valid_policy, spec, max_decisions=20)

    assert result.episode_id == 0
    assert result.workflow == "LOGIN"
    assert result.mutation_level == "L0"
    assert result.decision_count <= 20
    assert isinstance(result.episode_return, float)
    assert isinstance(result.success, bool)
    assert result.terminated or result.truncated
