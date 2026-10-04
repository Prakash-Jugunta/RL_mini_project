"""
Unit & Integration Tests for Authoritative Workflow Success Reporting

Verifies:
1. Successful trajectories for all 4 workflows (SEARCH, PROFILE, LOGIN, CHECKOUT) return Success: True.
2. Unsuccessful/truncated trajectories return Success: False.
3. Both UIRecoveryEnv info dict and train_dqn / episode_runner logging use the authoritative workflow_success / workflow_completed signal.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.training.episode_spec import EpisodeSpec
from rl.training.episode_runner import run_single_episode


def test_mock_successful_trajectories_all_workflows():
    """Asserts that executing all correct steps for each workflow produces success == True."""
    for workflow_id in ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]:
        adapter = MockBrowserAdapter(max_candidates=20)
        env = UIRecoveryEnv(workflow_id=workflow_id, mode="train", browser_adapter=adapter)
        obs, info = env.reset()

        terminated = False
        truncated = False
        step_info = info

        while not (terminated or truncated):
            # Select the correct candidate matching expected_role from private metadata
            expected_role = env.workflow.steps[env.current_step_index].expected_role
            correct_action = 0
            for idx, meta in enumerate(env._private_meta):
                if meta.semantic_role == expected_role:
                    correct_action = idx
                    break

            obs, reward, terminated, truncated, step_info = env.step(correct_action)

        assert terminated is True
        assert step_info.get("workflow_completed") is True
        assert step_info.get("workflow_success") is True


def test_mock_failed_trajectory():
    """Asserts that failing to complete a workflow returns success == False."""
    adapter = MockBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(workflow_id="LOGIN", max_episode_steps=3, mode="train", browser_adapter=adapter)
    obs, info = env.reset()

    # Step repeatedly with wrong action
    obs, reward, terminated, truncated, step_info = env.step(19)

    assert step_info.get("workflow_completed") is False
    assert step_info.get("workflow_success") is False


def test_episode_runner_success_reporting():
    """Asserts single episode runner reports EpisodeResult.success correctly."""
    adapter = MockBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(workflow_id="SEARCH", mode="train", browser_adapter=adapter)
    spec = EpisodeSpec(episode_id=0, workflow="SEARCH", mutation_level="L0", mutation_seed=11, environment_seed=42)

    def perfect_policy(obs, info):
        expected_role = env.workflow.steps[env.current_step_index].expected_role
        for idx, meta in enumerate(env._private_meta):
            if meta.semantic_role == expected_role:
                return idx
        return 0

    res = run_single_episode(env, perfect_policy, spec)
    assert res.success is True
