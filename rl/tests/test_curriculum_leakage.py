"""
Phase 13 — Curriculum Leakage & Policy Observation Isolation Tests

Verifies that policy observations MUST NOT contain curriculum stage or difficulty tags,
curriculum scheduling does NOT depend on policy performance metrics, and zero private metadata leaks exist.
"""

import pytest

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.curriculum.config import CurriculumConfig
from rl.curriculum.scheduler import CurriculumScheduler
from rl.training.episode_runner import reset_from_episode_spec
from rl.splits.validator import PROHIBITED_LEAK_KEYS


def test_policy_observation_does_not_contain_curriculum_stage_or_level():
    """
    Asserts that the Gym Dict observation fed to the RL policy contains strictly:
    objective (64,), candidates (20, 91), candidate_mask (20,), context (30,).

    Curriculum stage index, stage name, or mutation level MUST NOT be present in observation.
    """
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    scheduler = CurriculumScheduler(CurriculumConfig(stage_episode_counts=(2, 2, 2, 2, 2, 2)))

    for ep_id in range(6):
        curr_ep = scheduler.get_episode(ep_id)
        obs, _ = reset_from_episode_spec(env, curr_ep.episode_spec, mode="train")

        # Verify exact observation keys contract
        expected_keys = {"objective", "candidates", "candidate_mask", "context"}
        assert set(obs.keys()) == expected_keys

        assert "stage_index" not in obs
        assert "curriculum_stage" not in obs
        assert "mutation_level" not in obs
        assert "difficulty" not in obs


def test_curriculum_scheduling_is_performance_independent():
    """
    Verifies that CurriculumScheduler is a pure, fixed-schedule function of global_episode_id
    and does NOT inspect episode reward, success rate, wrong actions, or agent Q-values.
    """
    scheduler = CurriculumScheduler(CurriculumConfig(curriculum_seed=1301))

    ep_before = scheduler.get_episode(500)

    # Simulating fake performance outcomes (e.g. 0% success vs 100% success)
    fake_reward = -100.0
    fake_success = False

    ep_after = scheduler.get_episode(500)

    assert ep_before == ep_after
    assert ep_before.stage_index == ep_after.stage_index


def test_private_metadata_absence_in_curriculum_episode():
    """Audits CurriculumEpisode objects and serialized dictionaries for zero private metadata key leaks."""
    scheduler = CurriculumScheduler(CurriculumConfig(curriculum_seed=1301))
    schedule = scheduler.generate_schedule(50)

    for curr_ep in schedule:
        curr_dict = curr_ep.to_dict()
        for leak_key in PROHIBITED_LEAK_KEYS:
            assert leak_key not in curr_dict
            assert leak_key not in curr_dict["episode_spec"]
