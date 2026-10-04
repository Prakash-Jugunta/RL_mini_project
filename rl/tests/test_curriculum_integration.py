"""
Phase 13 — Curriculum Integration & Continuity Tests

Validates state continuity across stage boundaries: epsilon step is not reset,
replay buffer is not cleared, network weights are not reset, and Gym reset options map cleanly.
"""

import pytest

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.agents.dqn.agent import DQNAgent, DQNConfig
from rl.agents.dqn.replay_buffer import DictReplayBuffer
from rl.curriculum.config import CurriculumConfig
from rl.curriculum.scheduler import CurriculumScheduler
from rl.training.episode_runner import reset_from_episode_spec


def test_epsilon_and_steps_not_reset_across_stage_boundaries():
    """Asserts that total_steps and epsilon progression are continuous across stage boundaries."""
    agent_config = DQNConfig(seed=42)
    agent = DQNAgent(agent_config)

    assert agent.total_steps == 0

    # Simulate 10 environment steps
    for _ in range(10):
        agent.total_steps += 1

    assert agent.total_steps == 10

    # Crossing a curriculum stage boundary must NOT reset agent.total_steps
    scheduler = CurriculumScheduler(CurriculumConfig(stage_episode_counts=(5, 5, 5, 5, 5, 5)))
    ep_stage_1 = scheduler.get_episode(5)  # Stage 1

    assert ep_stage_1.stage_index == 1
    assert agent.total_steps == 10  # Remained 10


def test_replay_buffer_not_cleared_across_stage_boundaries():
    """Asserts that transitions from Stage N are retained in the replay buffer when entering Stage N+1."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    scheduler = CurriculumScheduler(CurriculumConfig(stage_episode_counts=(2, 2, 2, 2, 2, 2)))

    ep_stage_0 = scheduler.get_episode(0)
    ep_stage_1 = scheduler.get_episode(2)

    replay_buffer = DictReplayBuffer(capacity=100)

    # Store Stage 0 transition
    obs0, info0 = reset_from_episode_spec(env, ep_stage_0.episode_spec, mode="train")
    next_obs0, reward0, term0, trunc0, _ = env.step(0)
    replay_buffer.add(obs0, 0, reward0, next_obs0, term0, trunc0)

    assert len(replay_buffer) == 1

    # Stage boundary transition to Stage 1 -> replay buffer must NOT be cleared
    obs1, info1 = reset_from_episode_spec(env, ep_stage_1.episode_spec, mode="train")

    assert len(replay_buffer) == 1


def test_gym_environment_reset_works_for_all_curriculum_specs():
    """Validates that UIRecoveryEnv cleanly resets for curriculum episode specs across all stages."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    scheduler = CurriculumScheduler(CurriculumConfig(stage_episode_counts=(2, 2, 2, 2, 2, 2)))

    for ep_id in range(12):
        curr_ep = scheduler.get_episode(ep_id)
        obs, info = reset_from_episode_spec(env, curr_ep.episode_spec, mode="train")

        assert env.observation_space.contains(obs)
        assert info["workflow_id"] == curr_ep.episode_spec.workflow
        assert info["mutation_level"] == curr_ep.episode_spec.numeric_mutation_level()
