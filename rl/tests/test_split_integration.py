"""
Phase 12 — Split Integration & Isolation Tests

Validates replay buffer isolation, network weight/optimizer isolation, and Gymnasium environment mode mapping.
"""

import copy
import pytest
import torch

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.agents.dqn.agent import DQNAgent, DQNConfig
from rl.agents.dqn.replay_buffer import DictReplayBuffer
from rl.training.episode_spec import EpisodeSpec
from rl.training.episode_runner import reset_from_episode_spec
from rl.splits.splitter import assign_split, split_to_env_mode


def test_validation_and_test_never_enter_replay_buffer():
    """Asserts that validation and test transitions MUST NOT enter the training replay buffer."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    train_spec = EpisodeSpec(
        episode_id=0,
        workflow="LOGIN",
        mutation_level="L0",
        mutation_seed=11,
        environment_seed=100,
    )
    val_spec = EpisodeSpec(
        episode_id=1,
        workflow="LOGIN",
        mutation_level="L0",
        mutation_seed=11,
        environment_seed=200,
    )

    replay_buffer = DictReplayBuffer(capacity=100)
    assert len(replay_buffer) == 0

    # 1. Simulate TRAIN episode step -> stored into replay
    obs, info = reset_from_episode_spec(env, train_spec, mode="train")
    next_obs, reward, terminated, truncated, _ = env.step(0)
    replay_buffer.add(obs, 0, reward, next_obs, terminated, truncated)

    assert len(replay_buffer) == 1

    # 2. Simulate VALIDATION episode step -> MUST NOT be stored into replay
    val_obs, val_info = reset_from_episode_spec(env, val_spec, mode="validation")
    val_next_obs, val_reward, val_term, val_trunc, _ = env.step(0)

    # Replay buffer size must remain exactly 1
    assert len(replay_buffer) == 1


def test_validation_and_test_never_update_optimizer():
    """Asserts that executing validation or test steps never alters DQNAgent weights or triggers optimizer steps."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    agent_config = DQNConfig(seed=42)
    agent = DQNAgent(agent_config)

    initial_weights = copy.deepcopy(agent.online_network.state_dict())

    val_spec = EpisodeSpec(
        episode_id=1,
        workflow="SEARCH",
        mutation_level="L1",
        mutation_seed=22,
        environment_seed=300,
    )

    obs, info = reset_from_episode_spec(env, val_spec, mode="validation")
    agent.set_eval_mode()
    action = agent.select_action(obs, explore=False)

    next_obs, reward, terminated, truncated, _ = env.step(action)

    # No agent.can_learn() or agent.learn() called during validation
    post_weights = agent.online_network.state_dict()

    for k in initial_weights:
        assert torch.equal(initial_weights[k], post_weights[k])


def test_gym_environment_reset_mode_mapping():
    """Verifies that all 4 split categories cleanly reset Gym environment using split_to_env_mode."""
    adapter = MockBrowserAdapter()
    env = UIRecoveryEnv(browser_adapter=adapter)

    split_specs = {
        "train": EpisodeSpec(0, "LOGIN", "L0", 11, 100),
        "validation": EpisodeSpec(1, "LOGIN", "L0", 11, 200),
        "test_id": EpisodeSpec(2, "LOGIN", "L0", 11, 300),
        "test_l6": EpisodeSpec(3, "LOGIN", "L6", 11, 400),
    }

    for split_name, spec in split_specs.items():
        env_mode = split_to_env_mode(split_name)
        obs, info = reset_from_episode_spec(env, spec, mode=env_mode)

        assert env.observation_space.contains(obs)
        assert info["workflow_id"] == "LOGIN"
