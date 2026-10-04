"""
Phase 13 — Training CLI Guard and Confirmed Loop Execution Tests

Verifies that:
1. Without '--confirm-training', train_dqn.py exits cleanly with 0 training episodes.
2. With '--confirm-training' and a mock adapter, train_dqn.py executes training, populates replay,
   learns, respects --max-episodes, produces checkpoints, enforces Phase 12 TRAIN membership,
   has 0 L6 episodes, and supports stateful resume.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import subprocess
import pytest
import torch

from rl.env.browser_adapter import MockBrowserAdapter
from rl.training.train_dqn import run_training_loop


def test_train_dqn_safety_guard_activates_without_confirm_flag():
    """Asserts train_dqn.py exits cleanly and refuses to train without --confirm-training."""
    cmd = [
        sys.executable,
        "-m",
        "rl.training.train_dqn",
        "--max-episodes",
        "20",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="."
    )

    assert result.returncode == 0
    assert "[TRAINING SAFETY GUARD]" in result.stdout
    assert "Safety guard active: '--confirm-training' flag was NOT provided." in result.stdout
    assert "DQN training was NOT started." in result.stdout


def test_train_dqn_mock_confirmed_execution(tmp_path):
    """
    Asserts that passing --confirm-training with a mock browser adapter executes the requested
    number of episodes, populates replay, updates checkpoint files, and adheres to TRAIN split.
    """
    output_dir = str(tmp_path / "test_run")
    args = argparse.Namespace(
        curriculum_version="phase13-v1",
        curriculum_seed=1301,
        split_seed=1201,
        training_seed=2026,
        max_episodes=3,
        output_dir=output_dir,
        resume=None,
        confirm_training=True,
    )

    mock_adapter = MockBrowserAdapter(max_candidates=20)
    res = run_training_loop(args, browser_adapter=mock_adapter)

    assert res["episodes_executed"] == 3
    assert res["replay_final_size"] > 0

    # Verify Checkpoint Artifacts
    ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")
    state_path = os.path.join(output_dir, "training_state.json")
    log_path = os.path.join(output_dir, "episode_logs.json")

    assert os.path.exists(ckpt_path)
    assert os.path.exists(state_path)
    assert os.path.exists(log_path)

    with open(state_path, "r") as f:
        state_data = json.load(f)
    assert state_data["global_episode_id"] == 3

    with open(log_path, "r") as f:
        logs = json.load(f)
    assert len(logs) == 3

    # TRAIN split & zero L6 enforcement verification
    for log_item in logs:
        assert log_item["mutation_level"] != "L6"
        assert log_item["mutation_level"] != 6


def test_train_dqn_resume_support(tmp_path):
    """Asserts that --resume restores episode index and training progress without restarting episode 0."""
    output_dir = str(tmp_path / "resume_run")
    
    # Run 2 episodes initial run
    args1 = argparse.Namespace(
        curriculum_version="phase13-v1",
        curriculum_seed=1301,
        split_seed=1201,
        training_seed=2026,
        max_episodes=2,
        output_dir=output_dir,
        resume=None,
        confirm_training=True,
    )
    mock_adapter1 = MockBrowserAdapter(max_candidates=20)
    res1 = run_training_loop(args1, browser_adapter=mock_adapter1)
    assert res1["episodes_executed"] == 2

    ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")
    assert os.path.exists(ckpt_path)

    # Resume for 4 total episodes
    args2 = argparse.Namespace(
        curriculum_version="phase13-v1",
        curriculum_seed=1301,
        split_seed=1201,
        training_seed=2026,
        max_episodes=4,
        output_dir=output_dir,
        resume=ckpt_path,
        confirm_training=True,
    )
    mock_adapter2 = MockBrowserAdapter(max_candidates=20)
    res2 = run_training_loop(args2, browser_adapter=mock_adapter2)
    assert res2["episodes_executed"] == 4  # Total 4 episodes accumulated (2 initial + 2 resumed)

    with open(os.path.join(output_dir, "episode_logs.json"), "r") as f:
        logs = json.load(f)
    assert len(logs) == 4
    assert [l["global_episode_id"] for l in logs] == [0, 1, 2, 3]
