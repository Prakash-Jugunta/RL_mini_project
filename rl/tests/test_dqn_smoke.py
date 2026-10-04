"""
Smoke tests for Masked Double DQN Agent training loop & real browser integration.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import numpy as np

from rl.training.smoke_train_dqn import run_mock_smoke_training, run_real_browser_integration_test


def test_mock_smoke_training():
    res = run_mock_smoke_training(max_env_steps=150, verbose=False)

    assert res["env_steps"] == 150
    assert res["replay_final_size"] == 150
    assert res["learning_steps"] > 0
    assert not np.isnan(res["mean_loss"])
    assert not np.isnan(res["mean_q"])
    assert os.path.exists(res["checkpoint_file"])


def test_real_browser_integration_smoke():
    res = run_real_browser_integration_test(verbose=False)

    assert res["steps"] > 0
    assert res["replay_size"] == res["steps"]
    assert isinstance(res["workflow_completed"], bool)
