"""
Training package for RL agents.
"""

from rl.training.smoke_train_dqn import run_mock_smoke_training, run_real_browser_integration_test

__all__ = [
    "run_mock_smoke_training",
    "run_real_browser_integration_test",
]
