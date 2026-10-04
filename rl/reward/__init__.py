"""
Phase 8 — Reward Design & Shaping Package
=========================================

Centralized reward configuration, outcome-based reward calculation,
and reward breakdown structures for sequential UI recovery.
"""

from rl.reward.config import RewardConfig, REWARD_VERSION
from rl.reward.calculator import RewardCalculator, RewardBreakdown

__all__ = [
    "RewardConfig",
    "REWARD_VERSION",
    "RewardCalculator",
    "RewardBreakdown",
]
