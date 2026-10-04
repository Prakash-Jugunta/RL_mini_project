"""
Phase 7 — Action Space & Candidate Action Contract Package
===========================================================

Defines discrete action selection semantics, candidate-mask validity,
execution result contracts, and ground-truth-free sampling helpers.
"""

from rl.action.types import ActionExecutionResult
from rl.action.sampler import sample_valid_action

__all__ = [
    "ActionExecutionResult",
    "sample_valid_action",
]
