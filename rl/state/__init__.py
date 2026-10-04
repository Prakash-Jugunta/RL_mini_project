"""
Phase 6 — State Representation Package
======================================

Provides fixed-size observation encoding for Reinforcement Learning agents.
Decoupled from private evaluator metadata (ground truth) and experiment mutation parameters.
"""

from rl.state.feature_schema import StateFeatureSchema
from rl.state.text_encoder import DeterministicTextEncoder
from rl.state.encoder import StateEncoder

__all__ = [
    "StateFeatureSchema",
    "DeterministicTextEncoder",
    "StateEncoder",
]
