"""
Phase 13 — Curriculum Configuration Unit Tests

Validates stage level sets, anti-forgetting level retention, default 15,000 episode budget,
and L6 prohibition guards.
"""

import pytest

from rl.curriculum.config import (
    CurriculumConfig,
    DEFAULT_STAGE_EPISODE_COUNTS,
    DEFAULT_ACTIVE_LEVELS_BY_STAGE,
    CURRICULUM_VERSION,
)
from rl.training.episode_spec import HELD_OUT_MUTATION_LEVEL


def test_curriculum_config_defaults():
    """Validates default stage definitions, episode budget sum, and version string."""
    config = CurriculumConfig()

    assert config.version == CURRICULUM_VERSION
    assert config.curriculum_seed == 1301
    assert len(config.stage_episode_counts) == 6
    assert sum(config.stage_episode_counts) == 15000
    assert config.total_episodes() == 15000


def test_stage_level_sets_and_retention():
    """Asserts that Stage 0=(L0,), Stage 1=(L0,L1)... and each stage retains all prior levels."""
    config = CurriculumConfig()

    expected_stages = (
        ("L0",),
        ("L0", "L1"),
        ("L0", "L1", "L2"),
        ("L0", "L1", "L2", "L3"),
        ("L0", "L1", "L2", "L3", "L4"),
        ("L0", "L1", "L2", "L3", "L4", "L5"),
    )

    assert config.active_levels_by_stage == expected_stages

    # Verify anti-forgetting retention
    previous_levels = set()
    for stage_idx, levels in enumerate(config.active_levels_by_stage):
        current_set = set(levels)
        assert previous_levels.issubset(current_set)
        previous_levels = current_set


def test_l6_prohibition_guard_in_config():
    """Asserts that attempting to include L6 in any curriculum stage raises ValueError."""
    bad_stages = (
        ("L0",),
        ("L0", "L6"),  # Violation!
    )
    with pytest.raises(ValueError, match="HELD OUT and CANNOT be included"):
        CurriculumConfig(
            stage_episode_counts=(1000, 1500),
            active_levels_by_stage=bad_stages
        )


def test_invalid_level_retention_rejection():
    """Asserts that failing level retention (e.g. dropping L0 in Stage 1) raises ValueError."""
    flawed_retention_stages = (
        ("L0",),
        ("L1",),  # Dropped L0!
    )
    with pytest.raises(ValueError, match="failed previous level retention principle"):
        CurriculumConfig(
            stage_episode_counts=(1000, 1500),
            active_levels_by_stage=flawed_retention_stages
        )
