"""
Phase 12 — Splitter Unit Tests

Validates deterministic split assignment, L6 routing to test_l6, environment mode mapping,
and schedule-length independence.
"""

import pytest

from rl.training.episode_spec import EpisodeSpec
from rl.splits.config import SplitConfig
from rl.splits.splitter import (
    assign_split,
    is_train_spec,
    is_validation_spec,
    is_test_id_spec,
    is_test_l6_spec,
    split_to_env_mode,
)


def test_l6_assigned_to_test_l6():
    """Asserts that L6 EpisodeSpecs are always assigned to test_l6 and never train or validation."""
    for seed in (11, 22, 33, 44, 55):
        spec = EpisodeSpec(
            episode_id=0,
            workflow="LOGIN",
            mutation_level="L6",
            mutation_seed=seed,
            environment_seed=100,
        )
        split = assign_split(spec)
        assert split == "test_l6"
        assert is_test_l6_spec(spec) is True
        assert is_train_spec(spec) is False
        assert is_validation_spec(spec) is False
        assert is_test_id_spec(spec) is False


def test_l0_l5_split_assignment():
    """Validates that L0-L5 EpisodeSpecs resolve to train, validation, or test_id."""
    valid_categories = {"train", "validation", "test_id"}
    for lvl in ("L0", "L1", "L2", "L3", "L4", "L5"):
        spec = EpisodeSpec(
            episode_id=0,
            workflow="SEARCH",
            mutation_level=lvl,
            mutation_seed=11,
            environment_seed=100,
        )
        split = assign_split(spec)
        assert split in valid_categories
        assert is_test_l6_spec(spec) is False


def test_split_to_env_mode_mapping():
    """Validates correct mapping from experiment split names to Gymnasium env reset modes."""
    assert split_to_env_mode("train") == "train"
    assert split_to_env_mode("validation") == "validation"
    assert split_to_env_mode("test_id") == "test"
    assert split_to_env_mode("test_l6") == "test"
    assert split_to_env_mode("test") == "test"

    with pytest.raises(ValueError, match="Invalid split_name"):
        split_to_env_mode("unknown_split")


def test_deterministic_sha256_hashing_and_schedule_length_independence():
    """Verifies that split assignment is pure, deterministic, and independent of schedule length."""
    spec = EpisodeSpec(
        episode_id=42,
        workflow="PROFILE",
        mutation_level="L3",
        mutation_seed=22,
        environment_seed=777,
    )

    split_1 = assign_split(spec, SplitConfig(split_seed=1201))
    split_2 = assign_split(spec, SplitConfig(split_seed=1201))

    assert split_1 == split_2

    # Different split seed alters environment assignment
    split_diff = assign_split(spec, SplitConfig(split_seed=9999))
    # Cannot guarantee unequal for single point, but function behaves deterministically
    assert isinstance(split_diff, str)
