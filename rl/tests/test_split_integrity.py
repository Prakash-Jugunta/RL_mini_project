"""
Phase 12 — Split Integrity Unit Tests

Validates canonical materialization counts, split disjointness, L6 isolation, stratum stratification,
environment seed separation, and leakage audits.
"""

from collections import Counter
import pytest

from rl.training.episode_spec import EpisodeSpec, HELD_OUT_MUTATION_LEVEL
from rl.splits.config import SplitConfig
from rl.splits.splitter import assign_split
from rl.splits.materialize import materialize_canonical_splits
from rl.splits.validator import validate_split_integrity, PROHIBITED_LEAK_KEYS


def test_canonical_materialization_counts():
    """Validates exact canonical split materialization counts."""
    config = SplitConfig(split_seed=1201)
    splits = materialize_canonical_splits(config)

    assert len(splits["train"]) == 960
    assert len(splits["validation"]) == 120
    assert len(splits["test_id"]) == 120
    assert len(splits["test_l6"]) == 100


def test_disjointness_and_l6_isolation():
    """Asserts complete set disjointness and 100% L6 isolation in canonical materialization."""
    config = SplitConfig(split_seed=1201)
    splits = materialize_canonical_splits(config)

    # Integrity validator should pass with no exceptions
    validate_split_integrity(splits, config)

    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_id_set = set(splits["test_id"])
    test_l6_set = set(splits["test_l6"])

    assert len(train_set & val_set) == 0
    assert len(train_set & test_id_set) == 0
    assert len(train_set & test_l6_set) == 0
    assert len(val_set & test_id_set) == 0
    assert len(val_set & test_l6_set) == 0
    assert len(test_id_set & test_l6_set) == 0

    assert all(str(s.mutation_level).upper() != "L6" for s in splits["train"])
    assert all(str(s.mutation_level).upper() != "L6" for s in splits["validation"])
    assert all(str(s.mutation_level).upper() == "L6" for s in splits["test_l6"])


def test_stratification_coverage():
    """Verifies that all 120 L0-L5 strata have exact 8/1/1 allocation in canonical materialization."""
    config = SplitConfig(split_seed=1201)
    splits = materialize_canonical_splits(config)

    train_strata = Counter((s.workflow, s.mutation_level, s.mutation_seed) for s in splits["train"])
    val_strata = Counter((s.workflow, s.mutation_level, s.mutation_seed) for s in splits["validation"])
    test_id_strata = Counter((s.workflow, s.mutation_level, s.mutation_seed) for s in splits["test_id"])

    assert len(train_strata) == 120
    assert len(val_strata) == 120
    assert len(test_id_strata) == 120

    for stratum in train_strata:
        assert train_strata[stratum] == 8
        assert val_strata[stratum] == 1
        assert test_id_strata[stratum] == 1


def test_no_performance_or_private_metadata_leakage():
    """Audits EpisodeSpec objects to ensure zero private metadata or performance outcome keys."""
    config = SplitConfig(split_seed=1201)
    splits = materialize_canonical_splits(config)

    all_specs = splits["train"] + splits["validation"] + splits["test_id"] + splits["test_l6"]
    for spec in all_specs:
        spec_dict = spec.to_dict()
        for leak_key in PROHIBITED_LEAK_KEYS:
            assert leak_key not in spec_dict
