"""
Phase 12 — Train / Validation / Test Split Protocol
"""

from rl.splits.config import SplitConfig, SPLIT_VERSION
from rl.splits.splitter import (
    assign_split,
    is_train_spec,
    is_validation_spec,
    is_test_id_spec,
    is_test_l6_spec,
    split_to_env_mode,
)
from rl.splits.materialize import (
    materialize_canonical_splits,
    save_materialized_splits,
)
from rl.splits.validator import validate_split_integrity

__all__ = [
    "SplitConfig",
    "SPLIT_VERSION",
    "assign_split",
    "is_train_spec",
    "is_validation_spec",
    "is_test_id_spec",
    "is_test_l6_spec",
    "split_to_env_mode",
    "materialize_canonical_splits",
    "save_materialized_splits",
    "validate_split_integrity",
]
