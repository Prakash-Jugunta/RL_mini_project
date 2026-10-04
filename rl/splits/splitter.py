"""
Phase 12 — Split Assignment & Mapping Utility

Implements deterministic, SHA-256 based split assignment, membership query functions,
and Gymnasium environment mode mapping.
"""

import hashlib
import struct
from typing import Optional

from rl.training.episode_spec import EpisodeSpec, HELD_OUT_MUTATION_LEVEL
from rl.splits.config import SplitConfig, SPLIT_VERSION

from rl.training.episode_generator import _derive_seed

VALID_SPLIT_NAMES = ("train", "validation", "test_id", "test_l6")


def assign_split(
    spec: EpisodeSpec,
    config: Optional[SplitConfig] = None,
    repeat_index: Optional[int] = None
) -> str:
    """
    Deterministically assigns an EpisodeSpec to a split category.

    Categories:
    - 'test_l6': Held-out L6 mutation level exclusively.
    - 'train': L0-L5 environment-seed training realization.
    - 'validation': L0-L5 environment-seed validation realization.
    - 'test_id': L0-L5 environment-seed in-distribution test realization.

    Guarantees:
    - Pure function: deterministic across platforms and sessions (uses SHA-256).
    - Schedule-length independent.
    - L6 is NEVER assigned to 'train' or 'validation'.
    """
    cfg = config or SplitConfig()

    clean_lvl = str(spec.mutation_level).strip().upper()
    if not clean_lvl.startswith("L"):
        clean_lvl = f"L{clean_lvl}"

    if clean_lvl == HELD_OUT_MUTATION_LEVEL:
        return "test_l6"

    # Explicit repeat index partitioning if provided
    if repeat_index is not None:
        rem = repeat_index % 10
        if rem < 8:
            return "train"
        elif rem == 8:
            return "validation"
        else:
            return "test_id"

    # Search for canonical stratum realization match in rep range [0..100]
    for rep in range(100):
        expected_seed = _derive_seed(
            cfg.split_seed,
            f"stratum_env:{spec.workflow}:{clean_lvl}:{spec.mutation_seed}",
            rep
        )
        if expected_seed == spec.environment_seed:
            rem = rep % 10
            if rem < 8:
                return "train"
            elif rem == 8:
                return "validation"
            else:
                return "test_id"

    # Fallback deterministic SHA-256 threshold for arbitrary environment seeds
    raw_str = (
        f"{cfg.split_seed}:{spec.workflow}:{clean_lvl}:"
        f"{spec.mutation_seed}:{spec.environment_seed}"
    ).encode("utf-8")
    hash_bytes = hashlib.sha256(raw_str).digest()
    (val,) = struct.unpack(">I", hash_bytes[:4])
    norm_val = (val % 1000000) / 1000000.0

    if norm_val < cfg.train_ratio:
        return "train"
    elif norm_val < (cfg.train_ratio + cfg.validation_ratio):
        return "validation"
    else:
        return "test_id"


def is_train_spec(spec: EpisodeSpec, config: Optional[SplitConfig] = None) -> bool:
    """Returns True if spec is assigned to the 'train' split."""
    return assign_split(spec, config) == "train"


def is_validation_spec(spec: EpisodeSpec, config: Optional[SplitConfig] = None) -> bool:
    """Returns True if spec is assigned to the 'validation' split."""
    return assign_split(spec, config) == "validation"


def is_test_id_spec(spec: EpisodeSpec, config: Optional[SplitConfig] = None) -> bool:
    """Returns True if spec is assigned to the 'test_id' split."""
    return assign_split(spec, config) == "test_id"


def is_test_l6_spec(spec: EpisodeSpec, config: Optional[SplitConfig] = None) -> bool:
    """Returns True if spec is assigned to the 'test_l6' split."""
    return assign_split(spec, config) == "test_l6"


def split_to_env_mode(split_name: str) -> str:
    """
    Maps Phase 12 experiment split name to Gymnasium UIRecoveryEnv reset mode.

    Mapping:
        'train'       -> 'train'
        'validation'  -> 'validation'
        'test_id'     -> 'test'
        'test_l6'     -> 'test'
    """
    clean_split = str(split_name).strip().lower()
    if clean_split == "train":
        return "train"
    elif clean_split == "validation":
        return "validation"
    elif clean_split in ("test_id", "test_l6", "test"):
        return "test"
    else:
        raise ValueError(f"Invalid split_name '{split_name}'. Must be one of {VALID_SPLIT_NAMES}")
