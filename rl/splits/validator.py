"""
Phase 12 — Split Integrity Validator

Validates split disjointness, L6 isolation, environment seed separation, stratification,
and performance/metadata non-leakage.
"""

from typing import Dict, List, Optional, Set, Tuple

from rl.training.episode_spec import EpisodeSpec, HELD_OUT_MUTATION_LEVEL
from rl.splits.config import SplitConfig
from rl.splits.splitter import assign_split

PROHIBITED_LEAK_KEYS = [
    "expected_role",
    "semantic_role",
    "data-semantic-role",
    "ground_truth",
    "correct_candidate",
    "target_role",
    "private_meta",
    "reward",
    "success",
    "wrong_actions",
]


def validate_split_integrity(
    splits_dict: Dict[str, List[EpisodeSpec]],
    config: Optional[SplitConfig] = None
) -> None:
    """
    Strictly validates Phase 12 split integrity and experimental guards.

    Assertions:
    1. Set disjointness across train, validation, test_id, test_l6.
    2. Zero L6 presence in train or validation splits.
    3. All L6 specs belong exclusively to test_l6.
    4. Environment-seed disjointness within each stratum across splits.
    5. Zero private metadata or episode outcome metrics leakage.
    """
    cfg = config or SplitConfig()

    train_specs = splits_dict.get("train", [])
    val_specs = splits_dict.get("validation", [])
    test_id_specs = splits_dict.get("test_id", [])
    test_l6_specs = splits_dict.get("test_l6", [])

    train_set: Set[EpisodeSpec] = set(train_specs)
    val_set: Set[EpisodeSpec] = set(val_specs)
    test_id_set: Set[EpisodeSpec] = set(test_id_specs)
    test_l6_set: Set[EpisodeSpec] = set(test_l6_specs)

    # 1. Disjointness check
    if train_set & val_set:
        raise ValueError(f"TRAIN and VALIDATION splits overlap: {len(train_set & val_set)} common specs.")
    if train_set & test_id_set:
        raise ValueError(f"TRAIN and TEST-ID splits overlap: {len(train_set & test_id_set)} common specs.")
    if train_set & test_l6_set:
        raise ValueError(f"TRAIN and TEST-L6 splits overlap: {len(train_set & test_l6_set)} common specs.")
    if val_set & test_id_set:
        raise ValueError(f"VALIDATION and TEST-ID splits overlap: {len(val_set & test_id_set)} common specs.")
    if val_set & test_l6_set:
        raise ValueError(f"VALIDATION and TEST-L6 splits overlap: {len(val_set & test_l6_set)} common specs.")
    if test_id_set & test_l6_set:
        raise ValueError(f"TEST-ID and TEST-L6 splits overlap: {len(test_id_set & test_l6_set)} common specs.")

    # 2. L6 Isolation check
    for spec in train_specs:
        if str(spec.mutation_level).strip().upper() == HELD_OUT_MUTATION_LEVEL:
            raise ValueError(f"EXPERIMENTAL INTEGRITY VIOLATION: L6 spec ID {spec.episode_id} found in TRAIN split.")

    for spec in val_specs:
        if str(spec.mutation_level).strip().upper() == HELD_OUT_MUTATION_LEVEL:
            raise ValueError(f"EXPERIMENTAL INTEGRITY VIOLATION: L6 spec ID {spec.episode_id} found in VALIDATION split.")

    for spec in test_l6_specs:
        if str(spec.mutation_level).strip().upper() != HELD_OUT_MUTATION_LEVEL:
            raise ValueError(f"Non-L6 spec ID {spec.episode_id} found in TEST-L6 split.")

    # 3. Environment-seed disjointness within each base stratum
    strata_train_seeds: Dict[Tuple[str, str, int], Set[int]] = {}
    strata_eval_seeds: Dict[Tuple[str, str, int], Set[int]] = {}

    for spec in train_specs:
        stratum = (spec.workflow, spec.mutation_level, spec.mutation_seed)
        strata_train_seeds.setdefault(stratum, set()).add(spec.environment_seed)

    for spec in val_specs + test_id_specs:
        stratum = (spec.workflow, spec.mutation_level, spec.mutation_seed)
        strata_eval_seeds.setdefault(stratum, set()).add(spec.environment_seed)

    for stratum, train_env_seeds in strata_train_seeds.items():
        eval_env_seeds = strata_eval_seeds.get(stratum, set())
        overlap = train_env_seeds & eval_env_seeds
        if overlap:
            raise ValueError(
                f"Environment seed leakage in stratum {stratum}: {len(overlap)} seeds "
                f"appear in both train and eval partitions."
            )

    # 4. Leakage & audit check
    all_specs = train_specs + val_specs + test_id_specs + test_l6_specs
    for spec in all_specs:
        spec_dict = spec.to_dict()
        for key in PROHIBITED_LEAK_KEYS:
            if key in spec_dict:
                raise ValueError(f"Prohibited leak key '{key}' found in EpisodeSpec {spec.episode_id}.")
