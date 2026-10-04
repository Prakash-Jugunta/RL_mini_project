"""
Phase 13 — Curriculum Validator

Validates curriculum schedule integrity, TRAIN split membership, L6 exclusion, level retention,
and non-leakage bounds.
"""

from typing import List, Optional

from rl.training.episode_spec import (
    validate_episode_spec,
    HELD_OUT_MUTATION_LEVEL,
    TRAINING_MUTATION_LEVELS,
)
from rl.splits.config import SplitConfig
from rl.splits.splitter import assign_split
from rl.splits.validator import PROHIBITED_LEAK_KEYS
from rl.curriculum.config import CurriculumConfig
from rl.curriculum.scheduler import CurriculumEpisode


def validate_curriculum_integrity(
    schedule: List[CurriculumEpisode],
    config: Optional[CurriculumConfig] = None,
    split_config: Optional[SplitConfig] = None
) -> None:
    """
    Strictly validates CurriculumEpisode schedule integrity.

    Assertions:
    1. Zero L6 episodes present in training curriculum.
    2. Every single episode specification resolves to Phase 12 'train' split.
    3. Previous active mutation levels retained across stage progression.
    4. Continuous zero-indexed global_episode_id ordering.
    5. Zero private evaluator metadata leakage.
    """
    cfg = config or CurriculumConfig()
    split_cfg = split_config or SplitConfig()

    previous_global_id = -1
    stage_active_levels = {}

    for ep in schedule:
        # 1. Global ID sequence continuity
        if ep.global_episode_id != previous_global_id + 1:
            raise ValueError(
                f"Discontinuous global episode ID sequence: expected {previous_global_id + 1}, "
                f"got {ep.global_episode_id}"
            )
        previous_global_id = ep.global_episode_id

        # 2. Spec validation and L6 exclusion
        spec = ep.episode_spec
        validate_episode_spec(spec, mode="train")
        clean_lvl = str(spec.mutation_level).strip().upper()
        if clean_lvl == HELD_OUT_MUTATION_LEVEL:
            raise ValueError(
                f"EXPERIMENTAL INTEGRITY VIOLATION: Level L6 found in curriculum episode {ep.global_episode_id}."
            )

        # 3. Phase 12 TRAIN split membership verification
        split_name = assign_split(spec, split_cfg)
        if split_name != "train":
            raise ValueError(
                f"EXPERIMENTAL INTEGRITY VIOLATION: Episode {ep.global_episode_id} "
                f"assigned to non-train split '{split_name}'."
            )

        # 4. Track stage active levels
        stage_active_levels[ep.stage_index] = set(ep.active_levels)

        # 5. Private metadata audit
        spec_dict = spec.to_dict()
        for leak_key in PROHIBITED_LEAK_KEYS:
            if leak_key in spec_dict:
                raise ValueError(
                    f"Prohibited leak key '{leak_key}' found in CurriculumEpisode {ep.global_episode_id}."
                )

    # 6. Anti-forgetting level retention audit across stage progression
    sorted_stages = sorted(stage_active_levels.keys())
    accumulated = set()
    for s_idx in sorted_stages:
        current_levels = stage_active_levels[s_idx]
        if not accumulated.issubset(current_levels):
            missing = accumulated - current_levels
            raise ValueError(
                f"Stage {s_idx} failed level retention principle. Missing prior levels: {missing}"
            )
        accumulated.update(current_levels)
