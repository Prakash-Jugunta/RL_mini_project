"""
Phase 13 — Curriculum Configuration

Defines CurriculumConfig dataclass and stage progression rules for progressive mixed training.
"""

from dataclasses import dataclass
from typing import Tuple

from rl.training.episode_spec import (
    TRAINING_MUTATION_LEVELS,
    HELD_OUT_MUTATION_LEVEL,
)

CURRICULUM_VERSION: str = "phase13-v1"

DEFAULT_STAGE_EPISODE_COUNTS: Tuple[int, ...] = (1000, 1500, 2000, 2500, 3000, 5000)

DEFAULT_ACTIVE_LEVELS_BY_STAGE: Tuple[Tuple[str, ...], ...] = (
    ("L0",),
    ("L0", "L1"),
    ("L0", "L1", "L2"),
    ("L0", "L1", "L2", "L3"),
    ("L0", "L1", "L2", "L3", "L4"),
    ("L0", "L1", "L2", "L3", "L4", "L5"),
)


@dataclass(frozen=True)
class CurriculumConfig:
    """
    Configuration specifying progressive mixed training curriculum scheduling.

    Fields:
        curriculum_seed: Independent base seed for deterministic curriculum scheduling (default 1301).
        stage_episode_counts: Tuple of episode budget counts for stages 0..5 (default (1000, 1500, 2000, 2500, 3000, 5000)).
        active_levels_by_stage: Tuple of active mutation level tuples per stage.
        sampling_strategy: Intra-stage sampling strategy name ('balanced_active_levels').
        validation_interval: Training episode interval between validation evaluations (500).
        checkpoint_interval: Training episode interval between checkpoint saves (500).
        version: Protocol version string ('phase13-v1').
    """

    curriculum_seed: int = 1301
    stage_episode_counts: Tuple[int, ...] = DEFAULT_STAGE_EPISODE_COUNTS
    active_levels_by_stage: Tuple[Tuple[str, ...], ...] = DEFAULT_ACTIVE_LEVELS_BY_STAGE
    sampling_strategy: str = "balanced_active_levels"
    validation_interval: int = 500
    checkpoint_interval: int = 500
    version: str = CURRICULUM_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.curriculum_seed, int):
            raise ValueError(f"curriculum_seed must be an integer, got {type(self.curriculum_seed)}")

        if len(self.stage_episode_counts) != len(self.active_levels_by_stage):
            raise ValueError(
                f"Mismatch between stage_episode_counts length ({len(self.stage_episode_counts)}) "
                f"and active_levels_by_stage length ({len(self.active_levels_by_stage)})."
            )

        if any(count <= 0 for count in self.stage_episode_counts):
            raise ValueError("All stage_episode_counts must be strictly positive (> 0).")

        if self.validation_interval <= 0:
            raise ValueError(f"validation_interval must be positive, got {self.validation_interval}")

        if self.checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {self.checkpoint_interval}")

        # L6 prohibition check and prior level retention check
        previous_levels = set()
        for stage_idx, active_levels in enumerate(self.active_levels_by_stage):
            clean_levels = {str(lvl).strip().upper() for lvl in active_levels}
            if HELD_OUT_MUTATION_LEVEL in clean_levels:
                raise ValueError(
                    f"EXPERIMENTAL INTEGRITY VIOLATION: Level '{HELD_OUT_MUTATION_LEVEL}' is HELD OUT "
                    f"and CANNOT be included in curriculum Stage {stage_idx}."
                )

            for lvl in clean_levels:
                if lvl not in TRAINING_MUTATION_LEVELS:
                    raise ValueError(f"Invalid mutation level '{lvl}' in Stage {stage_idx}.")

            # Prior level retention check
            if not previous_levels.issubset(clean_levels):
                missing = previous_levels - clean_levels
                raise ValueError(
                    f"Curriculum Stage {stage_idx} failed previous level retention principle. "
                    f"Missing prior active levels: {missing}"
                )
            previous_levels.update(clean_levels)

    def total_episodes(self) -> int:
        """Returns the total cumulative training episode budget across all stages."""
        return sum(self.stage_episode_counts)
