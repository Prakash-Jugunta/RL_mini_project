"""
Phase 13 — Difficulty & Curriculum Strategy
"""

from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION
from rl.curriculum.scheduler import (
    CurriculumEpisode,
    CurriculumScheduler,
)
from rl.curriculum.validator import validate_curriculum_integrity
from rl.curriculum.manifest import (
    generate_curriculum_manifest,
    compute_curriculum_hash,
)

__all__ = [
    "CurriculumConfig",
    "CURRICULUM_VERSION",
    "CurriculumEpisode",
    "CurriculumScheduler",
    "validate_curriculum_integrity",
    "generate_curriculum_manifest",
    "compute_curriculum_hash",
]
