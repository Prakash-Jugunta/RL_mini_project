"""
Phase 13 — Curriculum Manifest & Hashing Utilities

Provides SHA-256 fingerprint hashing and manifest generation for CurriculumEpisode schedules.
"""

import hashlib
import json
from typing import Dict, List, Any, Optional

from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION
from rl.curriculum.scheduler import CurriculumEpisode


def compute_curriculum_hash(schedule: List[CurriculumEpisode]) -> str:
    """
    Computes a canonical SHA-256 hex digest fingerprint for a curriculum episode schedule.
    """
    canonical_list = [ep.to_dict() for ep in schedule]
    canonical_json = json.dumps(canonical_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_curriculum_manifest(
    schedule: List[CurriculumEpisode],
    config: Optional[CurriculumConfig] = None
) -> Dict[str, Any]:
    """
    Generates a formal curriculum manifest dictionary for experiment auditing.
    """
    cfg = config or CurriculumConfig()

    stage_counts = {}
    for ep in schedule:
        stage_counts[ep.stage_index] = stage_counts.get(ep.stage_index, 0) + 1

    manifest = {
        "curriculum_version": cfg.version,
        "curriculum_seed": cfg.curriculum_seed,
        "sampling_strategy": cfg.sampling_strategy,
        "total_episodes_scheduled": len(schedule),
        "stage_breakdown": stage_counts,
        "validation_interval": cfg.validation_interval,
        "checkpoint_interval": cfg.checkpoint_interval,
        "curriculum_hash": compute_curriculum_hash(schedule),
    }
    return manifest
