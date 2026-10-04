"""
Phase 11 — Schedule I/O, Serialization, Hashing, and Manifest Utilities

Provides JSON and CSV serialization, SHA-256 fingerprint hashing, schedule manifest generation,
and comprehensive coverage reports for EpisodeSpec schedules.
"""

import csv
import hashlib
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

from rl.training.episode_spec import EpisodeSpec
from rl.training.episode_generator import EpisodeGeneratorConfig


def compute_schedule_hash(schedule: List[EpisodeSpec]) -> str:
    """
    Computes a canonical SHA-256 hex digest fingerprint for an episode schedule.
    Identical schedules yield identical hashes; any spec change modifies the hash.
    """
    canonical_list = [spec.to_dict() for spec in schedule]
    canonical_json = json.dumps(canonical_list, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_coverage_report(schedule: List[EpisodeSpec]) -> Dict[str, Any]:
    """
    Analyzes an episode schedule and returns detailed coverage statistics.
    """
    if not schedule:
        return {
            "total_episodes": 0,
            "unique_base_configs": 0,
            "has_held_out_l6": False,
            "duplicate_episode_ids": 0,
            "workflow_counts": {},
            "level_counts": {},
            "seed_counts": {},
            "unique_env_seeds": 0,
        }

    episode_ids = [spec.episode_id for spec in schedule]
    dup_ids = len(episode_ids) - len(set(episode_ids))

    workflow_counts = dict(Counter(spec.workflow for spec in schedule))
    level_counts = dict(Counter(spec.mutation_level for spec in schedule))
    seed_counts = dict(Counter(spec.mutation_seed for spec in schedule))

    base_tuples = set(
        (spec.workflow, spec.mutation_level, spec.mutation_seed) for spec in schedule
    )

    has_l6 = any(
        str(spec.mutation_level).strip().upper() in ("L6", "6") for spec in schedule
    )

    unique_env_seeds = len(set(spec.environment_seed for spec in schedule))

    return {
        "total_episodes": len(schedule),
        "unique_base_configs": len(base_tuples),
        "has_held_out_l6": has_l6,
        "duplicate_episode_ids": dup_ids,
        "workflow_counts": workflow_counts,
        "level_counts": level_counts,
        "seed_counts": seed_counts,
        "unique_env_seeds": unique_env_seeds,
    }


def generate_schedule_manifest(
    schedule: List[EpisodeSpec],
    config: Optional[EpisodeGeneratorConfig] = None
) -> Dict[str, Any]:
    """
    Generates a formal schedule manifest dictionary for experiment auditing.
    """
    coverage = generate_coverage_report(schedule)
    schedule_hash = compute_schedule_hash(schedule)

    manifest = {
        "generation_version": config.version if config else (schedule[0].generation_version if schedule else "phase11-v1"),
        "schedule_seed": config.schedule_seed if config else None,
        "sampling_strategy": config.sampling_strategy if config else "balanced_block",
        "requested_episodes": len(schedule),
        "generated_episodes": len(schedule),
        "schedule_hash": schedule_hash,
        "coverage": coverage,
    }
    return manifest


def save_schedule_json(
    schedule: List[EpisodeSpec],
    file_path: str,
    manifest: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves an EpisodeSpec schedule and optional manifest to a JSON file.
    """
    data = {
        "manifest": manifest or generate_schedule_manifest(schedule),
        "episodes": [spec.to_dict() for spec in schedule],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_schedule_json(
    file_path: str,
    mode: str = "train"
) -> Tuple[List[EpisodeSpec], Dict[str, Any]]:
    """
    Loads an EpisodeSpec schedule and manifest from a JSON file.
    Validates each loaded EpisodeSpec against domain rules and mode guards.
    """
    from rl.training.episode_spec import validate_episode_spec

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_episodes = data.get("episodes", [])
    manifest = data.get("manifest", {})

    schedule: List[EpisodeSpec] = []
    for raw in raw_episodes:
        spec = EpisodeSpec.from_dict(raw)
        validate_episode_spec(spec, mode=mode)
        schedule.append(spec)

    return schedule, manifest


def save_schedule_csv(schedule: List[EpisodeSpec], file_path: str) -> None:
    """
    Saves an EpisodeSpec schedule to a CSV file.
    """
    fieldnames = [
        "episode_id",
        "workflow",
        "mutation_level",
        "mutation_seed",
        "environment_seed",
        "generation_version",
    ]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in schedule:
            writer.writerow(spec.to_dict())


def load_schedule_csv(
    file_path: str,
    mode: str = "train"
) -> List[EpisodeSpec]:
    """
    Loads an EpisodeSpec schedule from a CSV file.
    Validates each loaded EpisodeSpec against domain rules and mode guards.
    """
    from rl.training.episode_spec import validate_episode_spec

    schedule: List[EpisodeSpec] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec = EpisodeSpec.from_dict(row)
            validate_episode_spec(spec, mode=mode)
            schedule.append(spec)

    return schedule
