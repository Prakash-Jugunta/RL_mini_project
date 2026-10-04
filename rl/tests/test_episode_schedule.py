"""
Phase 11 — Schedule Serialization, Hashing, and Leakage Audit Tests

Validates JSON/CSV round-tripping, SHA-256 fingerprint hash stability, manifest generation,
coverage reporting, deserialization validation enforcement, and leakage prevention audits.
"""

import json
import os
import tempfile
import pytest

from rl.training.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from rl.training.schedule_io import (
    compute_schedule_hash,
    generate_schedule_manifest,
    generate_coverage_report,
    save_schedule_json,
    load_schedule_json,
    save_schedule_csv,
    load_schedule_csv,
)

PROHIBITED_LEAK_KEYS = [
    "expected_role",
    "semantic_role",
    "data-semantic-role",
    "ground_truth",
    "correct_candidate",
    "target_role",
    "private_meta",
]


def test_json_round_trip():
    """Validates JSON schedule saving and reloading round-trip equality."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "schedule.json")
        save_schedule_json(schedule, json_path)

        reloaded_schedule, manifest = load_schedule_json(json_path)

        assert len(reloaded_schedule) == len(schedule)
        assert reloaded_schedule == schedule
        assert manifest["generated_episodes"] == 120
        assert manifest["schedule_hash"] == compute_schedule_hash(schedule)


def test_csv_round_trip():
    """Validates CSV schedule saving and reloading round-trip equality."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "schedule.csv")
        save_schedule_csv(schedule, csv_path)

        reloaded_schedule = load_schedule_csv(csv_path)

        assert len(reloaded_schedule) == len(schedule)
        assert reloaded_schedule == schedule


def test_loaded_schedule_validation_guards():
    """Verifies that loading JSON/CSV files containing invalid mutation seeds, L6, or invalid workflows raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Invalid mutation_seed=999 in JSON
        json_invalid_seed = os.path.join(tmp_dir, "invalid_seed.json")
        data_seed = {
            "episodes": [{
                "episode_id": 0,
                "workflow": "LOGIN",
                "mutation_level": "L0",
                "mutation_seed": 999,
                "environment_seed": 100,
                "generation_version": "phase11-v1",
            }]
        }
        with open(json_invalid_seed, "w", encoding="utf-8") as f:
            json.dump(data_seed, f)

        with pytest.raises(ValueError, match="Invalid mutation_seed"):
            load_schedule_json(json_invalid_seed, mode="train")

        # 2. L6 in training mode JSON
        json_l6 = os.path.join(tmp_dir, "l6_training.json")
        data_l6 = {
            "episodes": [{
                "episode_id": 0,
                "workflow": "LOGIN",
                "mutation_level": "L6",
                "mutation_seed": 11,
                "environment_seed": 100,
                "generation_version": "phase11-v1",
            }]
        }
        with open(json_l6, "w", encoding="utf-8") as f:
            json.dump(data_l6, f)

        with pytest.raises(ValueError, match="EXPERIMENTAL INTEGRITY VIOLATION"):
            load_schedule_json(json_l6, mode="train")

        # 3. Invalid workflow in CSV
        csv_invalid_wf = os.path.join(tmp_dir, "invalid_wf.csv")
        with open(csv_invalid_wf, "w", encoding="utf-8") as f:
            f.write("episode_id,workflow,mutation_level,mutation_seed,environment_seed,generation_version\n")
            f.write("0,BAD_WORKFLOW,L0,11,100,phase11-v1\n")

        with pytest.raises(ValueError, match="Invalid workflow 'BAD_WORKFLOW'"):
            load_schedule_csv(csv_invalid_wf, mode="train")


def test_schedule_hash_stability():
    """Verifies that schedule SHA-256 hash is deterministic and changes when specs change."""
    generator_a = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    generator_b = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))

    sched_a = generator_a.generate(120)
    sched_b = generator_b.generate(120)

    hash_a = compute_schedule_hash(sched_a)
    hash_b = compute_schedule_hash(sched_b)

    assert hash_a == hash_b
    assert len(hash_a) == 64  # SHA-256 hex digest length

    # Mutate a single field
    mutated_sched = list(sched_a)
    first_spec = mutated_sched[0]
    mutated_sched[0] = first_spec.__class__(
        episode_id=first_spec.episode_id,
        workflow=first_spec.workflow,
        mutation_level=first_spec.mutation_level,
        mutation_seed=first_spec.mutation_seed,
        environment_seed=first_spec.environment_seed + 1,
        generation_version=first_spec.generation_version,
    )

    hash_mutated = compute_schedule_hash(mutated_sched)
    assert hash_a != hash_mutated


def test_manifest_and_coverage_report():
    """Validates structure and correct reporting in manifest and coverage dictionaries."""
    config = EpisodeGeneratorConfig(schedule_seed=42)
    generator = EpisodeGenerator(config)
    schedule = generator.generate(240)

    manifest = generate_schedule_manifest(schedule, config)
    coverage = generate_coverage_report(schedule)

    assert manifest["requested_episodes"] == 240
    assert manifest["generated_episodes"] == 240
    assert manifest["schedule_seed"] == 42
    assert manifest["coverage"]["has_held_out_l6"] is False

    assert coverage["total_episodes"] == 240
    assert coverage["unique_base_configs"] == 120
    assert coverage["duplicate_episode_ids"] == 0
    assert coverage["has_held_out_l6"] is False


def test_private_metadata_absence_audit():
    """Audits EpisodeSpec and serialized outputs to guarantee zero private metadata key leaks."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(50)

    for spec in schedule:
        spec_dict = spec.to_dict()
        for key in PROHIBITED_LEAK_KEYS:
            assert key not in spec_dict

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "audit_schedule.json")
        save_schedule_json(schedule, json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        for key in PROHIBITED_LEAK_KEYS:
            assert key not in raw_content
