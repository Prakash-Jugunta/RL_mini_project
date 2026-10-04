"""
Phase 12 — Split Materialization Engine

Generates canonical reference materializations for Train, Validation, Test-ID, and Test-L6 splits.
"""

import os
import json
from typing import Dict, List, Optional

from rl.training.episode_spec import (
    EpisodeSpec,
    VALID_WORKFLOWS,
    TRAINING_MUTATION_LEVELS,
    HELD_OUT_MUTATION_LEVEL,
    VALID_MUTATION_SEEDS,
)
from rl.training.episode_generator import _derive_seed
from rl.training.schedule_io import (
    save_schedule_json,
    compute_schedule_hash,
    generate_coverage_report,
)
from rl.splits.config import SplitConfig, SPLIT_VERSION
from rl.splits.splitter import assign_split


def materialize_canonical_splits(
    config: Optional[SplitConfig] = None
) -> Dict[str, List[EpisodeSpec]]:
    """
    Generates canonical reference materializations for Phase 12 splits.

    Canonical Strategy:
    - 120 L0-L5 strata (4 workflows × 6 levels × 5 seeds)
    - 10 environment realizations per stratum (1200 total L0-L5 episodes)
      -> Partitioned 80/10/10: 960 TRAIN, 120 VALIDATION, 120 TEST-ID
    - 20 L6 base configs (4 workflows × 5 seeds)
    - 5 environment realizations per L6 config
      -> 100 TEST-L6 episodes
    """
    cfg = config or SplitConfig()

    train_specs: List[EpisodeSpec] = []
    val_specs: List[EpisodeSpec] = []
    test_id_specs: List[EpisodeSpec] = []
    test_l6_specs: List[EpisodeSpec] = []

    global_ep_id = 0

    # 1. Process L0-L5 Strata
    for w in VALID_WORKFLOWS:
        for lvl in TRAINING_MUTATION_LEVELS:
            for m_seed in VALID_MUTATION_SEEDS:
                # 10 environment realizations per stratum
                for rep in range(10):
                    env_seed = _derive_seed(
                        cfg.split_seed,
                        f"stratum_env:{w}:{lvl}:{m_seed}",
                        rep
                    )
                    spec = EpisodeSpec(
                        episode_id=global_ep_id,
                        workflow=w,
                        mutation_level=lvl,
                        mutation_seed=m_seed,
                        environment_seed=env_seed,
                        generation_version=cfg.version,
                    )
                    global_ep_id += 1

                    split = assign_split(spec, cfg)
                    if split == "train":
                        train_specs.append(spec)
                    elif split == "validation":
                        val_specs.append(spec)
                    elif split == "test_id":
                        test_id_specs.append(spec)
                    else:
                        raise RuntimeError(f"Unexpected split '{split}' for L0-L5 spec {spec}")

    # 2. Process L6 Held-Out Strata
    for w in VALID_WORKFLOWS:
        for m_seed in VALID_MUTATION_SEEDS:
            for rep in range(cfg.l6_environment_repeats):
                env_seed = _derive_seed(
                    cfg.split_seed,
                    f"l6_env:{w}:{HELD_OUT_MUTATION_LEVEL}:{m_seed}",
                    rep
                )
                spec = EpisodeSpec(
                    episode_id=global_ep_id,
                    workflow=w,
                    mutation_level=HELD_OUT_MUTATION_LEVEL,
                    mutation_seed=m_seed,
                    environment_seed=env_seed,
                    generation_version=cfg.version,
                )
                global_ep_id += 1

                split = assign_split(spec, cfg)
                if split != "test_l6":
                    raise RuntimeError(f"L6 spec assigned to invalid split '{split}', expected 'test_l6'")
                test_l6_specs.append(spec)

    return {
        "train": train_specs,
        "validation": val_specs,
        "test_id": test_id_specs,
        "test_l6": test_l6_specs,
    }


def save_materialized_splits(
    splits_dict: Dict[str, List[EpisodeSpec]],
    output_dir: str,
    config: Optional[SplitConfig] = None
) -> Dict[str, str]:
    """
    Saves materialized split schedules and split manifest to output_dir.
    Returns dictionary mapping split names to saved file paths.
    """
    cfg = config or SplitConfig()
    os.makedirs(output_dir, exist_ok=True)

    file_paths = {}

    split_file_map = {
        "train": "train_reference.json",
        "validation": "validation.json",
        "test_id": "test_id.json",
        "test_l6": "test_l6.json",
    }

    manifest_split_summaries = {}

    for split_name, filename in split_file_map.items():
        specs = splits_dict.get(split_name, [])
        path = os.path.join(output_dir, filename)
        save_schedule_json(specs, path)
        file_paths[split_name] = path

        manifest_split_summaries[split_name] = {
            "episode_count": len(specs),
            "schedule_hash": compute_schedule_hash(specs),
            "coverage": generate_coverage_report(specs),
        }

    manifest_data = {
        "split_version": cfg.version,
        "split_seed": cfg.split_seed,
        "train_ratio": cfg.train_ratio,
        "validation_ratio": cfg.validation_ratio,
        "test_ratio": cfg.test_ratio,
        "l6_environment_repeats": cfg.l6_environment_repeats,
        "splits": manifest_split_summaries,
    }

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    file_paths["manifest"] = manifest_path
    return file_paths
