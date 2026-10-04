"""
Phase 12 — Split Inspection Tool

CLI utility for materializing, validating, hashing, and analyzing Phase 12 splits
without initiating any RL training or neural network updates.
"""

import argparse
import sys
from typing import Dict, List

from rl.splits.config import SplitConfig, SPLIT_VERSION
from rl.splits.splitter import assign_split
from rl.splits.materialize import materialize_canonical_splits, save_materialized_splits
from rl.splits.validator import validate_split_integrity
from rl.training.schedule_io import compute_schedule_hash


def print_split_summary(
    splits_dict: Dict[str, List],
    config: SplitConfig
) -> None:
    """Prints human-readable split inspection statistics to stdout."""
    train_specs = splits_dict["train"]
    val_specs = splits_dict["validation"]
    test_id_specs = splits_dict["test_id"]
    test_l6_specs = splits_dict["test_l6"]

    train_l6_count = sum(1 for s in train_specs if str(s.mutation_level).upper() in ("L6", "6"))
    val_l6_count = sum(1 for s in val_specs if str(s.mutation_level).upper() in ("L6", "6"))

    shared_specs_count = len(set(train_specs) & set(val_specs)) + len(set(train_specs) & set(test_id_specs))

    print("=" * 70)
    print(" PHASE 12 — TRAIN / VALIDATION / TEST SPLIT SUMMARY")
    print("=" * 70)
    print(f" Split Version:         {config.version}")
    print(f" Split Seed:            {config.split_seed}")
    print(f" Train Ratio:           {config.train_ratio * 100:.1f}%")
    print(f" Validation Ratio:      {config.validation_ratio * 100:.1f}%")
    print(f" Test-ID Ratio:         {config.test_ratio * 100:.1f}%")
    print("-" * 70)
    print(" Canonical Materialization Counts:")
    print(f"   TRAIN Reference:     {len(train_specs):>5} episodes")
    print(f"   VALIDATION:          {len(val_specs):>5} episodes")
    print(f"   TEST-ID:             {len(test_id_specs):>5} episodes")
    print(f"   TEST-L6 (Held-Out):  {len(test_l6_specs):>5} episodes")
    print("-" * 70)
    print(" Integrity Guards Verification:")
    print(f"   L6 in TRAIN:         {train_l6_count} (Must be 0)")
    print(f"   L6 in VALIDATION:    {val_l6_count} (Must be 0)")
    print(f"   Shared EpisodeSpecs: {shared_specs_count} (Must be 0)")
    print("-" * 70)
    print(" Schedule SHA-256 Fingerprint Hashes:")
    print(f"   TRAIN Hash:          {compute_schedule_hash(train_specs)}")
    print(f"   VALIDATION Hash:     {compute_schedule_hash(val_specs)}")
    print(f"   TEST-ID Hash:        {compute_schedule_hash(test_id_specs)}")
    print(f"   TEST-L6 Hash:        {compute_schedule_hash(test_l6_specs)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate Phase 12 Train/Validation/Test splits."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1201,
        help="Base seed for split assignment (default: 1201)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save materialized split JSON files and manifest."
    )

    args = parser.parse_args()

    config = SplitConfig(split_seed=args.split_seed)
    splits_dict = materialize_canonical_splits(config)
    validate_split_integrity(splits_dict, config)

    print_split_summary(splits_dict, config)

    if args.output_dir:
        saved_paths = save_materialized_splits(splits_dict, args.output_dir, config)
        print(f"\n Saved materialized split artifacts to: {args.output_dir}")
        for name, path in saved_paths.items():
            print(f"   {name:<12}: {path}")


if __name__ == "__main__":
    main()
