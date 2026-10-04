"""
Phase 11 — Schedule Inspection Tool

CLI utility for generating, validating, hashing, and analyzing EpisodeSpec schedules
without initiating any RL training or neural network updates.
"""

import argparse
import sys
from typing import List

from rl.training.episode_generator import EpisodeGenerator, EpisodeGeneratorConfig
from rl.training.episode_spec import EpisodeSpec
from rl.training.schedule_io import (
    generate_coverage_report,
    compute_schedule_hash,
    generate_schedule_manifest,
    save_schedule_json,
    save_schedule_csv,
)


def print_schedule_summary(
    schedule: List[EpisodeSpec],
    config: EpisodeGeneratorConfig,
    show_first: int = 10
) -> None:
    """Prints human-readable schedule inspection statistics to stdout."""
    coverage = generate_coverage_report(schedule)
    schedule_hash = compute_schedule_hash(schedule)

    print("=" * 70)
    print(" PHASE 11 — EPISODE SCHEDULE INSPECTION REPORT")
    print("=" * 70)
    print(f" Generator Version:    {config.version}")
    print(f" Schedule Seed:        {config.schedule_seed}")
    print(f" Sampling Strategy:    {config.sampling_strategy}")
    print(f" Total Episodes:       {len(schedule)}")
    print(f" Unique Base Configs:  {coverage['unique_base_configs']} / 120")
    print(f" Unique Env Seeds:     {coverage['unique_env_seeds']}")
    print(f" Duplicate Episode IDs: {coverage['duplicate_episode_ids']}")
    print(f" Held-Out L6 Included: {coverage['has_held_out_l6']} (Must be False)")
    print(f" Schedule SHA-256 Hash: {schedule_hash}")
    print("-" * 70)

    print(" Workflow Breakdown:")
    for wf, count in sorted(coverage["workflow_counts"].items()):
        pct = (count / len(schedule)) * 100
        print(f"   {wf:<12}: {count:>5} ({pct:>5.1f}%)")

    print("-" * 70)
    print(" Mutation Level Breakdown:")
    for lvl, count in sorted(coverage["level_counts"].items()):
        pct = (count / len(schedule)) * 100
        print(f"   {lvl:<12}: {count:>5} ({pct:>5.1f}%)")

    print("-" * 70)
    print(" Mutation Seed Breakdown:")
    for seed, count in sorted(coverage["seed_counts"].items()):
        pct = (count / len(schedule)) * 100
        print(f"   Seed {seed:<7}: {count:>5} ({pct:>5.1f}%)")

    print("=" * 70)

    if show_first > 0:
        n_show = min(show_first, len(schedule))
        print(f" First {n_show} Episode Specifications:")
        print(f" {'ID':<6} {'Workflow':<10} {'Level':<8} {'M_Seed':<8} {'Env_Seed':<12}")
        print("-" * 70)
        for spec in schedule[:n_show]:
            print(
                f" {spec.episode_id:<6} {spec.workflow:<10} {spec.mutation_level:<8} "
                f"{spec.mutation_seed:<8} {spec.environment_seed:<12}"
            )
        print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate Phase 11 Episode Generator schedules."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=240,
        help="Number of episodes to generate and inspect (default: 240)."
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=42,
        help="Base seed for episode schedule generation (default: 42)."
    )
    parser.add_argument(
        "--show-first",
        type=int,
        default=10,
        help="Number of initial episode specs to display (default: 10)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for JSON schedule serialization."
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional output path for CSV schedule serialization."
    )

    args = parser.parse_args()

    config = EpisodeGeneratorConfig(schedule_seed=args.schedule_seed)
    generator = EpisodeGenerator(config)
    schedule = generator.generate(args.episodes)
    generator.validate_schedule(schedule)

    print_schedule_summary(schedule, config, show_first=args.show_first)

    if args.output:
        manifest = generate_schedule_manifest(schedule, config)
        save_schedule_json(schedule, args.output, manifest=manifest)
        print(f" Saved JSON schedule to: {args.output}")

    if args.csv_output:
        save_schedule_csv(schedule, args.csv_output)
        print(f" Saved CSV schedule to: {args.csv_output}")


if __name__ == "__main__":
    main()
