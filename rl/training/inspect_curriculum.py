"""
Phase 13 — Curriculum Schedule Inspection Tool

CLI utility for generating, validating, hashing, and analyzing Phase 13 progressive mixed
curriculum schedules without initiating any RL training or neural network updates.
"""

import argparse
import sys
from collections import Counter
from typing import List

from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION
from rl.curriculum.scheduler import CurriculumScheduler, CurriculumEpisode
from rl.curriculum.validator import validate_curriculum_integrity
from rl.curriculum.manifest import generate_curriculum_manifest, compute_curriculum_hash
from rl.splits.config import SplitConfig


def print_curriculum_summary(
    schedule: List[CurriculumEpisode],
    config: CurriculumConfig,
    show_first: int = 10
) -> None:
    """Prints human-readable curriculum inspection statistics to stdout."""
    curr_hash = compute_curriculum_hash(schedule)

    stage_counts = Counter(ep.stage_index for ep in schedule)
    wf_counts = Counter(ep.episode_spec.workflow for ep in schedule)
    lvl_counts = Counter(ep.episode_spec.mutation_level for ep in schedule)
    seed_counts = Counter(ep.episode_spec.mutation_seed for ep in schedule)

    l6_count = sum(1 for ep in schedule if str(ep.episode_spec.mutation_level).upper() in ("L6", "6"))

    print("=" * 70)
    print(" PHASE 13 — DIFFICULTY / CURRICULUM INSPECTION REPORT")
    print("=" * 70)
    print(f" Curriculum Version:   {config.version}")
    print(f" Curriculum Seed:      {config.curriculum_seed}")
    print(f" Sampling Strategy:    {config.sampling_strategy}")
    print(f" Total Episodes:       {len(schedule)}")
    print(f" Held-Out L6 Count:    {l6_count} (Must be 0)")
    print(f" Validation Interval:  {config.validation_interval} episodes")
    print(f" Checkpoint Interval:  {config.checkpoint_interval} episodes")
    print(f" Curriculum SHA-256:   {curr_hash}")
    print("-" * 70)

    print(" Progressive Stage Breakdown:")
    for s_idx in sorted(stage_counts.keys()):
        count = stage_counts[s_idx]
        pct = (count / len(schedule)) * 100
        active = ", ".join(config.active_levels_by_stage[s_idx]) if s_idx < len(config.active_levels_by_stage) else "All"
        print(f"   Stage {s_idx} ({active:<18}): {count:>5} episodes ({pct:>5.1f}%)")

    print("-" * 70)
    print(" Workflow Breakdown:")
    for wf, count in sorted(wf_counts.items()):
        pct = (count / len(schedule)) * 100
        print(f"   {wf:<12}: {count:>5} ({pct:>5.1f}%)")

    print("-" * 70)
    print(" Active Mutation Level Breakdown:")
    for lvl, count in sorted(lvl_counts.items()):
        pct = (count / len(schedule)) * 100
        print(f"   {lvl:<12}: {count:>5} ({pct:>5.1f}%)")

    print("-" * 70)
    print(" Mutation Seed Breakdown:")
    for seed, count in sorted(seed_counts.items()):
        pct = (count / len(schedule)) * 100
        print(f"   Seed {seed:<7}: {count:>5} ({pct:>5.1f}%)")

    print("=" * 70)

    if show_first > 0:
        n_show = min(show_first, len(schedule))
        print(f" First {n_show} Curriculum Episode Specifications:")
        print(f" {'Global_ID':<10} {'Stage':<7} {'Workflow':<10} {'Level':<8} {'M_Seed':<8} {'Env_Seed':<12}")
        print("-" * 70)
        for ep in schedule[:n_show]:
            spec = ep.episode_spec
            print(
                f" {ep.global_episode_id:<10} {ep.stage_index:<7} {spec.workflow:<10} "
                f"{spec.mutation_level:<8} {spec.mutation_seed:<8} {spec.environment_seed:<12}"
            )
        print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate Phase 13 Progressive Mixed Curriculum schedules."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=15000,
        help="Number of curriculum episodes to generate and inspect (default: 15000)."
    )
    parser.add_argument(
        "--curriculum-seed",
        type=int,
        default=1301,
        help="Base seed for curriculum schedule generation (default: 1301)."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1201,
        help="Base seed for Phase 12 split membership verification (default: 1201)."
    )
    parser.add_argument(
        "--show-first",
        type=int,
        default=10,
        help="Number of initial curriculum episode specs to display (default: 10)."
    )

    args = parser.parse_args()

    config = CurriculumConfig(curriculum_seed=args.curriculum_seed)
    split_config = SplitConfig(split_seed=args.split_seed)
    scheduler = CurriculumScheduler(config=config, split_config=split_config)

    schedule = scheduler.generate_schedule(args.episodes)
    validate_curriculum_integrity(schedule, config=config, split_config=split_config)

    print_curriculum_summary(schedule, config, show_first=args.show_first)


if __name__ == "__main__":
    main()
