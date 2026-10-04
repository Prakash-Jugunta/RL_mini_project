"""
Phase 13 — Curriculum Scheduler Unit Tests

Validates progressive stage boundaries, 100% TRAIN split membership, zero L6,
intra-stage balancing, determinism, prefix stability, and random access.
"""

from collections import Counter
import pytest

from rl.training.episode_spec import HELD_OUT_MUTATION_LEVEL
from rl.splits.config import SplitConfig
from rl.splits.splitter import assign_split
from rl.curriculum.config import CurriculumConfig
from rl.curriculum.scheduler import CurriculumScheduler, CurriculumEpisode
from rl.curriculum.validator import validate_curriculum_integrity


def test_curriculum_stage_boundaries_and_indexing():
    """Validates global episode ID to stage index mapping."""
    config = CurriculumConfig()
    scheduler = CurriculumScheduler(config)

    # Stage 0: 0 .. 999
    s_idx, ep_idx, levels = scheduler.get_stage_info(0)
    assert (s_idx, ep_idx, levels) == (0, 0, ("L0",))

    s_idx, ep_idx, levels = scheduler.get_stage_info(999)
    assert (s_idx, ep_idx, levels) == (0, 999, ("L0",))

    # Stage 1: 1000 .. 2499
    s_idx, ep_idx, levels = scheduler.get_stage_info(1000)
    assert (s_idx, ep_idx, levels) == (1, 0, ("L0", "L1"))

    # Stage 5: 10000 .. 14999
    s_idx, ep_idx, levels = scheduler.get_stage_info(10000)
    assert (s_idx, ep_idx, levels) == (5, 0, ("L0", "L1", "L2", "L3", "L4", "L5"))


def test_100_percent_train_split_membership_and_zero_l6():
    """Asserts that 100% of curriculum episodes belong to Phase 12 TRAIN split and zero L6 episodes exist."""
    config = CurriculumConfig(
        curriculum_seed=1301,
        stage_episode_counts=(100, 100, 100, 100, 100, 100)  # Small test budget
    )
    split_config = SplitConfig(split_seed=1201)
    scheduler = CurriculumScheduler(config, split_config)
    schedule = scheduler.generate_schedule(600)

    validate_curriculum_integrity(schedule, config, split_config)

    for ep in schedule:
        spec = ep.episode_spec
        split_name = assign_split(spec, split_config)
        assert split_name == "train"
        assert str(spec.mutation_level).upper() != HELD_OUT_MUTATION_LEVEL


def test_same_seed_reproducibility():
    """Asserts that schedulers with identical seeds produce identical CurriculumEpisode schedules."""
    config = CurriculumConfig(
        curriculum_seed=1301,
        stage_episode_counts=(50, 50, 50, 50, 50, 50)
    )
    sched_a = CurriculumScheduler(config).generate_schedule(300)
    sched_b = CurriculumScheduler(config).generate_schedule(300)

    assert sched_a == sched_b


def test_different_seed_variation():
    """Asserts that different curriculum seeds yield distinct environment seed selections."""
    cfg_a = CurriculumConfig(curriculum_seed=1301, stage_episode_counts=(50, 50, 50, 50, 50, 50))
    cfg_b = CurriculumConfig(curriculum_seed=9999, stage_episode_counts=(50, 50, 50, 50, 50, 50))

    sched_a = CurriculumScheduler(cfg_a).generate_schedule(300)
    sched_b = CurriculumScheduler(cfg_b).generate_schedule(300)

    assert sched_a != sched_b


def test_prefix_stability():
    """Validates prefix stability: generate(100) == generate(500)[:100]."""
    config = CurriculumConfig(curriculum_seed=1301)
    scheduler = CurriculumScheduler(config)

    short_sched = scheduler.generate_schedule(100)
    long_sched = scheduler.generate_schedule(500)

    assert short_sched == long_sched[:100]


def test_random_access():
    """Verifies that get_episode(N) matches N-th element of generate_schedule(N+5)."""
    config = CurriculumConfig(curriculum_seed=1301)
    scheduler = CurriculumScheduler(config)

    full_sched = scheduler.generate_schedule(300)

    for idx in [0, 50, 100, 137, 250, 299]:
        direct_ep = scheduler.get_episode(idx)
        assert direct_ep == full_sched[idx]
