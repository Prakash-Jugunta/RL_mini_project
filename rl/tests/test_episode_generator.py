"""
Phase 11 — Episode Generator Unit Tests

Validates configuration universe, L6 prohibition guards, workflow/level/seed coverage,
block shuffling, determinism, prefix stability, random access, environment seed uniqueness,
and complete config & spec validation.
"""

from collections import Counter
import pytest

from rl.training.episode_spec import EpisodeSpec, validate_episode_spec
from rl.training.episode_generator import (
    EpisodeGenerator,
    EpisodeGeneratorConfig,
    VALID_WORKFLOWS,
    TRAINING_MUTATION_LEVELS,
    VALID_MUTATION_SEEDS,
)


def test_eligible_configuration_universe():
    """Verifies that the eligible base configuration space equals 4 × 6 × 5 = 120."""
    config = EpisodeGeneratorConfig()
    generator = EpisodeGenerator(config)
    assert len(generator.base_configurations) == 120
    assert generator.block_size == 120


def test_invalid_mutation_seed_rejected_in_spec():
    """Asserts that 11,22,33,44,55 are accepted while 999 and 0 are rejected by validate_episode_spec."""
    for valid_seed in (11, 22, 33, 44, 55):
        spec = EpisodeSpec(
            episode_id=0,
            workflow="LOGIN",
            mutation_level="L0",
            mutation_seed=valid_seed,
            environment_seed=100,
        )
        validate_episode_spec(spec, mode="train")

    for invalid_seed in (999, 0, -1, 100):
        spec = EpisodeSpec(
            episode_id=0,
            workflow="LOGIN",
            mutation_level="L0",
            mutation_seed=invalid_seed,
            environment_seed=100,
        )
        with pytest.raises(ValueError, match="Invalid mutation_seed"):
            validate_episode_spec(spec, mode="train")


def test_invalid_generator_config_validations():
    """Validates complete input bounds enforcement in EpisodeGeneratorConfig.__post_init__."""
    with pytest.raises(ValueError, match="Invalid workflow 'INVALID_WF'"):
        EpisodeGeneratorConfig(workflows=("INVALID_WF",))

    with pytest.raises(ValueError, match="Invalid mutation_seed '999'"):
        EpisodeGeneratorConfig(mutation_seeds=(999,))

    with pytest.raises(ValueError, match="Invalid mode 'unknown_mode'"):
        EpisodeGeneratorConfig(mode="unknown_mode")

    with pytest.raises(ValueError, match="Unsupported sampling_strategy 'custom'"):
        EpisodeGeneratorConfig(sampling_strategy="custom")

    with pytest.raises(ValueError, match="schedule_seed must be an integer"):
        EpisodeGeneratorConfig(schedule_seed="not_an_int")  # type: ignore


def test_l6_prohibited_for_training():
    """Asserts that attempting to include held-out Level 6 in training config or spec raises ValueError."""
    with pytest.raises(ValueError, match="HELD OUT and CANNOT be included in a training generator config"):
        EpisodeGeneratorConfig(mutation_levels=("L0", "L6"), mode="train")

    spec = EpisodeSpec(
        episode_id=0,
        workflow="LOGIN",
        mutation_level="L6",
        mutation_seed=11,
        environment_seed=100,
    )
    with pytest.raises(ValueError, match="EXPERIMENTAL INTEGRITY VIOLATION"):
        validate_episode_spec(spec, mode="train")


def test_eval_mode_allows_l6():
    """Validates that eval mode allows L6 in generator config and spec validation."""
    config = EpisodeGeneratorConfig(mutation_levels=("L6",), mode="eval")
    gen = EpisodeGenerator(config)
    spec = gen.get_episode(0)
    assert spec.mutation_level == "L6"
    validate_episode_spec(spec, mode="eval")


def test_workflow_coverage():
    """Validates that one full 120-episode block contains exactly equal workflow representation (30 each)."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)
    wf_counts = Counter(spec.workflow for spec in schedule)

    for wf in VALID_WORKFLOWS:
        assert wf_counts[wf] == 30


def test_level_coverage():
    """Validates that one full 120-episode block contains exactly equal level representation (20 each)."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)
    lvl_counts = Counter(spec.mutation_level for spec in schedule)

    for lvl in TRAINING_MUTATION_LEVELS:
        assert lvl_counts[lvl] == 20


def test_mutation_seed_coverage():
    """Validates that one full 120-episode block contains exactly equal mutation seed representation (24 each)."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)
    seed_counts = Counter(spec.mutation_seed for spec in schedule)

    for seed in VALID_MUTATION_SEEDS:
        assert seed_counts[seed] == 24


def test_block_shuffling():
    """Confirms that the generated block is deterministically shuffled and not plain lexicographical order."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(120)
    generated_tuples = [(s.workflow, s.mutation_level, s.mutation_seed) for s in schedule]

    assert generated_tuples != generator.base_configurations


def test_same_seed_reproducibility():
    """Asserts that generators initialized with identical seeds produce identical EpisodeSpec lists."""
    gen_a = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    gen_b = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))

    sched_a = gen_a.generate(240)
    sched_b = gen_b.generate(240)

    assert sched_a == sched_b


def test_different_seed_variation():
    """Asserts that different schedule seeds produce distinct episode orderings and environment seeds."""
    gen_a = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    gen_b = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=999))

    sched_a = gen_a.generate(120)
    sched_b = gen_b.generate(120)

    assert sched_a != sched_b


def test_prefix_stability():
    """Validates prefix stability: generate(50) == generate(500)[:50]."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=123))
    short_sched = generator.generate(50)
    long_sched = generator.generate(500)

    assert short_sched == long_sched[:50]


def test_random_access():
    """Verifies that get_episode(N) matches the N-th element of generate(N+5)."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=777))
    full_sched = generator.generate(150)

    for idx in [0, 10, 119, 120, 137, 149]:
        spec_direct = generator.get_episode(idx)
        assert spec_direct == full_sched[idx]


def test_block_repeats_with_new_environment_seeds():
    """Validates that repeated base configurations across blocks receive distinct environment seeds."""
    generator = EpisodeGenerator(EpisodeGeneratorConfig(schedule_seed=42))
    schedule = generator.generate(240)  # 2 full blocks

    block_0_spec = schedule[0]
    # Find matching base config in second block
    matching_specs_block_1 = [
        spec for spec in schedule[120:]
        if (spec.workflow, spec.mutation_level, spec.mutation_seed) ==
           (block_0_spec.workflow, block_0_spec.mutation_level, block_0_spec.mutation_seed)
    ]

    assert len(matching_specs_block_1) == 1
    block_1_spec = matching_specs_block_1[0]

    assert block_0_spec.episode_id != block_1_spec.episode_id
    assert block_0_spec.environment_seed != block_1_spec.environment_seed
