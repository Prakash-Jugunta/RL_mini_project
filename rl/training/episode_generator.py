"""
Phase 11 — Training Episode Generator

Implements deterministic, balanced-block episode schedule generation with random access,
prefix stability, explicit seed separation, and hard guards against held-out L6 contamination.
"""

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from rl.training.episode_spec import (
    EpisodeSpec,
    validate_episode_spec,
    EPISODE_GENERATOR_VERSION,
    VALID_WORKFLOWS,
    TRAINING_MUTATION_LEVELS,
    HELD_OUT_MUTATION_LEVEL,
    VALID_MUTATION_SEEDS,
)


def _derive_seed(schedule_seed: int, key: str, index: int) -> int:
    """
    Derives a deterministic 32-bit positive integer seed from a base seed, key, and index.
    Uses SHA-256 for uniform hash distribution without stateful RNG dependencies.
    """
    raw_str = f"{schedule_seed}:{key}:{index}".encode("utf-8")
    hash_bytes = hashlib.sha256(raw_str).digest()
    (val,) = struct.unpack(">I", hash_bytes[:4])
    return int(val % 2147483647)  # Bound to [0, 2^31 - 2] for Gym compatibility


@dataclass(frozen=True)
class EpisodeGeneratorConfig:
    """
    Configuration for EpisodeGenerator.

    Fields:
        workflows: Tuple of workflow strings ('LOGIN', 'SEARCH', 'PROFILE', 'CHECKOUT').
        mutation_levels: Tuple of level strings ('L0' .. 'L5').
        mutation_seeds: Tuple of UI mutation framework seeds (11, 22, 33, 44, 55).
        schedule_seed: Base seed controlling episode schedule generation order.
        mode: Operation mode ('train' or 'eval').
        sampling_strategy: Sampling strategy name ('balanced_block').
        version: Generator version tag ('phase11-v1').
    """

    workflows: Tuple[str, ...] = VALID_WORKFLOWS
    mutation_levels: Tuple[str, ...] = TRAINING_MUTATION_LEVELS
    mutation_seeds: Tuple[int, ...] = VALID_MUTATION_SEEDS
    schedule_seed: int = 42
    mode: str = "train"
    sampling_strategy: str = "balanced_block"
    version: str = EPISODE_GENERATOR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schedule_seed, int):
            raise ValueError(f"schedule_seed must be an integer, got {type(self.schedule_seed)}")

        if self.mode not in ("train", "eval"):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'train' or 'eval'.")

        if self.sampling_strategy != "balanced_block":
            raise ValueError(f"Unsupported sampling_strategy '{self.sampling_strategy}'. Must be 'balanced_block'.")

        if not self.workflows:
            raise ValueError("workflows configuration cannot be empty.")
        for w in self.workflows:
            if w not in VALID_WORKFLOWS:
                raise ValueError(f"Invalid workflow '{w}' in EpisodeGeneratorConfig. Must be one of {VALID_WORKFLOWS}")

        if not self.mutation_levels:
            raise ValueError("mutation_levels configuration cannot be empty.")
        valid_allowed_levels = TRAINING_MUTATION_LEVELS if self.mode == "train" else (*TRAINING_MUTATION_LEVELS, HELD_OUT_MUTATION_LEVEL)
        for lvl in self.mutation_levels:
            clean_lvl = str(lvl).strip().upper()
            if not clean_lvl.startswith("L"):
                clean_lvl = f"L{clean_lvl}"
            if self.mode == "train" and clean_lvl == HELD_OUT_MUTATION_LEVEL:
                raise ValueError(
                    f"EXPERIMENTAL INTEGRITY VIOLATION: Level '{HELD_OUT_MUTATION_LEVEL}' "
                    f"is HELD OUT and CANNOT be included in a training generator config."
                )
            if clean_lvl not in valid_allowed_levels:
                raise ValueError(
                    f"Invalid mutation_level '{lvl}' in EpisodeGeneratorConfig. "
                    f"Allowed for mode '{self.mode}': {valid_allowed_levels}"
                )

        if not self.mutation_seeds:
            raise ValueError("mutation_seeds configuration cannot be empty.")
        for s in self.mutation_seeds:
            if s not in VALID_MUTATION_SEEDS:
                raise ValueError(
                    f"Invalid mutation_seed '{s}' in EpisodeGeneratorConfig. "
                    f"Must be one of {VALID_MUTATION_SEEDS}"
                )


class EpisodeGenerator:
    """
    Generates deterministic EpisodeSpec streams for RL training or evaluation.

    Features:
    - Balanced 120-configuration block coverage (4 workflows × 6 levels × 5 seeds)
    - Deterministic shuffling within blocks controlled by schedule_seed
    - Random-access episode lookup via get_episode(episode_id)
    - Prefix-stable schedule generation
    - Distinct, deterministic environment seeds derived per episode
    - Hard guard against L6 training generation
    """

    def __init__(self, config: Optional[EpisodeGeneratorConfig] = None) -> None:
        self.config = config or EpisodeGeneratorConfig()
        self.base_configurations: List[Tuple[str, str, int]] = []

        for w in self.config.workflows:
            for lvl in self.config.mutation_levels:
                clean_lvl = str(lvl).strip().upper()
                if not clean_lvl.startswith("L"):
                    clean_lvl = f"L{clean_lvl}"
                for m_seed in self.config.mutation_seeds:
                    self.base_configurations.append((w, clean_lvl, int(m_seed)))

        self.block_size = len(self.base_configurations)
        if self.block_size == 0:
            raise ValueError("EpisodeGenerator base configuration space cannot be empty.")

    def _get_shuffled_block(self, block_index: int) -> List[Tuple[str, str, int]]:
        """Returns a deterministically shuffled copy of base_configurations for block_index."""
        block_seed = _derive_seed(self.config.schedule_seed, "block_shuffle", block_index)
        rng = np.random.default_rng(block_seed)
        block_copy = list(self.base_configurations)
        rng.shuffle(block_copy)
        return block_copy

    def get_episode(self, episode_id: int) -> EpisodeSpec:
        """
        Retrieves the exact EpisodeSpec for a given zero-indexed episode_id.
        Pure function: deterministic, stateless, and random-accessible.
        """
        if episode_id < 0:
            raise ValueError(f"episode_id must be non-negative, got {episode_id}")

        block_index = episode_id // self.block_size
        position_in_block = episode_id % self.block_size

        shuffled_block = self._get_shuffled_block(block_index)
        workflow, mutation_level, mutation_seed = shuffled_block[position_in_block]

        env_seed = _derive_seed(self.config.schedule_seed, "environment_seed", episode_id)

        spec = EpisodeSpec(
            episode_id=episode_id,
            workflow=workflow,
            mutation_level=mutation_level,
            mutation_seed=mutation_seed,
            environment_seed=env_seed,
            generation_version=self.config.version,
        )

        validate_episode_spec(spec, mode=self.config.mode)
        return spec

    def generate(self, n_episodes: int, start_episode_id: int = 0) -> List[EpisodeSpec]:
        """
        Generates a list of n_episodes EpisodeSpecs starting from start_episode_id.
        Guarantees prefix stability: generate(50) == generate(500)[:50].
        """
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")

        schedule: List[EpisodeSpec] = []
        for i in range(n_episodes):
            ep_id = start_episode_id + i
            schedule.append(self.get_episode(ep_id))
        return schedule

    def validate_schedule(self, schedule: List[EpisodeSpec]) -> None:
        """
        Validates an entire generated schedule list for uniqueness, L6 exclusion, and structural validity.
        """
        seen_ids = set()
        for spec in schedule:
            if spec.episode_id in seen_ids:
                raise ValueError(f"Duplicate episode_id {spec.episode_id} found in schedule.")
            seen_ids.add(spec.episode_id)
            validate_episode_spec(spec, mode=self.config.mode)
