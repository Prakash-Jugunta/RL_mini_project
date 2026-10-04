"""
Phase 13 — Curriculum Scheduler

Implements progressive mixed curriculum scheduling with random access, prefix stability,
deterministic rejection sampling for TRAIN split membership, and anti-forgetting level retention.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from rl.training.episode_spec import (
    EpisodeSpec,
    validate_episode_spec,
    VALID_WORKFLOWS,
    VALID_MUTATION_SEEDS,
)
from rl.training.episode_generator import _derive_seed
from rl.splits.config import SplitConfig
from rl.splits.splitter import assign_split
from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION


@dataclass(frozen=True)
class CurriculumEpisode:
    """
    Immutably specifies a single training episode within a curriculum progression.

    Fields:
        global_episode_id: Zero-indexed global training episode counter.
        stage_index: Current curriculum stage index (0..5).
        stage_episode_index: Zero-indexed episode counter within current stage.
        episode_spec: Underlying EpisodeSpec for Gym environment reset.
        active_levels: Tuple of active mutation levels in current stage.
        curriculum_version: Curriculum version string ('phase13-v1').
    """

    global_episode_id: int
    stage_index: int
    stage_episode_index: int
    episode_spec: EpisodeSpec
    active_levels: Tuple[str, ...]
    curriculum_version: str = CURRICULUM_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serializes CurriculumEpisode to dictionary."""
        d = asdict(self)
        d["episode_spec"] = self.episode_spec.to_dict()
        return d


class CurriculumScheduler:
    """
    Schedules training episodes across progressive mixed complexity stages.

    Features:
    - Progressive mixed level introduction (Stage 0: L0 -> Stage 5: L0-L5)
    - Anti-forgetting: retains all prior levels in subsequent stages
    - 100% TRAIN split partition enforcement via deterministic rejection sampling
    - Pure, random-accessible get_episode(global_episode_id) lookup
    - Prefix-stable schedule generation
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        split_config: Optional[SplitConfig] = None
    ) -> None:
        self.config = config or CurriculumConfig()
        self.split_config = split_config or SplitConfig()

        self.cumulative_counts: List[int] = []
        running_total = 0
        for count in self.config.stage_episode_counts:
            running_total += count
            self.cumulative_counts.append(running_total)

    def get_stage_info(self, global_episode_id: int) -> Tuple[int, int, Tuple[str, ...]]:
        """
        Determines stage_index, stage_episode_index, and active_levels for global_episode_id.
        """
        if global_episode_id < 0:
            raise ValueError(f"global_episode_id must be non-negative, got {global_episode_id}")

        stage_idx = 0
        for idx, cum_count in enumerate(self.cumulative_counts):
            if global_episode_id < cum_count:
                stage_idx = idx
                break
        else:
            # Beyond canonical episode budget: remain in final Stage (Stage 5)
            stage_idx = len(self.config.stage_episode_counts) - 1

        stage_start = 0 if stage_idx == 0 else self.cumulative_counts[stage_idx - 1]
        stage_ep_idx = global_episode_id - stage_start
        active_levels = self.config.active_levels_by_stage[stage_idx]

        return stage_idx, stage_ep_idx, active_levels

    def _get_shuffled_stage_block(
        self,
        stage_index: int,
        block_index: int,
        active_levels: Tuple[str, ...]
    ) -> List[Tuple[str, str, int]]:
        """Returns a deterministically shuffled base block for a given stage and block index."""
        base_combos: List[Tuple[str, str, int]] = []
        for w in VALID_WORKFLOWS:
            for lvl in active_levels:
                clean_lvl = str(lvl).strip().upper()
                if not clean_lvl.startswith("L"):
                    clean_lvl = f"L{clean_lvl}"
                for m_seed in VALID_MUTATION_SEEDS:
                    base_combos.append((w, clean_lvl, int(m_seed)))

        block_seed = _derive_seed(
            self.config.curriculum_seed,
            f"stage_{stage_index}_block",
            block_index
        )
        rng = np.random.default_rng(block_seed)
        block_copy = list(base_combos)
        rng.shuffle(block_copy)
        return block_copy

    def get_episode(self, global_episode_id: int) -> CurriculumEpisode:
        """
        Retrieves the exact CurriculumEpisode for global_episode_id.
        Stateless, deterministic, and random-accessible.
        """
        stage_idx, stage_ep_idx, active_levels = self.get_stage_info(global_episode_id)

        # Base combinations for current stage
        block_size = len(VALID_WORKFLOWS) * len(active_levels) * len(VALID_MUTATION_SEEDS)
        block_index = stage_ep_idx // block_size
        pos_in_block = stage_ep_idx % block_size

        shuffled_block = self._get_shuffled_stage_block(stage_idx, block_index, active_levels)
        workflow, mutation_level, mutation_seed = shuffled_block[pos_in_block]

        # Deterministic Rejection Sampling for TRAIN split environment seed
        accepted_spec: Optional[EpisodeSpec] = None
        for attempt in range(100):
            env_seed = _derive_seed(
                self.config.curriculum_seed,
                f"curr_env:{global_episode_id}",
                attempt
            )
            candidate_spec = EpisodeSpec(
                episode_id=global_episode_id,
                workflow=workflow,
                mutation_level=mutation_level,
                mutation_seed=mutation_seed,
                environment_seed=env_seed,
                generation_version="phase11-v1",
            )
            validate_episode_spec(candidate_spec, mode="train")

            split = assign_split(candidate_spec, self.split_config)
            if split == "train":
                accepted_spec = candidate_spec
                break

        if accepted_spec is None:
            raise RuntimeError(f"Failed to find a TRAIN split environment seed for episode {global_episode_id}")

        return CurriculumEpisode(
            global_episode_id=global_episode_id,
            stage_index=stage_idx,
            stage_episode_index=stage_ep_idx,
            episode_spec=accepted_spec,
            active_levels=active_levels,
            curriculum_version=self.config.version,
        )

    def generate_schedule(self, n_episodes: int, start_episode_id: int = 0) -> List[CurriculumEpisode]:
        """
        Generates a list of n_episodes CurriculumEpisodes starting from start_episode_id.
        Guarantees prefix stability.
        """
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {n_episodes}")

        schedule: List[CurriculumEpisode] = []
        for i in range(n_episodes):
            ep_id = start_episode_id + i
            schedule.append(self.get_episode(ep_id))
        return schedule

    def total_episodes(self) -> int:
        """Returns total episode count in canonical curriculum config."""
        return self.config.total_episodes()
