"""
Phase 11 — Training Episode Specification & Validation

Defines the EpisodeSpec dataclass and validation rules for deterministic,
reproducible, leak-free RL training episode generation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

EPISODE_GENERATOR_VERSION: str = "phase11-v1"

VALID_WORKFLOWS: Tuple[str, ...] = ("LOGIN", "SEARCH", "PROFILE", "CHECKOUT")
TRAINING_MUTATION_LEVELS: Tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")
HELD_OUT_MUTATION_LEVEL: str = "L6"
VALID_MUTATION_SEEDS: Tuple[int, ...] = (11, 22, 33, 44, 55)


@dataclass(frozen=True)
class EpisodeSpec:
    """
    Immutably specifies a single RL training or evaluation episode.

    Fields:
        episode_id: Zero-indexed global episode sequence identifier.
        workflow: Primary workflow name ('LOGIN', 'SEARCH', 'PROFILE', 'CHECKOUT').
        mutation_level: Mutation level string ('L0', 'L1', 'L2', 'L3', 'L4', 'L5').
        mutation_seed: UI mutation framework seed (11, 22, 33, 44, 55).
        environment_seed: Gymnasium environment candidate-ordering seed.
        generation_version: Generator version tag ('phase11-v1').
    """

    episode_id: int
    workflow: str
    mutation_level: str
    mutation_seed: int
    environment_seed: int
    generation_version: str = EPISODE_GENERATOR_VERSION

    def numeric_mutation_level(self) -> int:
        """Converts mutation level string (e.g. 'L3') to numeric integer (e.g. 3)."""
        clean_lvl = str(self.mutation_level).strip().upper()
        if clean_lvl.startswith("L") and clean_lvl[1:].isdigit():
            return int(clean_lvl[1:])
        if clean_lvl.isdigit():
            return int(clean_lvl)
        raise ValueError(f"Invalid mutation_level format: '{self.mutation_level}'")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes EpisodeSpec to a standard dictionary."""
        return {
            "episode_id": self.episode_id,
            "workflow": self.workflow,
            "mutation_level": self.mutation_level,
            "mutation_seed": self.mutation_seed,
            "environment_seed": self.environment_seed,
            "generation_version": self.generation_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeSpec":
        """Deserializes EpisodeSpec from a dictionary."""
        return cls(
            episode_id=int(data["episode_id"]),
            workflow=str(data["workflow"]),
            mutation_level=str(data["mutation_level"]),
            mutation_seed=int(data["mutation_seed"]),
            environment_seed=int(data["environment_seed"]),
            generation_version=str(data.get("generation_version", EPISODE_GENERATOR_VERSION)),
        )

    def to_reset_options(self, mode: str = "train") -> Dict[str, Any]:
        """
        Formats Gym environment reset options dictionary.
        """
        return {
            "workflow_id": self.workflow,
            "mode": mode,
            "mutation_level": self.numeric_mutation_level(),
            "mutation_seed": self.mutation_seed,
        }


def validate_episode_spec(spec: EpisodeSpec, mode: str = "train") -> None:
    """
    Strictly validates an EpisodeSpec against domain rules and experimental guards.
    Raises ValueError on any boundary violation.
    """
    if spec.episode_id < 0:
        raise ValueError(f"Episode ID must be non-negative, got {spec.episode_id}")

    if spec.workflow not in VALID_WORKFLOWS:
        raise ValueError(
            f"Invalid workflow '{spec.workflow}'. Must be one of {VALID_WORKFLOWS}"
        )

    clean_level = str(spec.mutation_level).strip().upper()
    if not clean_level.startswith("L"):
        clean_level = f"L{clean_level}"

    if mode == "train" and clean_level == HELD_OUT_MUTATION_LEVEL:
        raise ValueError(
            f"EXPERIMENTAL INTEGRITY VIOLATION: Level '{HELD_OUT_MUTATION_LEVEL}' is HELD OUT "
            f"and CANNOT be generated or used in training mode (spec ID {spec.episode_id})."
        )

    if clean_level not in (*TRAINING_MUTATION_LEVELS, HELD_OUT_MUTATION_LEVEL):
        raise ValueError(
            f"Invalid mutation_level '{spec.mutation_level}'. "
            f"Allowed: {(*TRAINING_MUTATION_LEVELS, HELD_OUT_MUTATION_LEVEL)}"
        )

    if not isinstance(spec.mutation_seed, int):
        raise ValueError(
            f"Mutation seed must be an integer, got {type(spec.mutation_seed)}"
        )

    if spec.mutation_seed not in VALID_MUTATION_SEEDS:
        raise ValueError(
            f"Invalid mutation_seed {spec.mutation_seed}. Must be one of {VALID_MUTATION_SEEDS}"
        )

    if not isinstance(spec.environment_seed, int):
        raise ValueError(
            f"Environment seed must be an integer, got {type(spec.environment_seed)}"
        )
