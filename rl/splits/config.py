"""
Phase 12 — Split Configuration

Defines SplitConfig dataclass and validation rules for Phase 12 split partitioning.
"""

from dataclasses import dataclass

SPLIT_VERSION: str = "phase12-v1"


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration specifying the train / validation / test partitioning protocol.

    Fields:
        split_seed: Independent base seed for deterministic split assignment (default 1201).
        train_ratio: Target proportion of L0-L5 environment realizations for training (0.8).
        validation_ratio: Target proportion for validation (0.1).
        test_ratio: Target proportion for in-distribution test (0.1).
        l6_environment_repeats: Canonical environment realizations count per L6 base config (5).
        version: Protocol version string ('phase12-v1').
    """

    split_seed: int = 1201
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    l6_environment_repeats: int = 5
    version: str = SPLIT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.split_seed, int):
            raise ValueError(f"split_seed must be an integer, got {type(self.split_seed)}")

        if self.train_ratio <= 0 or self.validation_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split ratios must be strictly positive (> 0).")

        ratio_sum = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got sum {ratio_sum} "
                f"(train={self.train_ratio}, val={self.validation_ratio}, test={self.test_ratio})"
            )

        if self.l6_environment_repeats <= 0:
            raise ValueError(f"l6_environment_repeats must be positive, got {self.l6_environment_repeats}")
