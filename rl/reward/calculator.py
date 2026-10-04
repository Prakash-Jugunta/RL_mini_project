from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from rl.reward.config import RewardConfig, REWARD_VERSION


@dataclass(frozen=True)
class RewardBreakdown:
    """
    Structured breakdown of Phase 8 reward components for diagnostics and evaluation.
    """
    step_cost: float
    progress: float
    wrong_action: float
    invalid_action: float
    completion: float
    failure: float

    @property
    def total(self) -> float:
        return (
            self.step_cost
            + self.progress
            + self.wrong_action
            + self.invalid_action
            + self.completion
            + self.failure
        )

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


class RewardCalculator:
    """
    Centralized Phase 8 Reward Calculator.

    Calculates deterministic, outcome-based rewards without access to raw DOM elements,
    embeddings, objective text, or candidate index positions.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def calculate(
        self,
        *,
        action_valid: bool,
        step_success: bool,
        workflow_completed: bool,
        truncated: bool
    ) -> RewardBreakdown:
        """
        Calculates the per-step reward breakdown based strictly on environment transition outcomes.
        """
        step_cost = self.config.step_cost
        progress = 0.0
        wrong_action = 0.0
        invalid_action = 0.0
        completion = 0.0
        failure = 0.0

        if not action_valid:
            invalid_action = self.config.invalid_action_penalty
        elif not step_success:
            wrong_action = self.config.wrong_action_penalty
        else:
            progress = self.config.progress_reward

        if workflow_completed:
            completion = self.config.completion_bonus

        if truncated and not workflow_completed:
            failure = self.config.failure_penalty

        return RewardBreakdown(
            step_cost=step_cost,
            progress=progress,
            wrong_action=wrong_action,
            invalid_action=invalid_action,
            completion=completion,
            failure=failure
        )
