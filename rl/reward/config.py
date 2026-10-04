from dataclasses import dataclass

REWARD_VERSION: str = "phase8-v1"


@dataclass(frozen=True)
class RewardConfig:
    """
    Centralized Phase 8 Immutable Reward Configuration.

    Default Coefficients:
    ---------------------
    step_cost : float = -0.05
        Every RL decision consumes time/budget and incurs a small negative step cost.
    progress_reward : float = 1.00
        Reward added when an action successfully advances the workflow step.
    wrong_action_penalty : float = -0.50
        Penalty added when a real candidate is selected but fails to advance workflow.
    invalid_action_penalty : float = -1.00
        Penalty added when a padded or out-of-space action index is selected.
    completion_bonus : float = 5.00
        Bonus added on the final transition completing the entire workflow.
    failure_penalty : float = -2.00
        Penalty added on the transition causing episode truncation (budget exhausted).
    """
    step_cost: float = -0.05
    progress_reward: float = 1.00
    wrong_action_penalty: float = -0.50
    invalid_action_penalty: float = -1.00
    completion_bonus: float = 5.00
    failure_penalty: float = -2.00
