"""
Phase 11 — Single-Episode Runner & Environment Reset Bridge

Provides clean reset initialization from EpisodeSpec objects and generic
single-episode execution machinery with diagnostic result structures.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Callable

import numpy as np

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.training.episode_spec import EpisodeSpec, validate_episode_spec


@dataclass(frozen=True)
class EpisodeResult:
    """
    Diagnostic summary result of executing a single RL episode.
    Contains no private evaluator metadata or target locator identities.
    """

    episode_id: int
    workflow: str
    mutation_level: str
    mutation_seed: int
    environment_seed: int
    success: bool
    episode_return: float
    decision_count: int
    wrong_actions: int
    invalid_actions: int
    execution_failures: int
    terminated: bool
    truncated: bool


def reset_from_episode_spec(
    env: UIRecoveryEnv,
    spec: EpisodeSpec,
    mode: str = "train"
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Bridges EpisodeSpec to Gymnasium environment reset options.
    Enforces specification verification post-reset.
    """
    validate_episode_spec(spec, mode=mode)
    options = spec.to_reset_options(mode=mode)
    obs, info = env.reset(seed=spec.environment_seed, options=options)

    # Post-reset sanity verification against spec
    if info.get("workflow_id") != spec.workflow:
        raise RuntimeError(
            f"Environment workflow mismatch: expected '{spec.workflow}', got '{info.get('workflow_id')}'"
        )

    if info.get("mutation_level") != spec.numeric_mutation_level():
        raise RuntimeError(
            f"Environment mutation level mismatch: expected {spec.numeric_mutation_level()}, "
            f"got {info.get('mutation_level')}"
        )

    if info.get("mutation_seed") != spec.mutation_seed:
        raise RuntimeError(
            f"Environment mutation seed mismatch: expected {spec.mutation_seed}, "
            f"got {info.get('mutation_seed')}"
        )

    return obs, info


def run_single_episode(
    env: UIRecoveryEnv,
    policy_fn: Callable[[Dict[str, np.ndarray], Dict[str, Any]], int],
    spec: EpisodeSpec,
    max_decisions: int = 50,
    mode: str = "train"
) -> EpisodeResult:
    """
    Executes a single episode in the Gymnasium environment using a policy function.
    Does NOT initiate multi-episode RL optimization.
    """
    obs, info = reset_from_episode_spec(env, spec, mode=mode)

    episode_return: float = 0.0
    decision_count: int = 0
    wrong_actions: int = 0
    invalid_actions: int = 0
    execution_failures: int = 0
    terminated: bool = False
    truncated: bool = False

    while not (terminated or truncated) and decision_count < max_decisions:
        action = policy_fn(obs, info)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        episode_return += float(reward)
        decision_count += 1

        if step_info.get("is_wrong_candidate"):
            wrong_actions += 1
        if step_info.get("is_invalid_candidate"):
            invalid_actions += 1
        if step_info.get("execution_failed"):
            execution_failures += 1

        obs = next_obs
        info = step_info

    if decision_count >= max_decisions and not terminated:
        truncated = True

    success = bool(info.get("workflow_completed", False))

    return EpisodeResult(
        episode_id=spec.episode_id,
        workflow=spec.workflow,
        mutation_level=spec.mutation_level,
        mutation_seed=spec.mutation_seed,
        environment_seed=spec.environment_seed,
        success=success,
        episode_return=episode_return,
        decision_count=decision_count,
        wrong_actions=wrong_actions,
        invalid_actions=invalid_actions,
        execution_failures=execution_failures,
        terminated=terminated,
        truncated=truncated,
    )
