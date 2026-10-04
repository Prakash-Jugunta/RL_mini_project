import time
from typing import Dict, Any, List, Optional
import numpy as np

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.workflow import WORKFLOW_REGISTRY
from rl.baselines.policies import RandomValidPolicy, PublicFeatureHeuristicPolicy, OraclePolicy

SLOW_STEP_SECONDS = 2.0


def run_episode(
    env: UIRecoveryEnv,
    policy: Any,
    workflow_id: str,
    mutation_level: int,
    mutation_seed: int,
    policy_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generic episode runner for baseline policies on UIRecoveryEnv.
    Strictly uses standard Gymnasium environment API (reset, step, get_action_mask).
    Supports decision-level diagnostic logging when verbose=True.
    """
    mode = "test" if mutation_level == 6 else "validation"
    options = {
        "workflow_id": workflow_id,
        "mode": mode,
        "mutation_level": mutation_level,
        "mutation_seed": mutation_seed,
    }

    if isinstance(policy, RandomValidPolicy) and policy_seed is not None:
        policy.rng = np.random.default_rng(policy_seed)

    obs, info = env.reset(options=options)

    episode_return = 0.0
    decisions = 0

    while True:
        if env.current_step_index >= len(env.workflow.steps):
            # Workflow already complete (e.g. all steps were page-state assertions)
            terminated = True
            truncated = False
            break

        current_step = env.workflow.steps[env.current_step_index]
        action_mask = env.get_action_mask()
        valid_count = int(np.sum(action_mask == 1))

        if isinstance(policy, RandomValidPolicy):
            action = policy.select_action(observation=obs, action_mask=action_mask)
        elif isinstance(policy, PublicFeatureHeuristicPolicy):
            action = policy.select_action(
                agent_intent=current_step.agent_intent,
                action_type=current_step.action_type,
                candidates=env.agent_candidates,
                action_mask=action_mask,
            )
        elif isinstance(policy, OraclePolicy):
            action = policy.select_action(
                expected_role=current_step.expected_role,
                private_meta=env._private_meta,
                action_mask=action_mask,
            )
        else:
            # Generic policy interface
            if hasattr(policy, "select_action"):
                action = policy.select_action(
                    observation=obs,
                    action_mask=action_mask,
                    agent_intent=current_step.agent_intent,
                    action_type=current_step.action_type,
                    candidates=env.agent_candidates,
                )
            else:
                raise ValueError(f"Unsupported policy type: {type(policy)}")

        decision_num = decisions + 1
        step_id = current_step.step_id

        if verbose:
            print(
                f"    DECISION {decision_num} | "
                f"step={step_id} | "
                f"action={action} | "
                f"valid_candidates={valid_count}",
                flush=True,
            )

        step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_elapsed = time.perf_counter() - step_start

        if verbose:
            print(
                f"      DONE {step_elapsed:.2f}s | "
                f"reward={reward:.2f} | "
                f"step_success={info.get('step_success')} | "
                f"execution_success={info.get('execution_success')} | "
                f"reason={info.get('invalid_reason')} | "
                f"terminated={terminated} | "
                f"truncated={truncated}",
                flush=True,
            )

        if step_elapsed >= SLOW_STEP_SECONDS:
            print(
                f"WARNING: SLOW ENVIRONMENT STEP "
                f"workflow={workflow_id} "
                f"level=L{mutation_level} "
                f"mutation_seed={mutation_seed} "
                f"decision={decision_num} "
                f"step_id={step_id} "
                f"action={action} "
                f"elapsed={step_elapsed:.2f}s",
                flush=True,
            )

        episode_return += float(reward)
        decisions += 1

        if terminated or truncated:
            break

    success = bool(terminated and info.get("workflow_completed", False))

    return {
        "policy_version": getattr(policy, "version", "unknown"),
        "workflow_id": workflow_id,
        "mutation_level": mutation_level,
        "mutation_seed": mutation_seed,
        "policy_seed": policy_seed if policy_seed is not None else 0,
        "success": success,
        "episode_return": float(episode_return),
        "decisions": int(decisions),
        "wrong_actions": int(info.get("wrong_actions", 0)),
        "invalid_actions": int(info.get("padded_actions", 0) + info.get("out_of_space_actions", 0)),
        "execution_failures": int(info.get("execution_failures", 0)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward_version": info.get("reward_version", "v1-canonical-phase8"),
    }


def evaluate_baseline_matrix(
    env: UIRecoveryEnv,
    policy: Any,
    workflows: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    policy_seeds: Optional[List[int]] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluates a policy across the full evaluation matrix (workflows x levels x seeds).
    """
    if workflows is None:
        workflows = ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]
    if levels is None:
        levels = [0, 1, 2, 3, 4, 5, 6]
    if seeds is None:
        seeds = [11, 22, 33, 44, 55]

    if isinstance(policy, RandomValidPolicy):
        p_seeds = policy_seeds if policy_seeds is not None else list(range(10))
    else:
        p_seeds = [0]

    results = []

    for w_id in workflows:
        for lvl in levels:
            for m_seed in seeds:
                for p_seed in p_seeds:
                    res = run_episode(
                        env=env,
                        policy=policy,
                        workflow_id=w_id,
                        mutation_level=lvl,
                        mutation_seed=m_seed,
                        policy_seed=p_seed,
                        verbose=verbose,
                    )
                    results.append(res)

    return results
