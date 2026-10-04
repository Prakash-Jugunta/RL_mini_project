"""
Phase 8 — Reward Inspection & Tracing Utility
=============================================

Run with:
    python -m rl.reward.inspect_reward
"""

import numpy as np
from rl.env.workflow import WORKFLOW_REGISTRY
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.reward import REWARD_VERSION, RewardConfig, RewardCalculator


def inspect_rewards():
    print("=" * 70)
    print(f"PHASE 8 REWARD CONTRACT & TRACE INSPECTION [{REWARD_VERSION}]")
    print("=" * 70)

    config = RewardConfig()
    print("\n1. REWARD CONFIGURATION DEFAULTS")
    print("-" * 50)
    print(f"  Step Cost               : {config.step_cost:+.2f}")
    print(f"  Progress Reward         : {config.progress_reward:+.2f}")
    print(f"  Wrong Action Penalty    : {config.wrong_action_penalty:+.2f}")
    print(f"  Invalid Action Penalty  : {config.invalid_action_penalty:+.2f}")
    print(f"  Completion Bonus        : {config.completion_bonus:+.2f}")
    print(f"  Failure Penalty         : {config.failure_penalty:+.2f}")

    print("\n2. THEORETICAL OPTIMAL RETURNS PER WORKFLOW")
    print("-" * 50)

    expected_returns = {
        "LOGIN": 8.80,     # 3 * 0.95 + 5.95
        "SEARCH": 7.85,    # 2 * 0.95 + 5.95 (VERIFY_PRODUCT_PAGE is 0 RL decisions)
        "PROFILE": 8.80,   # 3 * 0.95 + 5.95
        "CHECKOUT": 9.75,  # 4 * 0.95 + 5.95
    }

    for w_id, exp_ret in expected_returns.items():
        env = UIRecoveryEnv(workflow_id=w_id, max_candidates=20)
        obs, info = env.reset()

        total_return = 0.0
        decisions = 0

        while True:
            curr_step = env.workflow.steps[env.current_step_index]
            corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
            
            obs, reward, term, trunc, info = env.step(corr_idx)
            decisions += 1
            total_return += reward
            if term or trunc:
                break

        print(f"Workflow: {w_id:8s} | Decisions: {decisions} | Empirical Return: {total_return:+.2f} | Expected: {exp_ret:+.2f}")
        env.close()

    print("\n3. SAMPLE EPISODE REWARD BREAKDOWN TRACE (LOGIN Workflow)")
    print("-" * 50)

    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    episode_return = 0.0

    for decision_num in range(1, 5):
        curr_step = env.workflow.steps[env.current_step_index]
        corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
        
        obs, reward, term, trunc, info = env.step(corr_idx)
        episode_return += reward
        bd = info["reward_breakdown"]

        print(f"Decision {decision_num}: Step={curr_step.step_id:15s} | Action Valid={info['action_valid']} | Step Success={info['step_success']}")
        print(f"  Step cost: {bd['step_cost']:+.2f} | Progress: {bd['progress']:+.2f} | Wrong: {bd['wrong_action']:+.2f} | Invalid: {bd['invalid_action']:+.2f} | Completion: {bd['completion']:+.2f} | Failure: {bd['failure']:+.2f}")
        print(f"  -> Decision Reward: {bd['total']:+.2f} | Cumulative Return: {episode_return:+.2f}\n")

    env.close()

    print("=" * 70)
    print("PHASE 8 REWARD INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    inspect_rewards()
