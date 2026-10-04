"""
Phase 7 — Action Space Inspection & Statistics Utility
======================================================

Run with:
    python -m rl.action.inspect_action
"""

import numpy as np

from rl.env.workflow import WORKFLOW_REGISTRY
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.action.sampler import sample_valid_action


def inspect_action_space():
    print("=" * 70)
    print("PHASE 7 ACTION SPACE & CANDIDATE ACTION CONTRACT INSPECTION")
    print("=" * 70)

    candidate_counts = []
    rng = np.random.default_rng(42)

    workflows = list(WORKFLOW_REGISTRY.keys())
    levels = list(range(7))  # L0..L6 (L6 evaluation only)

    print("\n1. ACTION MASK ALIGNMENT & DECISION COUNT CHECK")
    print("-" * 50)

    for w_id in workflows:
        # Use mode='test' to allow inspecting L6 for evaluation statistics
        env = UIRecoveryEnv(workflow_id=w_id, mutation_level=0, mode="test")
        obs, info = env.reset(seed=42)

        mask = env.get_action_mask()
        is_aligned = np.array_equal(mask, obs["candidate_mask"])
        n_valid = int(mask.sum())
        candidate_counts.append(n_valid)

        print(f"Workflow: {w_id:8s} | Initial Step: {info['step_id']:20s} | Mask Valid: {n_valid:2d}/20 | Mask Aligned: {is_aligned}")

        # Execute an episode using random valid action sampler
        decision_steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and decision_steps < 10:
            current_mask = env.get_action_mask()
            action = sample_valid_action(current_mask, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            decision_steps += 1

        print(f"  -> Episode completed in {decision_steps} RL decisions (completed={info['workflow_completed']})")
        env.close()

    print("\n2. CANDIDATE COUNT & ACTION SPACE UTILIZATION (L0..L6 Real Browser Sweep)")
    print("-" * 50)

    from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter

    all_counts = []
    adapter = PlaywrightBrowserAdapter(base_url="http://localhost:3000", max_candidates=20)

    try:
        for lvl in levels:
            for w_id in workflows:
                env = UIRecoveryEnv(workflow_id=w_id, mutation_level=lvl, mode="test", browser_adapter=adapter)
                obs, info = env.reset(seed=42 + lvl)
                cnt = int(obs["candidate_mask"].sum())
                all_counts.append(cnt)
    finally:
        adapter.close()

    min_cnt = min(all_counts)
    max_cnt = max(all_counts)
    avg_cnt = float(np.mean(all_counts))
    utilization = (avg_cnt / 20.0) * 100.0

    print(f"Mutation Levels Evaluated : L0 – L6")
    print(f"Minimum Candidates (N_min): {min_cnt}")
    print(f"Maximum Candidates (N_max): {max_cnt}")
    print(f"Average Candidates (N_avg): {avg_cnt:.2f}")
    print(f"Average Valid Actions     : {avg_cnt:.2f} / 20")
    print(f"Average Padded Actions    : {20.0 - avg_cnt:.2f} / 20")
    print(f"Action Space Utilization  : {utilization:.1f}%")

    print("\n3. ROBUSTNESS TO PADDED AND OUT-OF-SPACE ACTIONS")
    print("-" * 50)

    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    obs, info = env.reset(seed=42)
    n_real = int(obs["candidate_mask"].sum())

    # Padded action
    padded_action = n_real + 1
    obs_p, reward_p, term_p, trunc_p, info_p = env.step(padded_action)
    print(f"Padded Action ({padded_action}): valid={info_p['action_valid']}, reason={info_p.get('invalid_reason')}, step_idx={info_p['step_index']}")

    # Out-of-space negative action
    obs_n, reward_n, term_n, trunc_n, info_n = env.step(-1)
    print(f"Negative Action (-1) : valid={info_n['action_valid']}, reason={info_n.get('invalid_reason')}, step_idx={info_n['step_index']}")

    # Out-of-space large action
    obs_l, reward_l, term_l, trunc_l, info_l = env.step(99)
    print(f"Large Action (99)    : valid={info_l['action_valid']}, reason={info_l.get('invalid_reason')}, step_idx={info_l['step_index']}")

    env.close()
    print("\n" + "=" * 70)
    print("PHASE 7 ACTION CONTRACT VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    inspect_action_space()
