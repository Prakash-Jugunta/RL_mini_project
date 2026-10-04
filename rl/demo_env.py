"""
===============================================================================
ENVIRONMENT SMOKE TEST — NOT LEARNED POLICY
===============================================================================

This script demonstrates deterministic environment transitions using a mock candidate
browser adapter. It validates Gymnasium reset, step, reward, and completion mechanics.

NOTE: This is NOT an RL agent and performs NO model inference/training.
===============================================================================
"""

from rl.env.ui_recovery_env import UIRecoveryEnv


def run_smoke_demo():
    print("=" * 79)
    print("  ENVIRONMENT SMOKE TEST — NOT LEARNED POLICY")
    print("=" * 79)
    print()

    workflow_id = "LOGIN"
    level = 1
    seed = 11

    print(f"Initializing UIRecoveryEnv (Workflow: {workflow_id}, Level: {level}, Seed: {seed}, Mode: train)...")
    env = UIRecoveryEnv(
        workflow_id=workflow_id,
        mutation_level=level,
        mutation_seed=seed,
        mode="train"
    )

    obs, info = env.reset(seed=42)

    print(f"RESET COMPLETE")
    print(f"  Workflow:        {info['workflow_id']}")
    print(f"  Initial Step:    {info['step_id']} (Index: {info['step_index']})")
    print(f"  Candidates:      {len(env.agent_candidates)}")
    print(f"  Observation:     {obs}")
    print(f"  Mutation:        Level {info['mutation_level']} / Seed {info['mutation_seed']}")
    print()

    episode_return = 0.0
    step_count = 0
    terminated = False
    truncated = False

    print("--- STARTING DETERMINISTIC SIMULATION ---")

    while not (terminated or truncated):
        current_step = env.workflow.steps[env.current_step_index]

        # Locate the target candidate index privately for demo advancement
        chosen_action = None
        for idx, cand in enumerate(env.agent_candidates):
            meta = env.browser_adapter.private_metadata_map.get(cand.candidate_id)
            if meta and meta.semantic_role == current_step.expected_role:
                chosen_action = idx
                break

        if chosen_action is None:
            chosen_action = 0  # Fallback action

        obs, reward, terminated, truncated, step_info = env.step(chosen_action)
        episode_return += reward
        step_count += 1

        next_step_str = step_info['step_id'] if step_info['step_id'] is not None else "NONE — WORKFLOW COMPLETE"

        print(f"Step {step_count}: Action {chosen_action}")
        print(f"  Executed Target: {current_step.step_id} (Type: {current_step.action_type})")
        print(f"  Reward:          {reward:+.1f}")
        print(f"  Step Success:    {step_info['action_success']}")
        print(f"  Next Objective:  {next_step_str} (Index: {step_info['step_index']})")
        print(f"  Terminated:      {terminated} | Truncated: {truncated}")
        print()

    print("=" * 79)
    print("EPISODE COMPLETE")
    print(f"  Total Return:    {episode_return:+.1f}")
    print(f"  Total Steps:     {step_count}")
    print(f"  Wrong Actions:   {step_info['wrong_actions']}")
    print(f"  Workflow Done:   {step_info['workflow_completed']}")
    print("=" * 79)

    env.close()


if __name__ == "__main__":
    run_smoke_demo()
