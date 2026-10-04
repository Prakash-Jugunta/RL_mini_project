"""
Phase 13 — Controlled DQN Workflow Diagnostic & Oracle Test Tool

Executes:
1. Real-browser Oracle Trajectories for LOGIN and CHECKOUT (evaluator-only ground truth).
2. Controlled Greedy DQN Trajectories (epsilon=0.0) with Top-5 Q-Value rank inspection per step.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import torch
import numpy as np
from typing import Dict, Any, List

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter
from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.network import apply_action_mask


def clean_ascii(text: str) -> str:
    if not text:
        return ""
    return text.encode("ascii", errors="replace").decode("ascii").replace("\n", " ")


def run_oracle_trajectory(workflow_id: str, adapter: PlaywrightBrowserAdapter) -> Dict[str, Any]:
    """
    Executes an oracle trajectory for a workflow on L0 by picking the correct ground-truth candidate at every step.
    """
    print(f"\n==================================================", flush=True)
    print(f" STARTING REAL BROWSER ORACLE TRAJECTORY: {workflow_id}", flush=True)
    print(f"==================================================", flush=True)

    env = UIRecoveryEnv(
        workflow_id=workflow_id,
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=adapter,
    )

    obs, info = env.reset()
    ep_return = 0.0
    decisions = 0
    terminated = False
    truncated = False
    step_info = info

    while not (terminated or truncated) and decisions < 20:
        current_step = env.workflow.steps[env.current_step_index]
        expected_role = current_step.expected_role

        # Find the correct candidate matching expected_role
        correct_action = None
        for idx, meta in enumerate(env._private_meta):
            if meta.semantic_role == expected_role:
                correct_action = idx
                break

        if correct_action is None:
            print(f"  [ORACLE FAIL] Could not find candidate with expected_role='{expected_role}' at step {env.current_step_index} ({current_step.step_id})")
            break

        cand = env.agent_candidates[correct_action]
        cand_txt = clean_ascii(cand.text)[:20]
        print(f"  Step {env.current_step_index} ({current_step.step_id}) -> Selecting Candidate {correct_action} (tag={cand.tag}, id={cand.attributes.get('id', '')}, text='{cand_txt}')")

        obs, reward, terminated, truncated, step_info = env.step(correct_action)
        ep_return += float(reward)
        decisions += 1

    success = bool(step_info.get("workflow_success", step_info.get("workflow_completed", False)))
    print(f"  ORACLE RESULT {workflow_id} | Success: {success} | Return: {ep_return:.2f} | Decisions: {decisions}", flush=True)

    return {
        "workflow_id": workflow_id,
        "success": success,
        "return": ep_return,
        "decisions": decisions,
    }


def run_greedy_dqn_diagnostic(
    workflow_id: str,
    agent: DQNAgent,
    adapter: PlaywrightBrowserAdapter
) -> Dict[str, Any]:
    """
    Executes a greedy DQN trajectory (epsilon=0.0) for a workflow and logs Top-5 Q-values and decision telemetry.
    """
    print(f"\n==================================================", flush=True)
    print(f" STARTING GREEDY DQN DIAGNOSTIC: {workflow_id} (epsilon=0.0)", flush=True)
    print(f"==================================================", flush=True)

    env = UIRecoveryEnv(
        workflow_id=workflow_id,
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=adapter,
    )

    obs, info = env.reset()
    ep_return = 0.0
    decisions = 0
    terminated = False
    truncated = False
    step_info = info
    failing_step = None

    agent.set_eval_mode()

    while not (terminated or truncated) and decisions < 20:
        current_step = env.workflow.steps[env.current_step_index]
        expected_role = current_step.expected_role
        candidate_mask = obs["candidate_mask"]
        valid_indices = np.flatnonzero(candidate_mask == 1)

        # Compute Q-values across all actions
        obs_t = {
            "objective": torch.tensor(obs["objective"], dtype=torch.float32, device=agent.device).unsqueeze(0),
            "context": torch.tensor(obs["context"], dtype=torch.float32, device=agent.device).unsqueeze(0),
            "candidates": torch.tensor(obs["candidates"], dtype=torch.float32, device=agent.device).unsqueeze(0),
        }
        mask_t = torch.tensor(candidate_mask, dtype=torch.float32, device=agent.device).unsqueeze(0)

        with torch.no_grad():
            raw_q = agent.online_network(obs_t)
            masked_q = apply_action_mask(raw_q, mask_t).squeeze(0).cpu().numpy()

        # Rank valid candidates by Q-value
        valid_q_tuples = []
        for idx in valid_indices:
            q_val = float(masked_q[idx])
            cand = env.agent_candidates[idx] if idx < len(env.agent_candidates) else None
            meta = env._private_meta[idx] if idx < len(env._private_meta) else None
            role = meta.semantic_role if meta else None
            is_correct = (role == expected_role) if expected_role else False
            valid_q_tuples.append((q_val, idx, cand, role, is_correct))

        valid_q_tuples.sort(key=lambda x: x[0], reverse=True)

        print(f"\n--- DECISION {decisions + 1} | Step Index: {env.current_step_index} | Step ID: {current_step.step_id} | Intent: '{current_step.agent_intent}' ---")
        print(f"  Expected Role (Private Ground Truth): '{expected_role}'")
        print(f"  TOP 5 VALID CANDIDATES BY Q-VALUE:")

        correct_candidate_q_rank = None
        for rank, (q_val, idx, cand, role, is_correct) in enumerate(valid_q_tuples[:5], 1):
            cand_tag = cand.tag if cand else "PAD"
            cand_id_attr = cand.attributes.get("id", "") if cand else ""
            cand_text = clean_ascii(cand.text)[:20] if cand else ""
            correct_mark = " [CORRECT TARGET!]" if is_correct else ""
            print(f"    Rank {rank}: Action {idx:2d} | Q = {q_val:7.4f} | Tag: {cand_tag:6s} | ID: {cand_id_attr:15s} | Text: '{cand_text}' | Role: {role}{correct_mark}")
            if is_correct and correct_candidate_q_rank is None:
                correct_candidate_q_rank = rank

        if correct_candidate_q_rank is None:
            # Check if correct candidate was outside top 5
            for rank, (q_val, idx, cand, role, is_correct) in enumerate(valid_q_tuples, 1):
                if is_correct:
                    correct_candidate_q_rank = rank
                    print(f"    -> Correct target candidate is Action {idx} at Rank {rank} with Q = {q_val:7.4f}")
                    break

        selected_action = valid_q_tuples[0][1]
        print(f"  Agent Selected Action: {selected_action} (Q = {valid_q_tuples[0][0]:.4f})")

        next_obs, reward, terminated, truncated, step_info = env.step(selected_action)
        ep_return += float(reward)
        decisions += 1

        print(f"  Step Outcome: Success = {step_info['step_success']} | Reward = {reward:.2f} | Execution Success = {step_info['execution_success']} | Failure Reason = {step_info.get('invalid_reason')}")

        if not step_info['step_success'] and failing_step is None:
            failing_step = current_step.step_id

        obs = next_obs

    success = bool(step_info.get("workflow_success", step_info.get("workflow_completed", False)))
    print(f"\nGREEDY DQN RESULT {workflow_id} | Success: {success} | Return: {ep_return:.2f} | Decisions: {decisions} | Failing Step: {failing_step}", flush=True)

    return {
        "workflow_id": workflow_id,
        "success": success,
        "return": ep_return,
        "decisions": decisions,
        "failing_step": failing_step,
    }


def main():
    adapter = PlaywrightBrowserAdapter(headless=True)
    try:
        print("=== TASK 2: RUNNING WORKFLOW ORACLE TRAJECTORY TESTS ===", flush=True)
        login_oracle = run_oracle_trajectory("LOGIN", adapter)
        checkout_oracle = run_oracle_trajectory("CHECKOUT", adapter)

        print("\n=== TASK 2: RUNNING GREEDY DQN WORKFLOW DIAGNOSTIC ===", flush=True)
        ckpt_path = "artifacts/dqn/test-run/latest_checkpoint.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "artifacts/dqn/phase13-v1/latest_checkpoint.pt"

        config = DQNConfig(seed=2026)
        agent = DQNAgent(config)

        if os.path.exists(ckpt_path):
            print(f"Loading trained DQN checkpoint from '{ckpt_path}'...", flush=True)
            agent.load_checkpoint(ckpt_path)
        else:
            print("No checkpoint found; running diagnostic on fresh network...", flush=True)

        login_greedy = run_greedy_dqn_diagnostic("LOGIN", agent, adapter)
        checkout_greedy = run_greedy_dqn_diagnostic("CHECKOUT", agent, adapter)

        print("\n==================================================", flush=True)
        print(" SUMMARY OF DIAGNOSTIC RESULTS", flush=True)
        print("==================================================", flush=True)
        print(f" LOGIN Oracle:    Success={login_oracle['success']} | Return={login_oracle['return']:.2f}")
        print(f" CHECKOUT Oracle: Success={checkout_oracle['success']} | Return={checkout_oracle['return']:.2f}")
        print(f" LOGIN Greedy:    Success={login_greedy['success']} | Return={login_greedy['return']:.2f} | Failing Step={login_greedy['failing_step']}")
        print(f" CHECKOUT Greedy: Success={checkout_greedy['success']} | Return={checkout_greedy['return']:.2f} | Failing Step={checkout_greedy['failing_step']}")

    finally:
        adapter.close()


if __name__ == "__main__":
    main()
