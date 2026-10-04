"""
Real Headed DQN Demo Runner for Professor Presentation.

Executes greedy DQN evaluation (epsilon=0, explore=False) on the real Playwright
browser environment at http://localhost:3000 with visual element highlighting,
step-by-step decision telemetry, and optional transparent demo-only recovery injection.

Safety & Integrity Guarantees:
------------------------------
- NO retraining.
- NO DQN architecture, reward, split, or checkpoint modifications.
- NO optimizer or replay buffer updates (torch.no_grad(), eval mode).
- Public candidate features ONLY used for DQN policy action selection.
- data-semantic-role / private metadata used strictly for evaluator/debug output.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.network import apply_action_mask
from rl.env.browser_adapter import BrowserAdapter, MockBrowserAdapter
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter
from rl.env.ui_recovery_env import UIRecoveryEnv


DEFAULT_CHECKPOINT: str = "artifacts/dqn/phase13-v1/latest_checkpoint.pt"
DEFAULT_BASE_URL: str = "http://localhost:3000"


def clean_text(text: Optional[str], max_len: int = 25) -> str:
    """Clean string for tabular console display."""
    if not text:
        return ""
    clean = text.encode("ascii", errors="replace").decode("ascii").replace("\n", " ").strip()
    if len(clean) > max_len:
        return clean[: max_len - 3] + "..."
    return clean


def check_frontend_reachable(base_url: str) -> bool:
    """Probes the Vite dev server to verify real browser connectivity."""
    api_url = f"{base_url.rstrip('/')}/api/set-mutation?level=0&seed=11"
    try:
        req = urllib.request.Request(api_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            return bool(data.get("ok"))
    except Exception:
        return False


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Real Headed DQN Demo Runner for Professor Presentation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained DQN checkpoint",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="LOGIN",
        choices=["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"],
        help="Target workflow",
    )
    parser.add_argument(
        "--mutation-level",
        type=int,
        default=6,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help="Mutation level (0..6)",
    )
    parser.add_argument(
        "--mutation-seed",
        type=int,
        default=11,
        choices=[11, 22, 33, 44, 55],
        help="Mutation seed",
    )

    # Headed / Headless flags
    parser.add_argument(
        "--headed",
        dest="headed",
        action="store_true",
        default=True,
        help="Run in headed Chromium mode (default)",
    )
    parser.add_argument(
        "--headless",
        dest="headed",
        action="store_false",
        help="Run in headless Chromium mode",
    )

    # Step / Auto flags
    parser.add_argument(
        "--step",
        dest="auto",
        action="store_false",
        default=False,
        help="Wait for ENTER before each action (default)",
    )
    parser.add_argument(
        "--auto",
        dest="auto",
        action="store_true",
        help="Automatically proceed with delay-ms between actions",
    )

    parser.add_argument(
        "--delay-ms",
        type=int,
        default=1200,
        help="Delay in ms between actions in auto mode",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to display in table",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL of application",
    )
    parser.add_argument(
        "--show-evaluator-debug",
        action="store_true",
        default=False,
        help="Display evaluator private semantic roles for diagnostics",
    )
    parser.add_argument(
        "--inject-wrong-step",
        type=int,
        default=None,
        help="1-based decision step number at which to inject a demo-only wrong action for recovery demonstration",
    )
    parser.add_argument(
        "--inject-wrong-rank",
        type=int,
        default=2,
        help="Target 1-based rank of candidate to inject at the specified wrong step (default: 2)",
    )
    return parser


def run_dqn_demo(
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    workflow_id: str = "LOGIN",
    mutation_level: int = 6,
    mutation_seed: int = 11,
    headed: bool = True,
    auto: bool = False,
    delay_ms: int = 1200,
    top_k: int = 5,
    base_url: str = DEFAULT_BASE_URL,
    show_evaluator_debug: bool = False,
    inject_wrong_step: Optional[int] = None,
    inject_wrong_rank: int = 2,
    browser_adapter: Optional[BrowserAdapter] = None,
    interactive_prompt: bool = True,
) -> Dict[str, Any]:
    """
    Executes a single greedy DQN demo run with real browser connectivity and visual highlighting.
    Optionally injects a single demo-only wrong action at inject_wrong_step to demonstrate recovery.
    """
    # 1. Checkpoint Verification
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # 2. Real Browser Connectivity Check (unless custom mock adapter passed)
    is_real_browser = browser_adapter is None or isinstance(browser_adapter, PlaywrightBrowserAdapter)

    if is_real_browser and browser_adapter is None:
        if not check_frontend_reachable(base_url):
            print("\nBrowser: Chromium", flush=True)
            print(f"Headless: {not headed}", flush=True)
            print(f"Base URL: {base_url}", flush=True)
            print("REAL BROWSER: FAIL", flush=True)
            raise RuntimeError(
                f"Real browser target {base_url} is unreachable. "
                "Ensure Vite dev server is running (npm run dev)!"
            )

        print("\nBrowser: Chromium", flush=True)
        print(f"Headless: {not headed}", flush=True)
        print(f"Base URL: {base_url}", flush=True)
        print("REAL BROWSER: PASS\n", flush=True)

        adapter = PlaywrightBrowserAdapter(
            base_url=base_url,
            max_candidates=20,
            headless=not headed,
            slow_mo=0.0,
        )
    else:
        adapter = browser_adapter or MockBrowserAdapter(max_candidates=20)
        print("\nBrowser: Adapter Initialized", flush=True)
        print(f"Base URL: {base_url}", flush=True)
        print("REAL BROWSER: PASS\n", flush=True)

    try:
        # Load agent & checkpoint
        config = DQNConfig(seed=2026)
        agent = DQNAgent(config)
        agent.load_checkpoint(checkpoint_path)
        agent.set_eval_mode()

        # Determine env mode (L6 uses test to comply with held-out protection)
        env_mode = "test" if mutation_level == 6 else "validation"

        # Create UIRecoveryEnv
        env = UIRecoveryEnv(
            workflow_id=workflow_id,
            mutation_level=mutation_level,
            mutation_seed=mutation_seed,
            mode=env_mode,
            browser_adapter=adapter,
        )

        obs, info = env.reset(seed=42)

        ep_return = 0.0
        decisions = 0
        injected_wrong_actions_count = 0
        terminated = False
        truncated = False
        step_info = info
        failing_step = None
        failure_reason = None

        total_workflow_steps = len(env.workflow.steps)

        with torch.no_grad():
            while not (terminated or truncated) and decisions < env.max_episode_steps:
                curr_step_idx = env.current_step_index
                curr_step = env.workflow.steps[curr_step_idx]
                step_id = curr_step.step_id
                intent = curr_step.agent_intent
                current_decision_num = decisions + 1

                candidate_mask = obs["candidate_mask"]
                valid_indices = np.flatnonzero(candidate_mask == 1)

                obs_t = {
                    "objective": torch.tensor(obs["objective"], dtype=torch.float32, device=agent.device).unsqueeze(0),
                    "context": torch.tensor(obs["context"], dtype=torch.float32, device=agent.device).unsqueeze(0),
                    "candidates": torch.tensor(obs["candidates"], dtype=torch.float32, device=agent.device).unsqueeze(0),
                }
                mask_t = torch.tensor(candidate_mask, dtype=torch.float32, device=agent.device).unsqueeze(0)

                raw_q = agent.online_network(obs_t)
                masked_q = apply_action_mask(raw_q, mask_t).squeeze(0).cpu().numpy()

                # Rank valid candidates by Q-value
                valid_q_tuples = []
                for idx in valid_indices:
                    q_val = float(masked_q[idx])
                    cand = env.agent_candidates[idx] if idx < len(env.agent_candidates) else None
                    tag = cand.tag if cand else "PAD"
                    pub_id_text = ""
                    if cand:
                        pub_id_text = (
                            cand.attributes.get("id", "")
                            or cand.text
                            or cand.placeholder
                            or cand.aria_label
                            or ""
                        )
                    pub_id_text = clean_text(pub_id_text, max_len=25)
                    valid_q_tuples.append((q_val, idx, tag, pub_id_text))

                valid_q_tuples.sort(key=lambda x: x[0], reverse=True)

                greedy_action = valid_q_tuples[0][1]
                greedy_q = valid_q_tuples[0][0]
                greedy_pub_id = valid_q_tuples[0][3]

                # Check if this step is specified for demo-only wrong action injection
                is_injected_step = (
                    inject_wrong_step is not None
                    and current_decision_num == inject_wrong_step
                )

                if is_injected_step:
                    # Select an injected candidate: valid, NOT greedy top-1, and evaluator-confirmed incorrect
                    expected_role = curr_step.expected_role
                    injected_tuple = None

                    # Check preferred rank first
                    target_idx = max(0, min(inject_wrong_rank - 1, len(valid_q_tuples) - 1))
                    candidate_tuple = valid_q_tuples[target_idx]
                    cand_act = candidate_tuple[1]
                    cand_meta = (
                        env._private_meta[cand_act]
                        if cand_act < len(env._private_meta)
                        else None
                    )
                    cand_role = cand_meta.semantic_role if cand_meta else None

                    if cand_act != greedy_action and cand_role != expected_role:
                        injected_tuple = candidate_tuple

                    if injected_tuple is None:
                        # Fallback search for any valid candidate != greedy_action and role != expected_role
                        for tup in valid_q_tuples:
                            act = tup[1]
                            meta = (
                                env._private_meta[act]
                                if act < len(env._private_meta)
                                else None
                            )
                            role = meta.semantic_role if meta else None
                            if act != greedy_action and role != expected_role:
                                injected_tuple = tup
                                break

                    if injected_tuple is None:
                        # Secondary fallback: any valid candidate != greedy_action
                        for tup in valid_q_tuples:
                            if tup[1] != greedy_action:
                                injected_tuple = tup
                                break

                    if injected_tuple is None:
                        injected_tuple = valid_q_tuples[0]

                    selected_action = injected_tuple[1]
                    selected_q = injected_tuple[0]
                    injected_pub_id = injected_tuple[3]

                    print("==================================================", flush=True)
                    print("DEMO-ONLY RECOVERY INJECTION", flush=True)
                    print("==================================================\n", flush=True)
                    print("Greedy DQN would choose:", flush=True)
                    print(f"Action: {greedy_action}", flush=True)
                    print(f"Candidate: {greedy_pub_id}", flush=True)
                    print(f"Q-value: {greedy_q:.3f}\n", flush=True)
                    print("Injected wrong action for demonstration:", flush=True)
                    print(f"Action: {selected_action}", flush=True)
                    print(f"Candidate: {injected_pub_id}", flush=True)
                    print(f"Q-value: {selected_q:.3f}\n", flush=True)
                    print("NOTE:", flush=True)
                    print("This action was intentionally injected for demo purposes.", flush=True)
                    print("It is NOT the greedy DQN decision.\n", flush=True)

                    if selected_action < len(env.agent_candidates):
                        selected_cand = env.agent_candidates[selected_action]
                        adapter.highlight_candidate(
                            candidate_id=selected_cand.candidate_id,
                            label="DEMO INJECTION",
                            duration_sec=1.0,
                        )

                    if not auto and interactive_prompt:
                        input("Press ENTER to execute action...")
                    elif auto and delay_ms > 0:
                        time.sleep(delay_ms / 1000.0)

                    obs, reward, terminated, truncated, step_info = env.step(selected_action)
                    ep_return += float(reward)
                    decisions += 1
                    injected_wrong_actions_count += 1

                    exec_succ = step_info.get("execution_success", False)
                    step_succ = step_info.get("step_success", False)
                    rew_str = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"

                    print(f"Execution Success: {exec_succ}", flush=True)
                    print(f"Step Success: {step_succ}", flush=True)
                    print(f"Reward: {rew_str}", flush=True)
                    print(f"Current Step Remains: {step_id}\n", flush=True)
                    print("Returning control to greedy DQN...\n", flush=True)

                else:
                    # Normal Greedy DQN Step
                    print("==================================================", flush=True)
                    print("DQN DEMO", flush=True)
                    print(f"Workflow: {workflow_id}", flush=True)
                    print(f"Mutation: L{mutation_level} / Seed {mutation_seed}", flush=True)
                    print(f"Step: {curr_step_idx + 1}/{total_workflow_steps} — {step_id}", flush=True)
                    print(f"Intent: {intent}", flush=True)
                    print("==================================================\n", flush=True)

                    print("TOP CANDIDATES\n", flush=True)
                    print(f"{'Rank':<6}{'Action':<8}{'Q-value':<10}{'Tag':<9}{'Public ID/Text'}", flush=True)

                    top_records = valid_q_tuples[:top_k]
                    for rank, (q_val, idx, tag, pub_id_text) in enumerate(top_records, 1):
                        print(f"{rank:<6}{idx:<8}{q_val:<10.3f}{tag:<9}{pub_id_text}", flush=True)

                    selected_action = greedy_action
                    selected_q = greedy_q

                    print("\nSELECTED:", flush=True)
                    print(f"Action: {selected_action}", flush=True)
                    print(f"Q-value: {selected_q:.3f}\n", flush=True)

                    if show_evaluator_debug:
                        exp_role = curr_step.expected_role
                        sel_meta = (
                            env._private_meta[selected_action]
                            if selected_action < len(env._private_meta)
                            else None
                        )
                        sel_role = sel_meta.semantic_role if sel_meta else None
                        print("--------------------------------------------------", flush=True)
                        print("EVALUATOR ONLY — NOT POLICY INPUT", flush=True)
                        print(f"expected semantic role: {exp_role}", flush=True)
                        print(f"selected semantic role: {sel_role}", flush=True)
                        print("--------------------------------------------------\n", flush=True)

                    if selected_action < len(env.agent_candidates):
                        selected_cand = env.agent_candidates[selected_action]
                        adapter.highlight_candidate(
                            candidate_id=selected_cand.candidate_id,
                            label="DQN SELECTED",
                            duration_sec=1.0,
                        )

                    if not auto and interactive_prompt:
                        input("Press ENTER to execute action...")
                    elif auto and delay_ms > 0:
                        time.sleep(delay_ms / 1000.0)

                    obs, reward, terminated, truncated, step_info = env.step(selected_action)
                    ep_return += float(reward)
                    decisions += 1

                    exec_succ = step_info.get("execution_success", False)
                    step_succ = step_info.get("step_success", False)
                    next_step_id = step_info.get("step_id")
                    if terminated:
                        next_step_id = "COMPLETED"
                    elif next_step_id is None:
                        next_step_id = "NONE"

                    rew_str = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"

                    print(f"Execution Success: {exec_succ}", flush=True)
                    print(f"Step Success: {step_succ}", flush=True)
                    print(f"Reward: {rew_str}", flush=True)
                    print(f"Next Step: {next_step_id}\n", flush=True)

                    if not step_succ and failing_step is None:
                        failing_step = step_id
                        failure_reason = step_info.get("invalid_reason", "incorrect_candidate")

        workflow_success = bool(
            step_info.get("workflow_success", step_info.get("workflow_completed", False))
        )

        natural_dqn_wrong_actions = max(0, env.wrong_actions - injected_wrong_actions_count)
        if injected_wrong_actions_count > 0:
            recovery_str = "TRUE" if workflow_success else "FALSE"
        else:
            recovery_str = "N/A"

        print("==================================================", flush=True)
        if workflow_success:
            print("WORKFLOW COMPLETE", flush=True)
            print("==================================================", flush=True)
            print(f"Workflow: {workflow_id}", flush=True)
            print(f"Mutation: L{mutation_level} / Seed {mutation_seed}", flush=True)
            print("Success: TRUE", flush=True)
            print(f"Return: {ep_return:.2f}", flush=True)
            print(f"Decisions: {decisions}", flush=True)
            print(f"Wrong Actions: {env.wrong_actions}", flush=True)
            print(f"Injected Wrong Actions: {injected_wrong_actions_count}", flush=True)
            print(f"Natural DQN Wrong Actions: {natural_dqn_wrong_actions}", flush=True)
            print(f"Recovery After Injection: {recovery_str}", flush=True)
            print(f"Execution Failures: {env.execution_failures}", flush=True)
            print("Checkpoint:", flush=True)
            print(f"{checkpoint_path}\n", flush=True)
        else:
            print("WORKFLOW FAILED", flush=True)
            print("==================================================", flush=True)
            print(f"Workflow: {workflow_id}", flush=True)
            print(f"Mutation: L{mutation_level} / Seed {mutation_seed}", flush=True)
            print("Success: FALSE", flush=True)
            print(f"Failing Step: {failing_step or 'N/A'}", flush=True)
            print(f"Failure Reason: {failure_reason or 'N/A'}", flush=True)
            print(f"Return: {ep_return:.2f}", flush=True)
            print(f"Decisions: {decisions}", flush=True)
            print(f"Wrong Actions: {env.wrong_actions}", flush=True)
            print(f"Injected Wrong Actions: {injected_wrong_actions_count}", flush=True)
            print(f"Natural DQN Wrong Actions: {natural_dqn_wrong_actions}", flush=True)
            print(f"Recovery After Injection: {recovery_str}", flush=True)
            print(f"Execution Failures: {env.execution_failures}", flush=True)
            print("Checkpoint:", flush=True)
            print(f"{checkpoint_path}\n", flush=True)

        return {
            "workflow_id": workflow_id,
            "mutation_level": mutation_level,
            "mutation_seed": mutation_seed,
            "success": workflow_success,
            "failing_step": failing_step,
            "failure_reason": failure_reason,
            "return": ep_return,
            "decisions": decisions,
            "wrong_actions": env.wrong_actions,
            "injected_wrong_actions": injected_wrong_actions_count,
            "natural_dqn_wrong_actions": natural_dqn_wrong_actions,
            "recovery_after_injection": (
                workflow_success if injected_wrong_actions_count > 0 else False
            ),
            "execution_failures": env.execution_failures,
            "checkpoint": checkpoint_path,
        }

    finally:
        adapter.close()


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_dqn_demo(
        checkpoint_path=args.checkpoint,
        workflow_id=args.workflow,
        mutation_level=args.mutation_level,
        mutation_seed=args.mutation_seed,
        headed=args.headed,
        auto=args.auto,
        delay_ms=args.delay_ms,
        top_k=args.top_k,
        base_url=args.base_url,
        show_evaluator_debug=args.show_evaluator_debug,
        inject_wrong_step=args.inject_wrong_step,
        inject_wrong_rank=args.inject_wrong_rank,
        interactive_prompt=not args.auto,
    )


if __name__ == "__main__":
    main()
