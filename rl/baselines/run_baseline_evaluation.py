"""
Phase 9 Baseline Evaluation Runner with Full Instrumentation, Checkpointing, and Resume Support.

Executes RandomValidPolicy and PublicFeatureHeuristicPolicy across 1,540 total evaluation configurations
using the real PlaywrightBrowserAdapter, incorporates frozen brittle Playwright baseline results,
saves machine-readable CSV/JSON result files and manifest, and generates publication-quality plots.
"""

import os
import json
import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Set, Tuple

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter
from rl.baselines.policies import (
    RandomValidPolicy,
    PublicFeatureHeuristicPolicy,
    RANDOM_POLICY_VERSION,
    HEURISTIC_POLICY_VERSION,
)
from rl.baselines.evaluate import run_episode
from rl.reward import REWARD_VERSION

RESULTS_DIR = "results/baselines"
HEURISTIC_CHECKPOINT_CSV = os.path.join(RESULTS_DIR, "heuristic_checkpoint.csv")
RANDOM_CHECKPOINT_CSV = os.path.join(RESULTS_DIR, "random_valid_checkpoint.csv")

WORKFLOWS = ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]
LEVELS = [0, 1, 2, 3, 4, 5, 6]
SEEDS = [11, 22, 33, 44, 55]
RANDOM_POLICY_SEEDS = list(range(10))

SLOW_EPISODE_SECONDS = 20.0

FIELDNAMES = [
    "policy_version", "workflow_id", "mutation_level", "mutation_seed", "policy_seed",
    "success", "episode_return", "decisions", "wrong_actions", "invalid_actions",
    "execution_failures", "terminated", "truncated", "reward_version"
]

# Frozen Phase 3 Brittle Playwright Baseline Results
BRITTLE_PLAYWRIGHT_ORIGINAL_6TESTS = {
    0: {"passed": 30, "total": 30, "pass_rate": 100.0},
    1: {"passed": 5,  "total": 30, "pass_rate": 16.7},
    2: {"passed": 30, "total": 30, "pass_rate": 100.0},
    3: {"passed": 30, "total": 30, "pass_rate": 100.0},
    4: {"passed": 30, "total": 30, "pass_rate": 100.0},
    5: {"passed": 15, "total": 30, "pass_rate": 50.0},
    6: {"passed": 15, "total": 30, "pass_rate": 50.0},
}

BRITTLE_PLAYWRIGHT_PRIMARY_WORKFLOWS = {
    0: {"passed": 20, "total": 20, "pass_rate": 100.0},
    1: {"passed": 0,  "total": 20, "pass_rate": 0.0},
    2: {"passed": 20, "total": 20, "pass_rate": 100.0},
    3: {"passed": 20, "total": 20, "pass_rate": 100.0},
    4: {"passed": 20, "total": 20, "pass_rate": 100.0},
    5: {"passed": 8,  "total": 20, "pass_rate": 40.0},
    6: {"passed": 8,  "total": 20, "pass_rate": 40.0},
}


def load_checkpoint(filepath: str) -> Tuple[List[Dict[str, Any]], Set[Tuple]]:
    """Loads existing episode results from a checkpoint CSV file."""
    if not os.path.exists(filepath):
        return [], set()

    episodes = []
    completed_keys = set()

    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {
                "policy_version": row["policy_version"],
                "workflow_id": row["workflow_id"],
                "mutation_level": int(row["mutation_level"]),
                "mutation_seed": int(row["mutation_seed"]),
                "policy_seed": int(row["policy_seed"]),
                "success": row["success"].lower() == "true",
                "episode_return": float(row["episode_return"]),
                "decisions": int(row["decisions"]),
                "wrong_actions": int(row["wrong_actions"]),
                "invalid_actions": int(row["invalid_actions"]),
                "execution_failures": int(row["execution_failures"]),
                "terminated": row["terminated"].lower() == "true",
                "truncated": row["truncated"].lower() == "true",
                "reward_version": row["reward_version"],
            }
            episodes.append(parsed)
            key = (
                parsed["policy_version"],
                parsed["workflow_id"],
                parsed["mutation_level"],
                parsed["mutation_seed"],
                parsed["policy_seed"],
            )
            completed_keys.add(key)

    return episodes, completed_keys


def append_checkpoint_row(filepath: str, res: Dict[str, Any]):
    """Appends a single completed episode result to the checkpoint CSV file immediately."""
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(res)
        f.flush()


def run_full_evaluation(verbose_decisions: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().isoformat()
    print(f"Starting Phase 9 Baseline Evaluation at {timestamp}...", flush=True)

    adapter = PlaywrightBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=adapter
    )

    heuristic_policy = PublicFeatureHeuristicPolicy()
    random_policy = RandomValidPolicy()

    # Load existing checkpoint data for resume support
    h_episodes, h_completed_keys = load_checkpoint(HEURISTIC_CHECKPOINT_CSV)
    r_episodes, r_completed_keys = load_checkpoint(RANDOM_CHECKPOINT_CSV)

    total_h_configs = len(WORKFLOWS) * len(LEVELS) * len(SEEDS)  # 140
    total_r_configs = total_h_configs * len(RANDOM_POLICY_SEEDS)  # 1400

    print(f"Loaded {len(h_completed_keys)} completed heuristic episodes. Remaining {total_h_configs - len(h_completed_keys)}/140.", flush=True)
    print(f"Loaded {len(r_completed_keys)} completed random valid episodes. Remaining {total_r_configs - len(r_completed_keys)}/1400.", flush=True)

    try:
        # ── 1. Evaluate Baseline B: PublicFeatureHeuristicPolicy ─────────────────
        print("\n--- Evaluating Baseline B: PublicFeatureHeuristicPolicy (140 episodes) ---", flush=True)
        h_start = time.perf_counter()
        episode_num = 0

        for w_id in WORKFLOWS:
            for lvl in LEVELS:
                for m_seed in SEEDS:
                    episode_num += 1
                    key = (HEURISTIC_POLICY_VERSION, w_id, lvl, m_seed, 0)

                    if key in h_completed_keys:
                        continue

                    print(f"[{episode_num}/140] START workflow={w_id} level=L{lvl} mutation_seed={m_seed}", flush=True)
                    ep_start = time.perf_counter()

                    res = run_episode(
                        env=env,
                        policy=heuristic_policy,
                        workflow_id=w_id,
                        mutation_level=lvl,
                        mutation_seed=m_seed,
                        verbose=verbose_decisions,
                    )

                    ep_elapsed = time.perf_counter() - ep_start
                    h_episodes.append(res)
                    h_completed_keys.add(key)
                    append_checkpoint_row(HEURISTIC_CHECKPOINT_CSV, res)

                    print(
                        f"[{episode_num}/140] DONE time={ep_elapsed:.2f}s | "
                        f"success={res['success']} | "
                        f"return={res['episode_return']:.2f} | "
                        f"decisions={res['decisions']} | "
                        f"wrong={res['wrong_actions']} | "
                        f"exec_fail={res['execution_failures']}",
                        flush=True,
                    )

                    if ep_elapsed >= SLOW_EPISODE_SECONDS:
                        print(
                            f"WARNING: SLOW EPISODE "
                            f"workflow={w_id} level=L{lvl} seed={m_seed} "
                            f"elapsed={ep_elapsed:.2f}s decisions={res['decisions']} success={res['success']}",
                            flush=True,
                        )

        print(f"Heuristic evaluation finished in {time.perf_counter() - h_start:.1f}s.", flush=True)

        # ── 2. Evaluate Baseline A: RandomValidPolicy ──────────────────────────────
        print("\n--- Evaluating Baseline A: RandomValidPolicy (1,400 episodes) ---", flush=True)
        r_start = time.perf_counter()
        episode_num = 0

        for w_id in WORKFLOWS:
            for lvl in LEVELS:
                for m_seed in SEEDS:
                    for p_seed in RANDOM_POLICY_SEEDS:
                        episode_num += 1
                        key = (RANDOM_POLICY_VERSION, w_id, lvl, m_seed, p_seed)

                        if key in r_completed_keys:
                            continue

                        if episode_num % 50 == 1 or episode_num == 1400:
                            print(f"[{episode_num}/1400] START workflow={w_id} level=L{lvl} mutation_seed={m_seed} p_seed={p_seed}", flush=True)

                        ep_start = time.perf_counter()

                        res = run_episode(
                            env=env,
                            policy=random_policy,
                            workflow_id=w_id,
                            mutation_level=lvl,
                            mutation_seed=m_seed,
                            policy_seed=p_seed,
                            verbose=verbose_decisions,
                        )

                        ep_elapsed = time.perf_counter() - ep_start
                        r_episodes.append(res)
                        r_completed_keys.add(key)
                        append_checkpoint_row(RANDOM_CHECKPOINT_CSV, res)

                        if episode_num % 50 == 0 or episode_num == 1400:
                            print(
                                f"[{episode_num}/1400] DONE time={ep_elapsed:.2f}s | "
                                f"success={res['success']} | "
                                f"return={res['episode_return']:.2f} | "
                                f"decisions={res['decisions']} | "
                                f"wrong={res['wrong_actions']} | "
                                f"exec_fail={res['execution_failures']}",
                                flush=True,
                            )

                        if ep_elapsed >= SLOW_EPISODE_SECONDS:
                            print(
                                f"WARNING: SLOW EPISODE "
                                f"workflow={w_id} level=L{lvl} seed={m_seed} p_seed={p_seed} "
                                f"elapsed={ep_elapsed:.2f}s decisions={res['decisions']} success={res['success']}",
                                flush=True,
                            )

        print(f"Random Valid evaluation finished in {time.perf_counter() - r_start:.1f}s.", flush=True)

    finally:
        env.close()

    # ── 3. Save Machine-Readable Final Result Files ────────────────────────────
    print("\n--- Saving Machine-Readable Result CSVs ---", flush=True)
    save_episodes_csv(os.path.join(RESULTS_DIR, "heuristic_episodes.csv"), h_episodes)
    save_episodes_csv(os.path.join(RESULTS_DIR, "random_valid_episodes.csv"), r_episodes)
    save_playwright_csv(os.path.join(RESULTS_DIR, "playwright_original.csv"), BRITTLE_PLAYWRIGHT_ORIGINAL_6TESTS, "Original 6-Test Suite")
    save_playwright_csv(os.path.join(RESULTS_DIR, "playwright_primary_workflows.csv"), BRITTLE_PLAYWRIGHT_PRIMARY_WORKFLOWS, "4 Primary Workflows")

    # ── 4. Compute Summaries ──────────────────────────────────────────────────
    summary_by_level = compute_summary_by_level(h_episodes, r_episodes)
    summary_by_workflow = compute_summary_by_workflow(h_episodes, r_episodes)

    save_summary_by_level_csv(os.path.join(RESULTS_DIR, "summary_by_level.csv"), summary_by_level)
    save_summary_by_workflow_csv(os.path.join(RESULTS_DIR, "summary_by_workflow.csv"), summary_by_workflow)

    # ── 5. Create Reproducibility Manifest ────────────────────────────────────
    manifest = {
        "experiment": "Phase 9 — Non-Learning Baselines Evaluation",
        "timestamp": timestamp,
        "reward_version": REWARD_VERSION,
        "policies": {
            "random_valid": {
                "version": RANDOM_POLICY_VERSION,
                "num_episodes": len(r_episodes),
                "policy_seeds": RANDOM_POLICY_SEEDS,
            },
            "heuristic": {
                "version": HEURISTIC_POLICY_VERSION,
                "num_episodes": len(h_episodes),
                "scoring_weights": {
                    "semantic_score": 0.75,
                    "structural_score": 0.20,
                    "class_score": 0.05,
                },
                "text_similarity_method": "Combined Jaccard token overlap + SequenceMatcher + Substring match",
                "held_out_L6_declaration": "L6 strictly held out. Heuristic scoring rules frozen prior to L6 evaluation.",
            },
            "brittle_playwright": {
                "source": "Phase 3 mutation degradation benchmark (summary.json)",
                "total_original_tests_per_level": 30,
                "total_primary_workflow_tests_per_level": 20,
            }
        },
        "evaluation_matrix": {
            "workflows": WORKFLOWS,
            "mutation_levels": LEVELS,
            "mutation_seeds": SEEDS,
            "total_configurations": len(WORKFLOWS) * len(LEVELS) * len(SEEDS),
        }
    }
    with open(os.path.join(RESULTS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ── 6. Generate Publication Graphs ────────────────────────────────────────
    print("\n--- Generating Publication Graphs ---", flush=True)
    generate_graph1_success_by_level(summary_by_level)
    generate_graph2_return_by_level(summary_by_level)
    generate_graph3_success_by_workflow(summary_by_workflow)

    print(f"\nPhase 9 Baseline Evaluation completed successfully! All results saved in '{RESULTS_DIR}'.", flush=True)


def save_episodes_csv(filepath: str, episodes: List[Dict[str, Any]]):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(episodes)


def save_playwright_csv(filepath: str, data: Dict[int, Dict[str, Any]], title: str):
    fieldnames = ["mutation_level", "passed", "total", "pass_rate"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lvl, d in data.items():
            writer.writerow({
                "mutation_level": lvl,
                "passed": d["passed"],
                "total": d["total"],
                "pass_rate": f"{d['pass_rate']:.1f}%"
            })


def compute_summary_by_level(h_episodes: List[Dict[str, Any]], r_episodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    summary = {}
    for lvl in LEVELS:
        h_lvl = [e for e in h_episodes if e["mutation_level"] == lvl]
        r_lvl = [e for e in r_episodes if e["mutation_level"] == lvl]

        h_success_rate = (sum(1 for e in h_lvl if e["success"]) / len(h_lvl)) * 100.0 if h_lvl else 0.0
        h_mean_return = np.mean([e["episode_return"] for e in h_lvl]) if h_lvl else 0.0
        h_mean_decisions = np.mean([e["decisions"] for e in h_lvl]) if h_lvl else 0.0

        r_successes = [e["success"] for e in r_lvl]
        r_success_rate = (sum(1 for s in r_successes if s) / len(r_successes)) * 100.0 if r_successes else 0.0
        r_returns = [e["episode_return"] for e in r_lvl]
        r_decisions = [e["decisions"] for e in r_lvl]

        seed_success_rates = []
        for p_seed in RANDOM_POLICY_SEEDS:
            seed_eps = [e for e in r_lvl if e["policy_seed"] == p_seed]
            if seed_eps:
                seed_success_rates.append((sum(1 for e in seed_eps if e["success"]) / len(seed_eps)) * 100.0)

        r_std = float(np.std(seed_success_rates)) if seed_success_rates else 0.0
        r_ci95 = float(1.96 * r_std / math.sqrt(len(seed_success_rates))) if len(seed_success_rates) > 1 else 0.0

        summary[lvl] = {
            "heuristic_success_rate": float(h_success_rate),
            "heuristic_mean_return": float(h_mean_return),
            "heuristic_mean_decisions": float(h_mean_decisions),
            "random_success_rate": float(r_success_rate),
            "random_std": float(r_std),
            "random_ci95": float(r_ci95),
            "random_mean_return": float(np.mean(r_returns)) if r_returns else 0.0,
            "random_mean_decisions": float(np.mean(r_decisions)) if r_decisions else 0.0,
            "playwright_original_pass_rate": BRITTLE_PLAYWRIGHT_ORIGINAL_6TESTS[lvl]["pass_rate"],
            "playwright_primary_pass_rate": BRITTLE_PLAYWRIGHT_PRIMARY_WORKFLOWS[lvl]["pass_rate"],
        }
    return summary


def compute_summary_by_workflow(h_episodes: List[Dict[str, Any]], r_episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for w_id in WORKFLOWS:
        h_wf = [e for e in h_episodes if e["workflow_id"] == w_id]
        r_wf = [e for e in r_episodes if e["workflow_id"] == w_id]

        h_success_rate = (sum(1 for e in h_wf if e["success"]) / len(h_wf)) * 100.0 if h_wf else 0.0
        h_mean_return = np.mean([e["episode_return"] for e in h_wf]) if h_wf else 0.0
        h_mean_decisions = np.mean([e["decisions"] for e in h_wf]) if h_wf else 0.0

        r_success_rate = (sum(1 for e in r_wf if e["success"]) / len(r_wf)) * 100.0 if r_wf else 0.0
        r_mean_return = np.mean([e["episode_return"] for e in r_wf]) if r_wf else 0.0
        r_mean_decisions = np.mean([e["decisions"] for e in r_wf]) if r_wf else 0.0

        summary[w_id] = {
            "heuristic_success_rate": float(h_success_rate),
            "heuristic_mean_return": float(h_mean_return),
            "heuristic_mean_decisions": float(h_mean_decisions),
            "random_success_rate": float(r_success_rate),
            "random_mean_return": float(r_mean_return),
            "random_mean_decisions": float(r_mean_decisions),
        }
    return summary


def save_summary_by_level_csv(filepath: str, summary: Dict[int, Dict[str, Any]]):
    fieldnames = [
        "mutation_level", "heuristic_success_rate", "heuristic_mean_return", "heuristic_mean_decisions",
        "random_success_rate", "random_std", "random_ci95", "random_mean_return", "random_mean_decisions",
        "playwright_original_pass_rate", "playwright_primary_pass_rate"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lvl, data in summary.items():
            row = {"mutation_level": lvl}
            row.update(data)
            writer.writerow(row)


def save_summary_by_workflow_csv(filepath: str, summary: Dict[str, Dict[str, Any]]):
    fieldnames = [
        "workflow_id", "heuristic_success_rate", "heuristic_mean_return", "heuristic_mean_decisions",
        "random_success_rate", "random_mean_return", "random_mean_decisions"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for w_id, data in summary.items():
            row = {"workflow_id": w_id}
            row.update(data)
            writer.writerow(row)


def generate_graph1_success_by_level(summary: Dict[int, Dict[str, Any]]):
    levels_str = [f"L{lvl}" for lvl in LEVELS]
    h_success = [summary[lvl]["heuristic_success_rate"] for lvl in LEVELS]
    r_success = [summary[lvl]["random_success_rate"] for lvl in LEVELS]
    pw_orig = [summary[lvl]["playwright_original_pass_rate"] for lvl in LEVELS]
    pw_prim = [summary[lvl]["playwright_primary_pass_rate"] for lvl in LEVELS]

    plt.figure(figsize=(10, 6))
    plt.plot(levels_str, h_success, marker='o', linewidth=2.5, color='#2ca02c', label='Heuristic Policy (Baseline B)')
    plt.plot(levels_str, r_success, marker='s', linewidth=2.5, color='#1f77b4', label='Random Valid Policy (Baseline A)')
    plt.plot(levels_str, pw_orig, marker='^', linewidth=2.0, linestyle='--', color='#ff7f0e', label='Brittle Playwright (Original 6-Test Suite)')
    plt.plot(levels_str, pw_prim, marker='v', linewidth=2.0, linestyle=':', color='#d62728', label='Brittle Playwright (4 Primary Workflows)')

    plt.title("Phase 9: Workflow Success Rate by Mutation Level", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mutation Level", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "graph1_success_by_level.png"), dpi=300)
    plt.close()


def generate_graph2_return_by_level(summary: Dict[int, Dict[str, Any]]):
    levels_str = [f"L{lvl}" for lvl in LEVELS]
    h_return = [summary[lvl]["heuristic_mean_return"] for lvl in LEVELS]
    r_return = [summary[lvl]["random_mean_return"] for lvl in LEVELS]

    plt.figure(figsize=(10, 6))
    plt.plot(levels_str, h_return, marker='o', linewidth=2.5, color='#2ca02c', label='Heuristic Policy (Baseline B)')
    plt.plot(levels_str, r_return, marker='s', linewidth=2.5, color='#1f77b4', label='Random Valid Policy (Baseline A)')

    plt.title("Phase 9: Average Episode Return by Mutation Level", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mutation Level", fontsize=12, labelpad=10)
    plt.ylabel("Mean Episode Return", fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "graph2_return_by_level.png"), dpi=300)
    plt.close()


def generate_graph3_success_by_workflow(summary: Dict[str, Dict[str, Any]]):
    x = np.arange(len(WORKFLOWS))
    width = 0.35

    h_success = [summary[w_id]["heuristic_success_rate"] for w_id in WORKFLOWS]
    r_success = [summary[w_id]["random_success_rate"] for w_id in WORKFLOWS]

    plt.figure(figsize=(9, 6))
    plt.bar(x - width/2, h_success, width, label='Heuristic Policy', color='#2ca02c')
    plt.bar(x + width/2, r_success, width, label='Random Valid Policy', color='#1f77b4')

    plt.title("Phase 9: Overall Success Rate by Workflow", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Workflow ID", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.xticks(x, WORKFLOWS, fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "graph3_success_by_workflow.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_full_evaluation(verbose_decisions=verbose)
