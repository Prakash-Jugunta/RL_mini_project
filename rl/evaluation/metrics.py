"""
Phase 14 — Metrics Computation, Aggregation, Serialization & Plotting Engine.

Calculates overall, per-workflow, per-mutation-level metrics and workflow x level matrices,
saves publication-ready CSVs and compact JSON summaries, and generates simple Matplotlib plots.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Matplotlib configuration for headless environment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Frozen Optimal Trajectory Reference Values
OPTIMAL_DECISIONS: Dict[str, int] = {
    "LOGIN": 4,
    "SEARCH": 3,
    "PROFILE": 4,
    "CHECKOUT": 5,
}

OPTIMAL_RETURNS: Dict[str, float] = {
    "LOGIN": 8.80,
    "SEARCH": 7.85,
    "PROFILE": 8.80,
    "CHECKOUT": 9.75,
}

WORKFLOWS: List[str] = ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]
MUTATION_LEVELS: List[int] = [0, 1, 2, 3, 4, 5, 6]

EPISODE_FIELDNAMES: List[str] = [
    "split",
    "episode_id",
    "workflow",
    "mutation_level",
    "mutation_seed",
    "environment_seed",
    "success",
    "return",
    "decisions",
    "optimal_decisions",
    "decision_overhead",
    "successful_step_actions",
    "wrong_step_actions",
    "invalid_actions",
    "execution_failures",
    "terminated",
    "truncated",
    "failure_reason",
    "checkpoint",
]


def compute_file_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum for a file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_overall_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes overall summary statistics over a list of episode records.
    """
    if not episodes:
        return {
            "episodes": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "mean_return": 0.0,
            "std_return": 0.0,
            "median_return": 0.0,
            "mean_decisions": 0.0,
            "median_decisions": 0.0,
            "mean_decision_overhead": 0.0,
            "wrong_action_rate": 0.0,
            "invalid_action_rate": 0.0,
            "execution_failure_rate": 0.0,
        }

    n = len(episodes)
    successes = [1 if e["success"] else 0 for e in episodes]
    returns = [float(e["return"]) for e in episodes]
    decisions = [int(e["decisions"]) for e in episodes]
    overheads = [int(e["decision_overhead"]) for e in episodes]

    total_decisions_sum = sum(decisions)
    wrong_sum = sum(int(e.get("wrong_step_actions", 0)) for e in episodes)
    invalid_sum = sum(int(e.get("invalid_actions", 0)) for e in episodes)
    exec_fail_sum = sum(int(e.get("execution_failures", 0)) for e in episodes)

    success_count = sum(successes)
    success_rate = (success_count / n) * 100.0

    wrong_rate = (wrong_sum / total_decisions_sum * 100.0) if total_decisions_sum > 0 else 0.0
    invalid_rate = (invalid_sum / total_decisions_sum * 100.0) if total_decisions_sum > 0 else 0.0
    exec_fail_rate = (exec_fail_sum / total_decisions_sum * 100.0) if total_decisions_sum > 0 else 0.0

    return {
        "episodes": n,
        "success_count": success_count,
        "success_rate": float(round(success_rate, 4)),
        "mean_return": float(round(np.mean(returns), 4)),
        "std_return": float(round(np.std(returns), 4)),
        "median_return": float(round(np.median(returns), 4)),
        "mean_decisions": float(round(np.mean(decisions), 4)),
        "median_decisions": float(round(np.median(decisions), 4)),
        "mean_decision_overhead": float(round(np.mean(overheads), 4)),
        "wrong_action_rate": float(round(wrong_rate, 4)),
        "invalid_action_rate": float(round(invalid_rate, 4)),
        "execution_failure_rate": float(round(exec_fail_rate, 4)),
    }


def compute_by_workflow_metrics(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes per-workflow breakdown metrics for LOGIN, SEARCH, PROFILE, CHECKOUT.
    """
    results = []
    for w in WORKFLOWS:
        wf_episodes = [e for e in episodes if e["workflow"] == w]
        if not wf_episodes:
            results.append({
                "workflow": w,
                "episodes": 0,
                "success_rate": 0.0,
                "mean_return": 0.0,
                "mean_decisions": 0.0,
                "mean_decision_overhead": 0.0,
                "wrong_action_rate": 0.0,
                "execution_failure_rate": 0.0,
            })
            continue

        n = len(wf_episodes)
        success_count = sum(1 for e in wf_episodes if e["success"])
        returns = [float(e["return"]) for e in wf_episodes]
        decisions = [int(e["decisions"]) for e in wf_episodes]
        overheads = [int(e["decision_overhead"]) for e in wf_episodes]

        tot_dec = sum(decisions)
        tot_wrong = sum(int(e.get("wrong_step_actions", 0)) for e in wf_episodes)
        tot_fail = sum(int(e.get("execution_failures", 0)) for e in wf_episodes)

        results.append({
            "workflow": w,
            "episodes": n,
            "success_rate": float(round((success_count / n) * 100.0, 4)),
            "mean_return": float(round(np.mean(returns), 4)),
            "mean_decisions": float(round(np.mean(decisions), 4)),
            "mean_decision_overhead": float(round(np.mean(overheads), 4)),
            "wrong_action_rate": float(round((tot_wrong / tot_dec * 100.0) if tot_dec > 0 else 0.0, 4)),
            "execution_failure_rate": float(round((tot_fail / tot_dec * 100.0) if tot_dec > 0 else 0.0, 4)),
        })

    return results


def compute_by_level_metrics(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Computes per-mutation-level breakdown metrics for L0..L6.
    """
    results = []
    present_levels = sorted(list(set(int(e["mutation_level"]) for e in episodes))) if episodes else MUTATION_LEVELS

    for lvl in present_levels:
        lvl_episodes = [e for e in episodes if int(e["mutation_level"]) == lvl]
        if not lvl_episodes:
            continue

        n = len(lvl_episodes)
        success_count = sum(1 for e in lvl_episodes if e["success"])
        returns = [float(e["return"]) for e in lvl_episodes]
        decisions = [int(e["decisions"]) for e in lvl_episodes]

        tot_dec = sum(decisions)
        tot_wrong = sum(int(e.get("wrong_step_actions", 0)) for e in lvl_episodes)
        tot_fail = sum(int(e.get("execution_failures", 0)) for e in lvl_episodes)

        results.append({
            "mutation_level": f"L{lvl}",
            "episodes": n,
            "success_rate": float(round((success_count / n) * 100.0, 4)),
            "mean_return": float(round(np.mean(returns), 4)),
            "mean_decisions": float(round(np.mean(decisions), 4)),
            "wrong_action_rate": float(round((tot_wrong / tot_dec * 100.0) if tot_dec > 0 else 0.0, 4)),
            "execution_failure_rate": float(round((tot_fail / tot_dec * 100.0) if tot_dec > 0 else 0.0, 4)),
        })

    return results


def compute_workflow_level_matrix(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates workflow x mutation-level matrix of success rates (%).
    """
    matrix_rows = []
    for w in WORKFLOWS:
        row: Dict[str, Any] = {"workflow": w}
        for lvl in MUTATION_LEVELS:
            cell_eps = [e for e in episodes if e["workflow"] == w and int(e["mutation_level"]) == lvl]
            if not cell_eps:
                row[f"L{lvl}"] = None
            else:
                succ = sum(1 for e in cell_eps if e["success"])
                row[f"L{lvl}"] = float(round((succ / len(cell_eps)) * 100.0, 2))
        matrix_rows.append(row)
    return matrix_rows


def save_split_artifacts(
    split_dir: str,
    episodes: List[Dict[str, Any]],
    split_name: str,
) -> Dict[str, str]:
    """
    Saves episodes.csv, episodes.json, summary.json, by_workflow.csv, by_level.csv,
    and workflow_level_matrix.csv to split_dir.
    """
    os.makedirs(split_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # 1. episodes.csv
    csv_path = os.path.join(split_dir, "episodes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(episodes)
    paths["episodes_csv"] = csv_path

    # 2. episodes.json
    json_path = os.path.join(split_dir, "episodes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2)
    paths["episodes_json"] = json_path

    # 3. Overall summary.json
    overall = compute_overall_metrics(episodes)
    summary_path = os.path.join(split_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": split_name,
            "ood_declaration": "HELD-OUT OOD TEST — L6" if split_name == "test_l6" else None,
            "overall": overall,
        }, f, indent=2)
    paths["summary_json"] = summary_path

    # 4. by_workflow.csv
    by_wf = compute_by_workflow_metrics(episodes)
    wf_path = os.path.join(split_dir, "by_workflow.csv")
    if by_wf:
        with open(wf_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(by_wf[0].keys()))
            writer.writeheader()
            writer.writerows(by_wf)
    paths["by_workflow_csv"] = wf_path

    # 5. by_level.csv
    by_lvl = compute_by_level_metrics(episodes)
    lvl_path = os.path.join(split_dir, "by_level.csv")
    if by_lvl:
        with open(lvl_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(by_lvl[0].keys()))
            writer.writeheader()
            writer.writerows(by_lvl)
    paths["by_level_csv"] = lvl_path

    # 6. workflow_level_matrix.csv
    matrix = compute_workflow_level_matrix(episodes)
    matrix_path = os.path.join(split_dir, "workflow_level_matrix.csv")
    if matrix:
        fieldnames = ["workflow"] + [f"L{lvl}" for lvl in MUTATION_LEVELS]
        with open(matrix_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matrix)
    paths["workflow_level_matrix_csv"] = matrix_path

    return paths


def save_combined_and_plots(
    output_dir: str,
    split_episodes: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, str]:
    """
    Saves combined CSV summaries and generates Matplotlib plots under output_dir/plots/.
    """
    combined_dir = os.path.join(output_dir, "combined")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # 1. combined/summary.csv
    summary_rows = []
    for split_name, eps in split_episodes.items():
        o = compute_overall_metrics(eps)
        o["split"] = split_name
        summary_rows.append(o)

    comb_summary_path = os.path.join(combined_dir, "summary.csv")
    if summary_rows:
        cols = ["split"] + [k for k in summary_rows[0].keys() if k != "split"]
        with open(comb_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(summary_rows)
    paths["combined_summary_csv"] = comb_summary_path

    # 2. combined/workflow_comparison.csv
    wf_comp_rows = []
    for split_name, eps in split_episodes.items():
        by_wf = compute_by_workflow_metrics(eps)
        for row in by_wf:
            r = {"split": split_name}
            r.update(row)
            wf_comp_rows.append(r)

    wf_comp_path = os.path.join(combined_dir, "workflow_comparison.csv")
    if wf_comp_rows:
        cols = ["split"] + [k for k in wf_comp_rows[0].keys() if k != "split"]
        with open(wf_comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(wf_comp_rows)
    paths["combined_workflow_csv"] = wf_comp_path

    # 3. combined/level_comparison.csv
    lvl_comp_rows = []
    for split_name, eps in split_episodes.items():
        by_lvl = compute_by_level_metrics(eps)
        for row in by_lvl:
            r = {"split": split_name}
            r.update(row)
            lvl_comp_rows.append(r)

    lvl_comp_path = os.path.join(combined_dir, "level_comparison.csv")
    if lvl_comp_rows:
        cols = ["split"] + [k for k in lvl_comp_rows[0].keys() if k != "split"]
        with open(lvl_comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(lvl_comp_rows)
    paths["combined_level_csv"] = lvl_comp_path

    # ── MATPLOTLIB PLOTS (Loaded strictly from computed values/CSVs) ────────────

    # Plot 1: Success Rate by Split
    plt.figure(figsize=(8, 5))
    splits_list = list(split_episodes.keys())
    success_rates = [compute_overall_metrics(split_episodes[s])["success_rate"] for s in splits_list]
    colors = ["#2ca02c" if s != "test_l6" else "#d62728" for s in splits_list]

    bars = plt.bar(splits_list, success_rates, color=colors, width=0.5)
    plt.title("DQN Final: Success Rate by Split", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Split", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.ylim(-5, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

    p1_path = os.path.join(plots_dir, "success_rate_by_split.png")
    plt.tight_layout()
    plt.savefig(p1_path, dpi=300)
    plt.close()
    paths["plot_success_rate_by_split"] = p1_path

    # Plot 2: Success Rate by Workflow
    plt.figure(figsize=(10, 6))
    x = np.arange(len(WORKFLOWS))
    num_splits = len(splits_list)
    bar_width = 0.8 / max(num_splits, 1)

    for i, s in enumerate(splits_list):
        wf_metrics = {r["workflow"]: r["success_rate"] for r in compute_by_workflow_metrics(split_episodes[s])}
        vals = [wf_metrics.get(w, 0.0) for w in WORKFLOWS]
        plt.bar(x + (i - num_splits / 2.0 + 0.5) * bar_width, vals, bar_width, label=s)

    plt.title("DQN Final: Success Rate by Workflow", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Workflow ID", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.xticks(x, WORKFLOWS, fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    p2_path = os.path.join(plots_dir, "success_rate_by_workflow.png")
    plt.tight_layout()
    plt.savefig(p2_path, dpi=300)
    plt.close()
    paths["plot_success_rate_by_workflow"] = p2_path

    # Plot 3: Success Rate by Mutation Level
    plt.figure(figsize=(10, 6))
    all_levels_str = [f"L{lvl}" for lvl in MUTATION_LEVELS]

    for s in splits_list:
        lvl_metrics = {r["mutation_level"]: r["success_rate"] for r in compute_by_level_metrics(split_episodes[s])}
        x_vals = [lvl for lvl in all_levels_str if lvl in lvl_metrics]
        y_vals = [lvl_metrics[lvl] for lvl in x_vals]
        if y_vals:
            plt.plot(x_vals, y_vals, marker="o", linewidth=2.5, label=f"Split: {s}")

    plt.title("DQN Final: Success Rate by Mutation Level", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Mutation Level", fontsize=12, labelpad=10)
    plt.ylabel("Success Rate (%)", fontsize=12, labelpad=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    p3_path = os.path.join(plots_dir, "success_rate_by_level.png")
    plt.tight_layout()
    plt.savefig(p3_path, dpi=300)
    plt.close()
    paths["plot_success_rate_by_level"] = p3_path

    # Plot 4: Mean Return by Split
    plt.figure(figsize=(8, 5))
    mean_returns = [compute_overall_metrics(split_episodes[s])["mean_return"] for s in splits_list]
    bars = plt.bar(splits_list, mean_returns, color="#1f77b4", width=0.5)
    plt.title("DQN Final: Mean Episode Return by Split", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Split", fontsize=12, labelpad=10)
    plt.ylabel("Mean Return", fontsize=12, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    p4_path = os.path.join(plots_dir, "mean_return_by_split.png")
    plt.tight_layout()
    plt.savefig(p4_path, dpi=300)
    plt.close()
    paths["plot_mean_return_by_split"] = p4_path

    # Plot 5: Mean Decision Overhead by Workflow
    plt.figure(figsize=(10, 6))
    for i, s in enumerate(splits_list):
        wf_metrics = {r["workflow"]: r["mean_decision_overhead"] for r in compute_by_workflow_metrics(split_episodes[s])}
        vals = [wf_metrics.get(w, 0.0) for w in WORKFLOWS]
        plt.bar(x + (i - num_splits / 2.0 + 0.5) * bar_width, vals, bar_width, label=s)

    plt.title("DQN Final: Mean Decision Overhead by Workflow", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Workflow ID", fontsize=12, labelpad=10)
    plt.ylabel("Mean Decision Overhead (decisions - optimal)", fontsize=12, labelpad=10)
    plt.xticks(x, WORKFLOWS, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)

    p5_path = os.path.join(plots_dir, "decision_overhead_by_workflow.png")
    plt.tight_layout()
    plt.savefig(p5_path, dpi=300)
    plt.close()
    paths["plot_decision_overhead_by_workflow"] = p5_path

    return paths
