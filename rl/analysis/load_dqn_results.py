"""
Phase 16 — DQN Evaluation Artifact Loader & Input Integrator

Loads raw Phase 14 DQN evaluation artifacts from input_dir and normalizes them into a
unified episode-level dataset. Enforces strict input validation and split integrity.
"""

import json
import os
import math
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


EXPECTED_SPLIT_COUNTS = {
    "validation": 120,
    "test-id": 120,
    "test-l6": 100,
}

SPLIT_DIR_MAP = {
    "validation": "validation",
    "test-id": "test-id",
    "test-l6": "test-l6",
}


def compute_wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson binomial confidence interval for proportion successes / total.
    Returns (lower_bound_pct, upper_bound_pct).
    """
    if total <= 0:
        return (0.0, 0.0)

    p_hat = successes / total
    z = 1.959963984540054  # 95% confidence level

    denom = 1.0 + (z ** 2) / total
    center = (p_hat + (z ** 2) / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z ** 2) / (4 * (total ** 2))))

    lower = max(0.0, center - margin) * 100.0
    upper = min(1.0, center + margin) * 100.0
    return (round(lower, 2), round(upper, 2))


def validate_input_artifacts(input_dir: str, strict: bool = False) -> Dict[str, Any]:
    """
    Inspects input_dir to validate checkpoint details, protocol settings, episode counts,
    and split integrity.
    """
    report = {
        "status": "PASS",
        "input_dir": os.path.abspath(input_dir),
        "checkpoint_path": "NA",
        "sha256": "NA",
        "training_during_evaluation": False,
        "exploration_during_evaluation": False,
        "l6_leakage_detected": False,
        "split_counts": {},
        "missing_splits": [],
        "errors": [],
    }

    # Inspect checkpoint metadata / manifest if available
    chk_meta_path = os.path.join(input_dir, "checkpoint_metadata.json")
    manifest_path = os.path.join(input_dir, "manifest.json")

    if os.path.exists(chk_meta_path):
        with open(chk_meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            report["checkpoint_path"] = data.get("checkpoint_path", "NA")
            report["sha256"] = data.get("sha256", "NA")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            proto = data.get("protocol", {})
            if proto.get("epsilon", 0.0) > 0.0:
                report["exploration_during_evaluation"] = True
                report["errors"].append("Exploration (epsilon > 0) detected during evaluation.")

    # Check existence and counts of each canonical split
    total_episodes_found = 0
    all_seen_episode_ids = set()

    for split_name, folder_name in SPLIT_DIR_MAP.items():
        split_dir = os.path.join(input_dir, folder_name)
        if not os.path.exists(split_dir):
            report["missing_splits"].append(split_name)
            report["errors"].append(f"Missing required split directory: '{folder_name}'")
            report["status"] = "FAIL"
            continue

        episodes_csv = os.path.join(split_dir, "episodes.csv")
        if not os.path.exists(episodes_csv):
            report["errors"].append(f"Missing episodes.csv in '{folder_name}'")
            report["status"] = "FAIL"
            continue

        df = pd.read_csv(episodes_csv)
        count = len(df)
        report["split_counts"][split_name] = count
        total_episodes_found += count

        expected_count = EXPECTED_SPLIT_COUNTS[split_name]
        if count != expected_count:
            msg = f"Split '{split_name}' count mismatch: found {count}, expected {expected_count}."
            report["errors"].append(msg)
            if strict:
                report["status"] = "FAIL"

        # Check duplicate episode IDs
        if "episode_id" in df.columns:
            ep_ids = list(df["episode_id"])
            for eid in ep_ids:
                if eid in all_seen_episode_ids:
                    report["errors"].append(f"Duplicate episode_id '{eid}' detected.")
                    if strict:
                        report["status"] = "FAIL"
                all_seen_episode_ids.add(eid)

        # Check L6 isolation
        if split_name in ("validation", "test-id") and "mutation_level" in df.columns:
            l6_count = (df["mutation_level"].astype(str).str.upper().isin(["6", "L6"])).sum()
            if l6_count > 0:
                report["l6_leakage_detected"] = True
                report["errors"].append(f"L6 mutation level found in in-distribution split '{split_name}'.")
                report["status"] = "FAIL"

    if report["errors"] and strict:
        report["status"] = "FAIL"

    return report


def load_normalized_dataset(input_dir: str) -> pd.DataFrame:
    """
    Loads raw CSVs from validation, test-id, and test-l6 subdirectories in input_dir
    and concatenates them into a single normalized DataFrame.
    Missing metrics are safely populated with 'NA'.
    """
    frames = []

    for split_name, folder_name in SPLIT_DIR_MAP.items():
        split_dir = os.path.join(input_dir, folder_name)
        episodes_csv = os.path.join(split_dir, "episodes.csv")
        if os.path.exists(episodes_csv):
            df = pd.read_csv(episodes_csv)
            # Standardize split column name
            df["split"] = split_name
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No episode CSVs found in input directory '{input_dir}'")

    combined_df = pd.concat(frames, ignore_index=True)

    # Ensure required columns exist, fill missing with 'NA'
    required_cols = [
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
        "latency",
    ]

    for col in required_cols:
        if col not in combined_df.columns:
            combined_df[col] = "NA"
        else:
            combined_df[col] = combined_df[col].fillna("NA")

    # Order columns
    ordered_cols = required_cols + [c for c in combined_df.columns if c not in required_cols]
    return combined_df[ordered_cols]


def save_normalized_dataset(input_dir: str, output_dir: str) -> str:
    """
    Loads raw results from input_dir and writes artifacts/analysis/dqn-final/all_episodes.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_normalized_dataset(input_dir)
    out_path = os.path.join(output_dir, "all_episodes.csv")
    df.to_csv(out_path, index=False)
    return out_path
