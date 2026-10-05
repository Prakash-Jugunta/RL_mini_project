"""
Phase 18 — Training History Loader & Artifact Inventory Utility

Loads raw training logs (episode_logs.json, training_state.json) from training_dir,
normalizes them into a clean tabular DataFrame, and computes exact artifact metadata.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Tuple, Optional


STAGE_ACTIVE_LEVELS_MAP = {
    0: "L0",
    1: "L0-L1",
    2: "L0-L2",
    3: "L0-L3",
    4: "L0-L4",
    5: "L0-L5",
}

STAGE_BOUNDARIES = [
    {"stage": 0, "start_episode": 0, "end_episode": 1000, "active_levels": "L0"},
    {"stage": 1, "start_episode": 1000, "end_episode": 2500, "active_levels": "L0-L1"},
    {"stage": 2, "start_episode": 2500, "end_episode": 4500, "active_levels": "L0-L2"},
    {"stage": 3, "start_episode": 4500, "end_episode": 7000, "active_levels": "L0-L3"},
    {"stage": 4, "start_episode": 7000, "end_episode": 10000, "active_levels": "L0-L4"},
    {"stage": 5, "start_episode": 10000, "end_episode": 15000, "active_levels": "L0-L5"},
]


def discover_training_artifacts(training_dir: str) -> Dict[str, Any]:
    """
    Programmatically inspects training_dir for episode logs, training state, and checkpoints.
    """
    abs_dir = os.path.abspath(training_dir)
    inventory = {
        "training_dir": abs_dir,
        "files_found": [],
        "has_episode_logs": False,
        "has_training_state": False,
        "has_checkpoint": False,
        "total_episodes_logged": 0,
        "total_global_steps": 0,
        "checkpoint_file_size_bytes": 0,
        "limitations": [],
    }

    if not os.path.exists(abs_dir):
        inventory["limitations"].append(f"Training directory '{abs_dir}' does not exist.")
        return inventory

    for fname in os.listdir(abs_dir):
        fpath = os.path.join(abs_dir, fname)
        if os.path.isfile(fpath):
            size_b = os.path.getsize(fpath)
            inventory["files_found"].append({"filename": fname, "size_bytes": size_b})

            if fname == "episode_logs.json":
                inventory["has_episode_logs"] = True
            elif fname == "training_state.json":
                inventory["has_training_state"] = True
            elif fname.endswith(".pt"):
                inventory["has_checkpoint"] = True
                inventory["checkpoint_file_size_bytes"] = size_b

    # Parse details if available
    logs_path = os.path.join(abs_dir, "episode_logs.json")
    if os.path.exists(logs_path):
        try:
            with open(logs_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
                inventory["total_episodes_logged"] = len(logs)
                inventory["total_global_steps"] = sum(item.get("decisions", 0) for item in logs)
        except Exception as e:
            inventory["limitations"].append(f"Failed to parse episode_logs.json: {e}")

    state_path = os.path.join(abs_dir, "training_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
                inventory["training_state"] = st
                if "total_steps" in st:
                    inventory["total_global_steps"] = st["total_steps"]
        except Exception as e:
            inventory["limitations"].append(f"Failed to parse training_state.json: {e}")

    inventory["limitations"].append("Historical wall-clock training timestamps per episode were not logged in episode_logs.json.")
    return inventory


def load_normalized_training_history(training_dir: str) -> pd.DataFrame:
    """
    Loads raw episode_logs.json and normalizes it into a unified DataFrame.
    Calculates cumulative transitions, rolling return, rolling success rate, and maps stage levels.
    """
    logs_path = os.path.join(training_dir, "episode_logs.json")
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Episode logs file not found: '{logs_path}'")

    with open(logs_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    df = pd.DataFrame(logs)

    # Standardize column names
    if "episode" not in df.columns and "global_episode_id" in df.columns:
        df["episode"] = df["global_episode_id"] + 1
    elif "episode" in df.columns:
        df["episode"] = df["episode"].astype(int)

    # Map stage levels
    if "stage_index" in df.columns:
        df["active_levels"] = df["stage_index"].map(lambda idx: STAGE_ACTIVE_LEVELS_MAP.get(int(idx), "L0-L5"))
    else:
        df["active_levels"] = "NA"

    # Cumulative transitions
    if "decisions" in df.columns:
        df["cumulative_steps"] = df["decisions"].cumsum()
    else:
        df["cumulative_steps"] = "NA"

    # Rolling metrics (100-episode window)
    if "return" in df.columns:
        df["rolling_return_100"] = df["return"].rolling(window=100, min_periods=1).mean().round(4)
    if "success" in df.columns:
        df["rolling_success_100"] = (df["success"].astype(float) * 100.0).rolling(window=100, min_periods=1).mean().round(2)
    if "recent_loss" in df.columns:
        df["rolling_loss_100"] = df["recent_loss"].rolling(window=100, min_periods=1).mean().round(4)

    # Ensure all required schema columns exist
    required_cols = [
        "episode",
        "global_episode_id",
        "stage_index",
        "active_levels",
        "workflow",
        "mutation_level",
        "mutation_seed",
        "environment_seed",
        "epsilon",
        "return",
        "success",
        "decisions",
        "cumulative_steps",
        "replay_size",
        "recent_loss",
        "rolling_return_100",
        "rolling_success_100",
        "rolling_loss_100",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = "NA"

    return df[required_cols]


def save_training_artifacts(training_dir: str, output_dir: str) -> Tuple[str, str]:
    """
    Saves artifact_inventory.json and training_history.csv into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    inventory = discover_training_artifacts(training_dir)
    inv_path = os.path.join(output_dir, "artifact_inventory.json")
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    df_hist = load_normalized_training_history(training_dir)
    hist_path = os.path.join(output_dir, "training_history.csv")
    df_hist.to_csv(hist_path, index=False)

    return (inv_path, hist_path)
