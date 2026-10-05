"""
Checkpointing utilities for PPO Agent.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Any, Dict, Optional
import torch

PPO_POLICY_VERSION = "phase19-ppo-v1"


def save_checkpoint(filepath: str, agent: Any, extra_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Saves PPO agent state (network, optimizer, counters, config) to disk.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        "network_state_dict": agent.network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "global_step": agent.global_step,
        "update_count": agent.update_count,
        "config": agent.config,
        "policy_version": PPO_POLICY_VERSION,
        "state_encoding_version": getattr(agent.config, "state_encoding_version", "v1"),
        "reward_version": getattr(agent.config, "reward_version", "v1"),
    }
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, agent: Any) -> Dict[str, Any]:
    """
    Loads PPO agent state from disk into the provided agent instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=agent.device, weights_only=False)
    agent.network.load_state_dict(checkpoint["network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.global_step = checkpoint.get("global_step", 0)
    agent.update_count = checkpoint.get("update_count", 0)
    return checkpoint
