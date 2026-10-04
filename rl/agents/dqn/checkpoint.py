"""
Checkpointing utilities for Masked Double DQN Agent.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from typing import Any, Dict, Optional

DQN_POLICY_VERSION = "phase10-dqn-state-v2"


def save_checkpoint(filepath: str, agent: Any, extra_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Saves the agent's full state (online net, target net, optimizer, counters, config) to disk.
    Optionally includes extra training state metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    checkpoint = {
        "online_network_state_dict": agent.online_network.state_dict(),
        "target_network_state_dict": agent.target_network.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "total_steps": agent.total_steps,
        "learning_steps": agent.learning_steps,
        "config": agent.config,
        "policy_version": DQN_POLICY_VERSION,
        "state_encoding_version": getattr(agent.config, "state_encoding_version", None),
    }
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, agent: Any) -> Dict[str, Any]:
    """
    Loads agent state from disk into the provided agent instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=agent.device, weights_only=False)
    expected_policy = getattr(agent, "version", None)
    actual_policy = checkpoint.get("policy_version")
    if actual_policy != expected_policy:
        raise ValueError(
            f"Incompatible DQN checkpoint policy version: expected {expected_policy!r}, "
            f"found {actual_policy!r}. Start a fresh model for the new state encoding."
        )
    expected_state = getattr(agent.config, "state_encoding_version", None)
    actual_state = checkpoint.get("state_encoding_version")
    if actual_state != expected_state:
        raise ValueError(
            f"Incompatible DQN checkpoint state encoding: expected {expected_state!r}, "
            f"found {actual_state!r}. Start a fresh model for the new state encoding."
        )
    agent.online_network.load_state_dict(checkpoint["online_network_state_dict"])
    agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.total_steps = checkpoint.get("total_steps", 0)
    agent.learning_steps = checkpoint.get("learning_steps", 0)
    return checkpoint
