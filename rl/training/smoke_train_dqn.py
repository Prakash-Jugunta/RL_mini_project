"""
Phase 10 — Masked Double DQN Smoke Training Script.
Performs a small controlled training run on MockBrowserAdapter and a tiny 1-episode Real Browser integration test.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
from typing import Dict, Any

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.agents.dqn import DQNAgent, DQNConfig


def run_mock_smoke_training(max_env_steps: int = 300, verbose: bool = True) -> Dict[str, Any]:
    """
    Runs a controlled smoke-training run on UIRecoveryEnv with MockBrowserAdapter.
    Proves environment interaction, valid masking, replay storage, learning machinery,
    loss stability, target network updates, and checkpointing.
    """
    if verbose:
        print("=== Phase 10: Starting Mock Browser Smoke Training ===", flush=True)

    config = DQNConfig(
        gamma=0.99,
        learning_rate=3e-4,
        batch_size=32,
        learning_starts=50,
        train_frequency=1,
        target_update_frequency=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=200,
        seed=42,
    )

    env = UIRecoveryEnv(
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
    )

    agent = DQNAgent(config)

    obs, info = env.reset()
    episodes_completed = 0
    total_reward = 0.0
    losses = []
    mean_qs = []

    start_time = time.time()

    for step in range(1, max_env_steps + 1):
        action = agent.select_action(obs, explore=True)

        # VALID-ACTION INVARIANT ASSERTION
        assert obs["candidate_mask"][action] == 1.0, f"DQN selected invalid padded action {action}!"

        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        agent.store_transition(obs, action, reward, next_obs, terminated, truncated)

        if agent.can_learn():
            diag = agent.learn()
            if diag:
                loss = diag["loss"]
                mean_q = diag["mean_q"]
                assert not np.isnan(loss) and not np.isinf(loss), "NaN/Inf detected in loss!"
                assert not np.isnan(mean_q) and not np.isinf(mean_q), "NaN/Inf detected in mean Q!"
                losses.append(loss)
                mean_qs.append(mean_q)

        if terminated or truncated:
            episodes_completed += 1
            obs, info = env.reset()
        else:
            obs = next_obs

        if verbose and step % 100 == 0:
            avg_loss = np.mean(losses[-50:]) if losses else 0.0
            avg_q = np.mean(mean_qs[-50:]) if mean_qs else 0.0
            print(
                f"  Step {step}/{max_env_steps} | "
                f"Episodes: {episodes_completed} | "
                f"Replay: {len(agent.replay_buffer)} | "
                f"Eps: {diag.get('epsilon', 1.0):.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Mean Q: {avg_q:.4f}",
                flush=True,
            )

    env.close()
    elapsed = time.time() - start_time

    # Verify Checkpoint Save & Load
    checkpoint_file = "results/checkpoints/phase10_smoke_dqn.pt"
    agent.save_checkpoint(checkpoint_file)
    assert os.path.exists(checkpoint_file), "Checkpoint file was not created!"

    agent_reloaded = DQNAgent(config)
    agent_reloaded.load_checkpoint(checkpoint_file)

    if verbose:
        print(f"Mock smoke training completed cleanly in {elapsed:.2f}s!", flush=True)

    return {
        "env_steps": max_env_steps,
        "episodes_completed": episodes_completed,
        "replay_final_size": len(agent.replay_buffer),
        "learning_steps": agent.learning_steps,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "mean_q": float(np.mean(mean_qs)) if mean_qs else 0.0,
        "elapsed_seconds": float(elapsed),
        "checkpoint_file": checkpoint_file,
    }


def run_real_browser_integration_test(verbose: bool = True) -> Dict[str, Any]:
    """
    Runs a tiny 1-episode real-browser integration smoke test with PlaywrightBrowserAdapter.
    Proves real candidate extraction -> DQN valid action selection -> Playwright execution -> reward -> transition storage.
    """
    from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter

    if verbose:
        print("=== Phase 10: Starting Real Browser Integration Smoke Test ===", flush=True)

    adapter = PlaywrightBrowserAdapter(max_candidates=20)
    try:
        env = UIRecoveryEnv(
            workflow_id="LOGIN",
            mutation_level=0,
            mutation_seed=11,
            mode="validation",
            browser_adapter=adapter,
        )

        config = DQNConfig(learning_starts=10, seed=42)
        agent = DQNAgent(config)

        obs, info = env.reset()
        steps = 0
        total_reward = 0.0

        while True:
            action = agent.select_action(obs, explore=False)
            assert obs["candidate_mask"][action] == 1.0, f"DQN selected invalid padded action {action}!"

            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)

            steps += 1
            total_reward += reward

            if terminated or truncated:
                break
            obs = next_obs

        if verbose:
            print(
                f"Real browser integration smoke test PASSED | "
                f"Steps: {steps} | "
                f"Total Reward: {total_reward:.2f} | "
                f"Replay Size: {len(agent.replay_buffer)}",
                flush=True,
            )

        return {
            "steps": steps,
            "total_reward": float(total_reward),
            "replay_size": len(agent.replay_buffer),
            "workflow_completed": bool(info.get("workflow_completed", False)),
        }

    finally:
        adapter.close()


if __name__ == "__main__":
    res_mock = run_mock_smoke_training(max_env_steps=300, verbose=True)
    res_real = run_real_browser_integration_test(verbose=True)
    print("\nPhase 10 Smoke Training & Real Browser Integration Successful!")
