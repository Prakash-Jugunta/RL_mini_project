"""
Phase 19 — PPO & A2C Mock and Real-Browser Smoke Test Suite
Verifies end-to-end integration without starting full 15,000-episode training.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import numpy as np
import torch

from rl.agents.ppo import PPOAgent, PPOConfig
from rl.agents.a2c import A2CAgent, A2CConfig
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter, PLAYWRIGHT_AVAILABLE


def run_mock_smoke_test_ppo(steps: int = 300) -> bool:
    print("\n--- Running PPO Mock Environment Smoke Test ---")
    mock_adapter = MockBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(mode="train", browser_adapter=mock_adapter)
    config = PPOConfig(rollout_steps=64, minibatch_size=32, update_epochs=2, device="cpu")
    agent = PPOAgent(config)

    obs, _ = env.reset(seed=42, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 1})
    step_count = 0
    updates_count = 0

    while step_count < steps:
        action, value, log_prob = agent.select_action(obs, eval_mode=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        assert not np.isnan(value), "NaN in predicted value!"
        assert not np.isnan(log_prob), "NaN in action log_prob!"

        agent.rollout_buffer.add(obs, action, reward, value, log_prob, terminated, truncated)
        step_count += 1
        agent.global_step += 1

        if agent.rollout_buffer.ptr >= agent.rollout_buffer.capacity:
            with torch.no_grad():
                obs_t = {
                    "objective": torch.tensor(next_obs["objective"], dtype=torch.float32).unsqueeze(0),
                    "context": torch.tensor(next_obs["context"], dtype=torch.float32).unsqueeze(0),
                    "candidates": torch.tensor(next_obs["candidates"], dtype=torch.float32).unsqueeze(0),
                    "candidate_mask": torch.tensor(next_obs["candidate_mask"], dtype=torch.float32).unsqueeze(0),
                }
                last_val = agent.network.forward_critic(obs_t).item()
            diag = agent.update(last_value=last_val, last_done=(terminated or truncated))
            updates_count += 1
            assert not np.isnan(diag["policy_loss"]), "NaN in PPO policy loss!"
            assert not np.isnan(diag["value_loss"]), "NaN in PPO value loss!"

        if terminated or truncated:
            obs, _ = env.reset(seed=42 + step_count, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 1})
        else:
            obs = next_obs

    env.close()
    print(f"PPO Mock Smoke Test PASSED! ({step_count} steps, {updates_count} updates executed cleanly)")
    return True


def run_mock_smoke_test_a2c(steps: int = 300) -> bool:
    print("\n--- Running A2C Mock Environment Smoke Test ---")
    mock_adapter = MockBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(mode="train", browser_adapter=mock_adapter)
    config = A2CConfig(rollout_steps=64, device="cpu")
    agent = A2CAgent(config)

    obs, _ = env.reset(seed=42, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 1})
    step_count = 0
    updates_count = 0

    while step_count < steps:
        action, value, log_prob = agent.select_action(obs, eval_mode=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        assert not np.isnan(value), "NaN in predicted value!"
        assert not np.isnan(log_prob), "NaN in action log_prob!"

        agent.rollout_buffer.add(obs, action, reward, value, terminated, truncated)
        step_count += 1
        agent.global_step += 1

        if agent.rollout_buffer.ptr >= agent.rollout_buffer.capacity:
            with torch.no_grad():
                obs_t = {
                    "objective": torch.tensor(next_obs["objective"], dtype=torch.float32).unsqueeze(0),
                    "context": torch.tensor(next_obs["context"], dtype=torch.float32).unsqueeze(0),
                    "candidates": torch.tensor(next_obs["candidates"], dtype=torch.float32).unsqueeze(0),
                    "candidate_mask": torch.tensor(next_obs["candidate_mask"], dtype=torch.float32).unsqueeze(0),
                }
                last_val = agent.network.forward_critic(obs_t).item()
            diag = agent.update(last_value=last_val, last_done=(terminated or truncated))
            updates_count += 1
            assert not np.isnan(diag["actor_loss"]), "NaN in A2C actor loss!"
            assert not np.isnan(diag["critic_loss"]), "NaN in A2C critic loss!"

        if terminated or truncated:
            obs, _ = env.reset(seed=42 + step_count, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 1})
        else:
            obs = next_obs

    env.close()
    print(f"A2C Mock Smoke Test PASSED! ({step_count} steps, {updates_count} updates executed cleanly)")
    return True


def run_real_browser_smoke_test_ppo(episodes: int = 5) -> bool:
    print("\n--- Running Real Playwright Browser PPO Smoke Test ---")
    if not PLAYWRIGHT_AVAILABLE:
        print("[SKIP] Playwright is not available.")
        return False

    try:
        pw_adapter = PlaywrightBrowserAdapter(base_url="http://localhost:3000", max_candidates=20, headless=True)
        env = UIRecoveryEnv(mode="train", browser_adapter=pw_adapter)
        agent = PPOAgent(PPOConfig(rollout_steps=16, minibatch_size=8, update_epochs=2))

        for ep in range(episodes):
            obs, info = env.reset(seed=100 + ep, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 11})
            done = False
            truncated = False
            ep_ret = 0.0

            while not (done or truncated):
                action, val, logp = agent.select_action(obs, eval_mode=False)
                next_obs, reward, done, truncated, step_info = env.step(action)
                agent.rollout_buffer.add(obs, action, reward, val, logp, done, truncated)
                ep_ret += reward
                obs = next_obs

                if agent.rollout_buffer.ptr >= agent.rollout_buffer.capacity:
                    agent.update(last_value=0.0, last_done=(done or truncated))

            print(f"  PPO Real Episode {ep+1}/{episodes} finished | Return: {ep_ret:.2f} | Success: {info.get('workflow_completed', False)}")

        env.close()
        print("PPO Real Playwright Browser Smoke Test PASSED!")
        return True
    except Exception as e:
        print(f"[REAL BROWSER ERROR] PPO smoke test encountered: {e}")
        return False


def run_real_browser_smoke_test_a2c(episodes: int = 5) -> bool:
    print("\n--- Running Real Playwright Browser A2C Smoke Test ---")
    if not PLAYWRIGHT_AVAILABLE:
        print("[SKIP] Playwright is not available.")
        return False

    try:
        pw_adapter = PlaywrightBrowserAdapter(base_url="http://localhost:3000", max_candidates=20, headless=True)
        env = UIRecoveryEnv(mode="train", browser_adapter=pw_adapter)
        agent = A2CAgent(A2CConfig(rollout_steps=16))

        for ep in range(episodes):
            obs, info = env.reset(seed=200 + ep, options={"workflow_id": "LOGIN", "mutation_level": 0, "mutation_seed": 11})
            done = False
            truncated = False
            ep_ret = 0.0

            while not (done or truncated):
                action, val, logp = agent.select_action(obs, eval_mode=False)
                next_obs, reward, done, truncated, step_info = env.step(action)
                agent.rollout_buffer.add(obs, action, reward, val, done, truncated)
                ep_ret += reward
                obs = next_obs

                if agent.rollout_buffer.ptr >= agent.rollout_buffer.capacity:
                    agent.update(last_value=0.0, last_done=(done or truncated))

            print(f"  A2C Real Episode {ep+1}/{episodes} finished | Return: {ep_ret:.2f} | Success: {info.get('workflow_completed', False)}")

        env.close()
        print("A2C Real Playwright Browser Smoke Test PASSED!")
        return True
    except Exception as e:
        print(f"[REAL BROWSER ERROR] A2C smoke test encountered: {e}")
        return False


def main():
    print("=" * 60)
    print(" PHASE 19 — PPO & A2C SMOKE TESTING")
    print("=" * 60)
    ppo_mock_pass = run_mock_smoke_test_ppo(steps=200)
    a2c_mock_pass = run_mock_smoke_test_a2c(steps=200)

    ppo_real_pass = run_real_browser_smoke_test_ppo(episodes=2)
    a2c_real_pass = run_real_browser_smoke_test_a2c(episodes=2)

    print("\n" + "=" * 60)
    print(" SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f" PPO Mock Environment Smoke Test: {'PASS' if ppo_mock_pass else 'FAIL'}")
    print(f" A2C Mock Environment Smoke Test: {'PASS' if a2c_mock_pass else 'FAIL'}")
    print(f" PPO Real Playwright Browser Smoke Test: {'PASS' if ppo_real_pass else 'FAIL'}")
    print(f" A2C Real Playwright Browser Smoke Test: {'PASS' if a2c_real_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
