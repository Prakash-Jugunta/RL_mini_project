"""
Phase 19 — Guarded PPO Training Orchestration CLI Entry Point (Corrected)

Executes candidate-aware PPO training when '--confirm-training' is provided.
When '--confirm-training' is absent, prints safety guard notice and exits cleanly without training.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import random
import sys
import time
from typing import Optional, Dict, Any, List

import numpy as np
import torch

from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION
from rl.curriculum.scheduler import CurriculumScheduler
from rl.splits import SplitConfig, SPLIT_VERSION, assign_split, materialize_canonical_splits
from rl.agents.ppo import PPOAgent, PPOConfig, save_checkpoint, load_checkpoint
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import BrowserAdapter, MockBrowserAdapter
from rl.training.episode_runner import reset_from_episode_spec


def set_global_seeds(seed: int) -> None:
    """Sets random seed across Python, NumPy, and PyTorch for training reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_validation_evaluation(
    agent: PPOAgent,
    split_config: SplitConfig,
    browser_adapter: Optional[BrowserAdapter] = None,
    num_val_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Executes greedy evaluation on canonical validation episodes (argmax action logits, no gradients).
    Validation failure exceptions are recorded explicitly without silent swallowing.
    """
    canonical_splits = materialize_canonical_splits(split_config)
    val_specs = canonical_splits["validation"]
    eval_specs = val_specs[:num_val_episodes]

    val_env = UIRecoveryEnv(
        mode="validation",
        browser_adapter=browser_adapter or MockBrowserAdapter(max_candidates=20)
    )

    requested_episodes = len(eval_specs)
    completed_episodes = 0
    failed_execution_episodes = 0
    success_count = 0
    returns: List[float] = []

    for val_ep_id, spec in enumerate(eval_specs):
        split = assign_split(spec, split_config)
        if split != "validation":
            raise RuntimeError(f"SPLIT INTEGRITY VIOLATION: Validation spec {spec.episode_id} assigned split '{split}', expected 'validation'.")

        try:
            obs, info = val_env.reset(
                seed=spec.environment_seed + 10000,
                options=spec.to_reset_options(mode="validation")
            )
            ep_ret = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _, _ = agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, step_info = val_env.step(action)
                ep_ret += float(reward)
            
            completed_episodes += 1
            returns.append(ep_ret)
            if bool(step_info.get("workflow_completed", False)):
                success_count += 1
        except Exception as exc:
            failed_execution_episodes += 1
            print(f"[VALIDATION WARNING] Episode {spec.episode_id} failed execution: {exc}", flush=True)

    val_env.close()

    success_rate = (success_count / completed_episodes) if completed_episodes > 0 else 0.0
    mean_return = float(np.mean(returns)) if returns else 0.0

    return {
        "requested_episodes": requested_episodes,
        "completed_episodes": completed_episodes,
        "failed_execution_episodes": failed_execution_episodes,
        "success_count": success_count,
        "val_success_rate": success_rate,
        "val_mean_return": mean_return,
    }


def run_training_loop(
    args: argparse.Namespace,
    browser_adapter: Optional[BrowserAdapter] = None,
) -> Dict[str, Any]:
    """
    Executes the multi-episode PPO training loop.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    set_global_seeds(args.training_seed)

    curr_config = CurriculumConfig(
        curriculum_seed=args.curriculum_seed,
        version=args.curriculum_version,
    )
    split_config = SplitConfig(split_seed=args.split_seed)
    curriculum_scheduler = CurriculumScheduler(config=curr_config, split_config=split_config)

    ppo_config = PPOConfig(device="cpu")
    agent = PPOAgent(ppo_config)
    agent.set_seed(args.training_seed)

    start_episode = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt_meta = load_checkpoint(args.resume, agent)
        start_episode = ckpt_meta.get("global_episode_id", 0)
        print(f"\n[RESUME] Loaded PPO checkpoint '{args.resume}' | Resuming from episode {start_episode}", flush=True)

    target_episodes = min(args.max_episodes, curriculum_scheduler.total_episodes())
    if start_episode >= target_episodes:
        print(f"Start episode {start_episode} >= target episodes {target_episodes}. Nothing to train.")
        return {"episodes_executed": 0}

    if browser_adapter is None:
        from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter
        adapter_instance = PlaywrightBrowserAdapter()
    else:
        adapter_instance = browser_adapter

    env = UIRecoveryEnv(
        mode="train",
        browser_adapter=adapter_instance,
    )

    episode_logs: List[Dict[str, Any]] = []
    if args.resume and os.path.exists(os.path.join(args.output_dir, "episode_logs.json")):
        try:
            with open(os.path.join(args.output_dir, "episode_logs.json"), "r") as f:
                episode_logs = json.load(f)
        except Exception:
            episode_logs = []

    latest_update_diag: Dict[str, float] = {}
    prev_stage_idx: Optional[int] = None

    print(f"\n[PPO TRAINING LOOP STARTED] Target: {start_episode} -> {target_episodes} episodes", flush=True)
    start_time = time.time()

    for global_ep in range(start_episode, target_episodes):
        curr_ep = curriculum_scheduler.get_episode(global_ep)
        spec = curr_ep.episode_spec

        split = assign_split(spec, split_config)
        if split != "train":
            raise RuntimeError(f"SPLIT INTEGRITY VIOLATION: Episode {global_ep} assigned split '{split}', expected 'train'.")

        if spec.numeric_mutation_level() == 6:
            raise RuntimeError(f"HELD-OUT LEAKAGE VIOLATION: Episode {global_ep} has mutation level L6!")

        if prev_stage_idx is not None and curr_ep.stage_index != prev_stage_idx:
            print(
                f"\n======================================================================\n"
                f" CURRICULUM STAGE TRANSITION: Stage {prev_stage_idx} -> Stage {curr_ep.stage_index}\n"
                f" Active Levels: {curr_ep.active_levels}\n"
                f"======================================================================\n",
                flush=True
            )
        prev_stage_idx = curr_ep.stage_index

        obs, info = reset_from_episode_spec(env, spec, mode="train")
        ep_return = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, value, log_prob = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            agent.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=terminated,
                truncated=truncated,
            )
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
                latest_update_diag = agent.update(last_value=last_val, last_done=(terminated or truncated))

            obs = next_obs
            info = step_info
            ep_return += float(reward)
            ep_steps += 1

        success = bool(info.get("workflow_success", info.get("workflow_completed", False)))

        print(
            f"Episode {global_ep + 1}/{target_episodes}\n"
            f"Stage: {curr_ep.stage_index}\n"
            f"Workflow: {spec.workflow}\n"
            f"Mutation level: {spec.mutation_level}\n"
            f"Return: {ep_return:.2f}\n"
            f"Success: {success}\n"
            f"Decisions: {ep_steps}\n"
            f"Policy Loss: {latest_update_diag.get('policy_loss', 0.0):.4f}\n"
            f"Value Loss: {latest_update_diag.get('value_loss', 0.0):.4f}\n"
            f"Entropy: {latest_update_diag.get('entropy', 0.0):.4f}\n",
            flush=True
        )

        ep_log = {
            "episode": global_ep + 1,
            "global_episode_id": global_ep,
            "stage_index": curr_ep.stage_index,
            "workflow": spec.workflow,
            "mutation_level": spec.mutation_level,
            "mutation_seed": spec.mutation_seed,
            "environment_seed": spec.environment_seed,
            "return": ep_return,
            "success": success,
            "decisions": ep_steps,
            "policy_loss": latest_update_diag.get("policy_loss", 0.0),
            "value_loss": latest_update_diag.get("value_loss", 0.0),
            "entropy": latest_update_diag.get("entropy", 0.0),
        }
        episode_logs.append(ep_log)

        is_interval = (global_ep + 1) % 500 == 0
        is_final = (global_ep + 1) == target_episodes

        if (target_episodes >= 500 and is_interval) or is_final or target_episodes <= 50:
            ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
            save_checkpoint(ckpt_path, agent, extra_info={
                "global_episode_id": global_ep + 1,
                "stage_index": curr_ep.stage_index,
            })

            with open(os.path.join(args.output_dir, "episode_logs.json"), "w") as f:
                json.dump(episode_logs, f, indent=2)

    # Flush partial rollout if left over
    if agent.rollout_buffer.ptr > 0:
        print(f"\n[PARTIAL ROLLOUT FLUSH] Updating remaining {agent.rollout_buffer.ptr} transitions in PPO buffer...", flush=True)
        agent.update(last_value=0.0, last_done=True)

    env.close()
    elapsed_total = time.time() - start_time
    print(f"\n[PPO TRAINING COMPLETED CLEANLY] Executed {len(episode_logs)} episodes in {elapsed_total:.2f}s!", flush=True)

    return {
        "episodes_executed": len(episode_logs),
        "global_steps": agent.global_step,
        "elapsed_seconds": elapsed_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 19 Guarded PPO Training Orchestration Entry Point.")
    parser.add_argument("--curriculum-version", type=str, default=CURRICULUM_VERSION)
    parser.add_argument("--curriculum-seed", type=int, default=1301)
    parser.add_argument("--split-seed", type=int, default=1201)
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--max-episodes", "--episodes", type=int, default=15000)
    parser.add_argument("--output-dir", type=str, default="artifacts/ppo/phase19-v1")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--confirm-training", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print(" PHASE 19 — PPO TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f" Curriculum Seed:    {args.curriculum_seed}")
    print(f" Split Seed:         {args.split_seed}")
    print(f" Training Seed:      {args.training_seed}")
    print(f" Target Budget:      {args.max_episodes} episodes")
    print(f" Output Directory:   {args.output_dir}")
    print(f" Training Confirmed: {args.confirm_training}")
    print("=" * 70)

    if not args.confirm_training:
        print("\n[TRAINING SAFETY GUARD]")
        print("Safety guard active: '--confirm-training' flag was NOT provided.")
        print("PPO training was NOT started.")
        print("\nWhen you are ready to launch training manually, execute this command:")
        print(
            f"  python -u -m rl.training.train_ppo --curriculum-seed {args.curriculum_seed} "
            f"--split-seed {args.split_seed} --training-seed {args.training_seed} "
            f"--output-dir {args.output_dir} --max-episodes {args.max_episodes} --confirm-training\n"
        )
        sys.exit(0)

    run_training_loop(args)


if __name__ == "__main__":
    main()
