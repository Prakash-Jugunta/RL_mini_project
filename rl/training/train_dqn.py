"""
Phase 13 — Guarded DQN Training Orchestration CLI Entry Point

Executes multi-episode Masked Double DQN training when '--confirm-training' is provided.
When '--confirm-training' is absent, prints safety guard notice and exits cleanly without training.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import time
from typing import Optional, Dict, Any, List

import numpy as np
import torch

from rl.curriculum.config import CurriculumConfig, CURRICULUM_VERSION
from rl.curriculum.scheduler import CurriculumScheduler
from rl.splits.config import SplitConfig, SPLIT_VERSION
from rl.splits.splitter import assign_split
from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.config import get_epsilon_at_step
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import BrowserAdapter, MockBrowserAdapter
from rl.training.episode_runner import reset_from_episode_spec


def run_validation_evaluation(
    agent: DQNAgent,
    curriculum_scheduler: CurriculumScheduler,
    split_config: SplitConfig,
    browser_adapter: Optional[BrowserAdapter] = None,
    num_val_episodes: int = 10,
) -> Dict[str, Any]:
    """
    Executes greedy evaluation on validation episodes without replay insertion or gradient updates.
    """
    agent.set_eval_mode()

    val_env = UIRecoveryEnv(
        mode="validation",
        browser_adapter=browser_adapter or MockBrowserAdapter(max_candidates=20)
    )

    returns = []
    successes = []

    for val_ep_id in range(num_val_episodes):
        try:
            curr_ep = curriculum_scheduler.get_episode(val_ep_id)
            spec = curr_ep.episode_spec
            obs, info = val_env.reset(
                seed=spec.environment_seed + 10000,
                options={
                    "workflow_id": spec.workflow,
                    "mutation_level": spec.numeric_mutation_level(),
                    "mutation_seed": spec.mutation_seed,
                    "mode": "validation"
                }
            )
            ep_ret = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.select_action(obs, explore=False)
                obs, reward, terminated, truncated, step_info = val_env.step(action)
                ep_ret += float(reward)
            returns.append(ep_ret)
            successes.append(bool(step_info.get("workflow_completed", False)))
        except Exception:
            pass

    val_env.close()
    agent.set_train_mode()

    return {
        "val_episodes": len(returns),
        "val_mean_return": float(np.mean(returns)) if returns else 0.0,
        "val_success_rate": float(np.mean(successes)) if successes else 0.0
    }


def run_training_loop(
    args: argparse.Namespace,
    browser_adapter: Optional[BrowserAdapter] = None,
) -> Dict[str, Any]:
    """
    Executes the multi-episode DQN training loop.
    Connects CurriculumScheduler -> TRAIN split check -> UIRecoveryEnv -> DQNAgent.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    curr_config = CurriculumConfig(
        curriculum_seed=args.curriculum_seed,
        version=args.curriculum_version,
    )
    split_config = SplitConfig(split_seed=args.split_seed)
    curriculum_scheduler = CurriculumScheduler(config=curr_config, split_config=split_config)

    dqn_config = DQNConfig(seed=args.training_seed)
    agent = DQNAgent(dqn_config)

    # Resume support
    start_episode = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt_meta = agent.load_checkpoint(args.resume)
        start_episode = ckpt_meta.get("global_episode_id", 0)
        print(
            f"\n[RESUME] Loaded checkpoint '{args.resume}' | "
            f"Resuming from global episode {start_episode} "
            f"(total_steps={agent.total_steps}, learning_steps={agent.learning_steps})",
            flush=True
        )

    target_episodes = min(args.max_episodes, curriculum_scheduler.total_episodes())
    if start_episode >= target_episodes:
        print(f"Start episode {start_episode} >= target episodes {target_episodes}. Nothing to train.")
        return {"episodes_executed": 0}

    # Initialize environment
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

    recent_losses: List[float] = []
    prev_stage_idx: Optional[int] = None

    print(f"\n[TRAINING LOOP STARTED] Target: {start_episode} -> {target_episodes} episodes", flush=True)

    start_time = time.time()

    for global_ep in range(start_episode, target_episodes):
        curr_ep = curriculum_scheduler.get_episode(global_ep)
        spec = curr_ep.episode_spec

        # Phase 12 TRAIN membership validation
        split = assign_split(spec, split_config)
        if split != "train":
            raise RuntimeError(f"SPLIT INTEGRITY VIOLATION: Episode {global_ep} assigned split '{split}', expected 'train'.")

        # Zero L6 training episodes check
        if spec.numeric_mutation_level() == 6:
            raise RuntimeError(f"HELD-OUT LEAKAGE VIOLATION: Episode {global_ep} has mutation level L6!")

        # Stage transition notice
        if prev_stage_idx is not None and curr_ep.stage_index != prev_stage_idx:
            print(
                f"\n======================================================================\n"
                f" CURRICULUM STAGE TRANSITION: Stage {prev_stage_idx} -> Stage {curr_ep.stage_index}\n"
                f" Active Levels: {curr_ep.active_levels}\n"
                f"======================================================================\n",
                flush=True
            )
        prev_stage_idx = curr_ep.stage_index

        # Reset environment with episode spec
        obs, info = reset_from_episode_spec(env, spec, mode="train")

        ep_return = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            step_start = time.time()
            action = agent.select_action(obs, explore=True)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            elapsed_sec = time.time() - step_start

            if elapsed_sec > 10.0:
                print(
                    f"[SLOWNESS WARNING] episode {global_ep + 1}/{target_episodes} | "
                    f"workflow {spec.workflow} | step {step_info.get('step_index', 0)} | "
                    f"action {action} | elapsed {elapsed_sec:.2f}s",
                    flush=True
                )

            # Replay insertion (TRAIN only)
            agent.store_transition(obs, action, reward, next_obs, terminated, truncated)

            # DQN learning step
            if agent.can_learn() and agent.total_steps % agent.config.train_frequency == 0:
                diag = agent.learn()
                if diag and "loss" in diag:
                    recent_losses.append(diag["loss"])
                    if len(recent_losses) > 100:
                        recent_losses.pop(0)

            obs = next_obs
            info = step_info
            ep_return += float(reward)
            ep_steps += 1

        success = bool(info.get("workflow_success", info.get("workflow_completed", False)))
        curr_eps = get_epsilon_at_step(agent.config, agent.total_steps)
        recent_loss_val = float(np.mean(recent_losses)) if recent_losses else 0.0

        # Print per-episode progress
        print(
            f"Episode {global_ep + 1}/{target_episodes}\n"
            f"Stage: {curr_ep.stage_index}\n"
            f"Workflow: {spec.workflow}\n"
            f"Mutation level: {spec.mutation_level}\n"
            f"Mutation seed: {spec.mutation_seed}\n"
            f"Environment seed: {spec.environment_seed}\n"
            f"Epsilon: {curr_eps:.4f}\n"
            f"Return: {ep_return:.2f}\n"
            f"Success: {success}\n"
            f"Decisions: {ep_steps}\n"
            f"Replay size: {len(agent.replay_buffer)}\n"
            f"Recent loss: {recent_loss_val:.4f}\n",
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
            "epsilon": curr_eps,
            "return": ep_return,
            "success": success,
            "decisions": ep_steps,
            "replay_size": len(agent.replay_buffer),
            "recent_loss": recent_loss_val,
        }
        episode_logs.append(ep_log)

        # Checkpointing & Validation logic
        is_interval = (global_ep + 1) % 500 == 0
        is_final = (global_ep + 1) == target_episodes

        if (target_episodes >= 500 and is_interval) or is_final or target_episodes <= 50:
            ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
            agent.save_checkpoint(ckpt_path, extra_info={
                "global_episode_id": global_ep + 1,
                "stage_index": curr_ep.stage_index,
            })

            state_manifest = {
                "global_episode_id": global_ep + 1,
                "total_episodes_budget": target_episodes,
                "stage_index": curr_ep.stage_index,
                "total_steps": agent.total_steps,
                "learning_steps": agent.learning_steps,
                "replay_size": len(agent.replay_buffer),
                "epsilon": curr_eps,
                "curriculum_version": curr_config.version,
                "split_version": split_config.version,
            }
            with open(os.path.join(args.output_dir, "training_state.json"), "w") as f:
                json.dump(state_manifest, f, indent=2)

            with open(os.path.join(args.output_dir, "episode_logs.json"), "w") as f:
                json.dump(episode_logs, f, indent=2)

        # Periodic validation for canonical run
        if target_episodes >= 500 and is_interval:
            print(f"\n[VALIDATION EVALUATION @ Episode {global_ep + 1}]", flush=True)
            val_res = run_validation_evaluation(agent, curriculum_scheduler, split_config, browser_adapter=adapter_instance)
            print(f"  Val Mean Return: {val_res['val_mean_return']:.2f} | Val Success Rate: {val_res['val_success_rate']:.2%}\n", flush=True)

    env.close()
    elapsed_total = time.time() - start_time
    print(f"\n[DQN TRAINING COMPLETED CLEANLY] Executed {len(episode_logs)} episodes in {elapsed_total:.2f}s!", flush=True)

    return {
        "episodes_executed": len(episode_logs),
        "total_steps": agent.total_steps,
        "learning_steps": agent.learning_steps,
        "replay_final_size": len(agent.replay_buffer),
        "elapsed_seconds": elapsed_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 13 Guarded DQN Training Orchestration Entry Point."
    )
    parser.add_argument(
        "--curriculum-version",
        type=str,
        default=CURRICULUM_VERSION,
        help=f"Curriculum version string (default: {CURRICULUM_VERSION})."
    )
    parser.add_argument(
        "--curriculum-seed",
        type=int,
        default=1301,
        help="Base seed for curriculum schedule generation (default: 1301)."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1201,
        help="Base seed for Phase 12 split partitioning (default: 1201)."
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=2026,
        help="Base seed for DQN policy network initialization (default: 2026)."
    )
    parser.add_argument(
        "--max-episodes",
        "--episodes",
        type=int,
        default=15000,
        help="Total training episode budget override (default: 15000)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/dqn/phase13-v1",
        help="Directory to save training checkpoints, logs, and manifests."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint file for resuming training."
    )
    parser.add_argument(
        "--confirm-training",
        action="store_true",
        help="Mandatory confirmation flag required to launch training execution."
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" PHASE 13 — DQN TRAINING CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f" Curriculum Version: {args.curriculum_version}")
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
        print("DQN training was NOT started.")
        print("\nWhen you are ready to launch training manually, execute this command:")
        print(
            f"  python -u -m rl.training.train_dqn --curriculum-seed {args.curriculum_seed} "
            f"--split-seed {args.split_seed} --training-seed {args.training_seed} "
            f"--output-dir {args.output_dir} --max-episodes {args.max_episodes} --confirm-training\n"
        )
        sys.exit(0)

    run_training_loop(args)


if __name__ == "__main__":
    main()
