"""
Phase 14 — Comprehensive Unit & Integration Tests for DQN Evaluation Pipeline.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import tempfile
import pytest
import torch
import numpy as np


from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.checkpoint import save_checkpoint
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.splits import (
    SplitConfig,
    materialize_canonical_splits,
    validate_split_integrity,
    assign_split,
)
from rl.training.episode_spec import EpisodeSpec, HELD_OUT_MUTATION_LEVEL
from rl.evaluation.metrics import (
    compute_file_sha256,
    compute_overall_metrics,
    compute_by_workflow_metrics,
    compute_by_level_metrics,
    compute_workflow_level_matrix,
    save_split_artifacts,
    save_combined_and_plots,
    OPTIMAL_DECISIONS,
    OPTIMAL_RETURNS,
)
from rl.evaluation.evaluate_dqn import (
    evaluate_episode_spec,
    DEFAULT_CHECKPOINT_PATH,
)


def test_checkpoint_loads_correctly():
    """1. Verify DQN checkpoint loading and metadata retrieval."""
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "test_checkpoint.pt")
        extra_meta = {
            "global_episode_id": 15000,
            "total_steps": 68459,
            "curriculum_version": "phase13-v1",
            "val_success_rate": 95.0,
            "val_mean_return": 8.50,
        }
        agent.save_checkpoint(ckpt_path, extra_info=extra_meta)

        # Hash generation check
        sha = compute_file_sha256(ckpt_path)
        assert len(sha) == 64

        # Fresh agent load
        agent2 = DQNAgent(config)
        meta = agent2.load_checkpoint(ckpt_path)

        assert meta["global_episode_id"] == 15000
        assert meta["total_steps"] == 68459
        assert meta["curriculum_version"] == "phase13-v1"
        assert meta["val_success_rate"] == 95.0
        assert meta["val_mean_return"] == 8.50


def test_evaluation_uses_greedy_policy_and_no_replay_write():
    """2. Verify action selection uses greedy policy (epsilon=0) and does NOT mutate replay buffer."""
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)
    agent.set_eval_mode()

    initial_replay_size = len(agent.replay_buffer)
    assert initial_replay_size == 0

    obs = {
        "objective": np.zeros(config.objective_dim, dtype=np.float32),
        "context": np.zeros(config.context_dim, dtype=np.float32),
        "candidates": np.zeros((config.max_candidates, config.candidate_feature_dim), dtype=np.float32),
        "candidate_mask": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    }

    action = agent.select_action(obs, explore=False)
    assert action in (0, 1)  # Must select from valid mask

    # Replay buffer remains empty
    assert len(agent.replay_buffer) == 0


def test_evaluation_performs_no_optimizer_update():
    """3. Verify model parameters are 100% immutable before vs after evaluation step."""
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)
    agent.set_eval_mode()

    # Capture weight snapshot
    weights_before = {k: v.clone().cpu() for k, v in agent.online_network.state_dict().items()}

    # Run mock episode using MockBrowserAdapter
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="validation")
    spec = EpisodeSpec(episode_id=1, workflow="LOGIN", mutation_level=0, mutation_seed=11, environment_seed=42)

    _ = evaluate_episode_spec(env, agent, spec, "validation", "mock_checkpoint.pt")
    env.close()

    weights_after = {k: v.clone().cpu() for k, v in agent.online_network.state_dict().items()}

    for k in weights_before:
        assert torch.equal(weights_before[k], weights_after[k]), f"Weight '{k}' mutated during evaluation!"


def test_success_reporting_is_correct():
    """4. Verify workflow success rule (optimal trajectory -> True, wrong action -> False)."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    obs, info = env.reset(seed=42)

    # 1. Failed trajectory (wrong candidate selection)
    wrong_idx = None
    current_step = env.workflow.steps[env.current_step_index]
    for idx, cand in enumerate(env.agent_candidates):
        meta = env.browser_adapter.private_metadata_map.get(cand.candidate_id)
        if meta and meta.semantic_role != current_step.expected_role:
            wrong_idx = idx
            break

    assert wrong_idx is not None
    _, reward, terminated, truncated, fail_info = env.step(wrong_idx)
    fail_succ = bool(fail_info.get("workflow_success", fail_info.get("workflow_completed", False)))
    assert fail_succ is False

    # 2. Successful trajectory (all correct candidates selected)
    env.reset(seed=42)
    total_steps = len(env.workflow.steps)
    succ_info = None

    for _ in range(total_steps):
        step_obj = env.workflow.steps[env.current_step_index]
        correct_idx = [
            idx for idx, c in enumerate(env.agent_candidates)
            if env.browser_adapter.private_metadata_map[c.candidate_id].semantic_role == step_obj.expected_role
        ][0]
        _, _, terminated, _, succ_info = env.step(correct_idx)

    assert terminated is True
    opt_succ = bool(succ_info.get("workflow_success", succ_info.get("workflow_completed", False)))
    assert opt_succ is True
    env.close()


def test_split_integrity_and_l6_isolation():
    """5. Verify Phase 12 split disjointness, canonical counts, and L6 isolation."""
    cfg = SplitConfig()
    splits = materialize_canonical_splits(cfg)

    # Integrity assertions (must not raise ValueError)
    validate_split_integrity(splits, cfg)

    # Canonical episode counts
    assert len(splits["validation"]) == 120
    assert len(splits["test_id"]) == 120
    assert len(splits["test_l6"]) == 100

    # L6 isolation
    for s in splits["train"]:
        assert str(s.mutation_level) != HELD_OUT_MUTATION_LEVEL
    for s in splits["validation"]:
        assert str(s.mutation_level) != HELD_OUT_MUTATION_LEVEL
    for s in splits["test_id"]:
        assert str(s.mutation_level) != HELD_OUT_MUTATION_LEVEL
    for s in splits["test_l6"]:
        assert str(s.mutation_level) == HELD_OUT_MUTATION_LEVEL


def test_metrics_aggregation_matrix_and_serialization():
    """6. Verify metrics computation, workflow-level matrix, and file serialization."""
    dummy_episodes = [
        {
            "split": "validation",
            "episode_id": 1,
            "workflow": "LOGIN",
            "mutation_level": 0,
            "mutation_seed": 11,
            "environment_seed": 42,
            "success": True,
            "return": 8.80,
            "decisions": 4,
            "optimal_decisions": 4,
            "decision_overhead": 0,
            "successful_step_actions": 4,
            "wrong_step_actions": 0,
            "invalid_actions": 0,
            "execution_failures": 0,
            "terminated": True,
            "truncated": False,
            "failure_reason": None,
            "checkpoint": "test.pt",
        },
        {
            "split": "validation",
            "episode_id": 2,
            "workflow": "LOGIN",
            "mutation_level": 0,
            "mutation_seed": 11,
            "environment_seed": 43,
            "success": False,
            "return": -0.55,
            "decisions": 1,
            "optimal_decisions": 4,
            "decision_overhead": -3,
            "successful_step_actions": 0,
            "wrong_step_actions": 1,
            "invalid_actions": 0,
            "execution_failures": 0,
            "terminated": False,
            "truncated": True,
            "failure_reason": "incorrect_candidate",
            "checkpoint": "test.pt",
        },
    ]

    # Overall metrics
    overall = compute_overall_metrics(dummy_episodes)
    assert overall["episodes"] == 2
    assert overall["success_count"] == 1
    assert overall["success_rate"] == 50.0

    # Per workflow metrics
    by_wf = compute_by_workflow_metrics(dummy_episodes)
    login_wf = [w for w in by_wf if w["workflow"] == "LOGIN"][0]
    assert login_wf["episodes"] == 2
    assert login_wf["success_rate"] == 50.0

    # Per level metrics
    by_lvl = compute_by_level_metrics(dummy_episodes)
    l0_lvl = [l for l in by_lvl if l["mutation_level"] == "L0"][0]
    assert l0_lvl["episodes"] == 2

    # Workflow x Level Matrix
    matrix = compute_workflow_level_matrix(dummy_episodes)
    login_row = [r for r in matrix if r["workflow"] == "LOGIN"][0]
    assert login_row["L0"] == 50.0
    assert login_row["L1"] is None

    # Serialization test
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = save_split_artifacts(tmp_dir, dummy_episodes, "validation")
        assert os.path.exists(paths["episodes_csv"])
        assert os.path.exists(paths["episodes_json"])
        assert os.path.exists(paths["summary_json"])
        assert os.path.exists(paths["by_workflow_csv"])
        assert os.path.exists(paths["by_level_csv"])
        assert os.path.exists(paths["workflow_level_matrix_csv"])

        comb_paths = save_combined_and_plots(tmp_dir, {"validation": dummy_episodes})
        assert os.path.exists(comb_paths["combined_summary_csv"])
        assert os.path.exists(comb_paths["plot_success_rate_by_split"])


def test_episode_spec_mutation_level_type_conversion():
    """7. Regression Test: Verify EpisodeSpec.to_reset_options converts 'L0'..'L6' strings to integers 0..6."""
    for lvl_idx in range(7):
        lvl_str = f"L{lvl_idx}"
        spec = EpisodeSpec(
            episode_id=1,
            workflow="LOGIN",
            mutation_level=lvl_str,
            mutation_seed=11,
            environment_seed=42,
        )
        assert spec.numeric_mutation_level() == lvl_idx
        opts = spec.to_reset_options(mode="test")
        assert opts["mutation_level"] == lvl_idx
        assert isinstance(opts["mutation_level"], int)


def test_evaluation_resets_for_validation_test_id_and_l6():
    """8. Regression Test: Verify env.reset succeeds for validation L0, test-id L5, test-l6 L6, and L6 train protection remains strict."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="validation")

    # 1. Validation L0 spec reset succeeds
    val_spec = EpisodeSpec(episode_id=1, workflow="LOGIN", mutation_level="L0", mutation_seed=11, environment_seed=42)
    obs_val, info_val = env.reset(seed=val_spec.environment_seed, options=val_spec.to_reset_options(mode="validation"))
    assert info_val["mutation_level"] == 0
    assert info_val["mode"] == "validation"

    # 2. Test-ID L5 spec reset succeeds
    test_id_spec = EpisodeSpec(episode_id=2, workflow="SEARCH", mutation_level="L5", mutation_seed=22, environment_seed=100)
    obs_test, info_test = env.reset(seed=test_id_spec.environment_seed, options=test_id_spec.to_reset_options(mode="test"))
    assert info_test["mutation_level"] == 5
    assert info_test["mode"] == "test"

    # 3. Test-L6 L6 spec reset succeeds in test mode
    l6_spec = EpisodeSpec(episode_id=3, workflow="PROFILE", mutation_level="L6", mutation_seed=33, environment_seed=200)
    obs_l6, info_l6 = env.reset(seed=l6_spec.environment_seed, options=l6_spec.to_reset_options(mode="test"))
    assert info_l6["mutation_level"] == 6
    assert info_l6["mode"] == "test"

    # 4. L6 in train mode MUST throw ValueError (held-out protection enforced)
    with pytest.raises(ValueError, match="HELD OUT"):
        env.reset(seed=l6_spec.environment_seed, options=l6_spec.to_reset_options(mode="train"))

    env.close()


def test_cli_split_name_normalization_and_counts():
    """9. Regression Test: Verify CLI split name normalization mapping, canonical episode counts, and split integrity."""
    from rl.evaluation.evaluate_dqn import CLI_TO_INTERNAL_SPLIT, main_evaluation

    # Normalization mapping assertions
    assert CLI_TO_INTERNAL_SPLIT["validation"] == "validation"
    assert CLI_TO_INTERNAL_SPLIT["test-id"] == "test_id"
    assert CLI_TO_INTERNAL_SPLIT["test-l6"] == "test_l6"

    # Verify dry-run orchestration for all split choices
    for choice, expected_internal in [("validation", "validation"), ("test-id", "test_id"), ("test-l6", "test_l6"), ("all", "all")]:
        res = main_evaluation(split_choice=choice, run_full=False)
        assert res["split_integrity"] == "PASS"
        assert res["val_count"] == 120
        assert res["test_id_count"] == 120
        assert res["test_l6_count"] == 100


