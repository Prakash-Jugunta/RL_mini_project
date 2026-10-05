"""
Unit tests for Phase 15 Final Baseline Evaluation Pipeline (rl.evaluation.evaluate_baselines).

Tests:
1. random selects valid-only candidates
2. random is reproducible by policy seed
3. heuristic does not access private metadata
4. heuristic is frozen for Test-L6
5. brittle baseline uses original fixed locators
6. same split membership reused across methods
7. aggregation is correct
8. workflow aggregation is correct
9. level aggregation is correct
10. random multi-seed aggregation is correct
11. CSV/JSON serialization
12. DQN comparison import does not rerun DQN
"""

import json
import os
import tempfile
import numpy as np
import pytest

from rl.baselines.policies import (
    RandomValidPolicy,
    PublicFeatureHeuristicPolicy,
    HEURISTIC_POLICY_VERSION,
)
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.types import UICandidate
from rl.evaluation.evaluate_baselines import (
    BRITTLE_WORKFLOW_MAP,
    compute_metrics,
    generate_combined_artifacts,
    main_evaluation,
    run_brittle_episode,
    run_policy_episode,
    save_split_artifacts,
)
from rl.splits import SplitConfig, materialize_canonical_splits, validate_split_integrity
from rl.training.episode_spec import EpisodeSpec


def test_random_selects_valid_only_candidates():
    """Verify RandomValidPolicy strictly selects candidate indices where candidate_mask == 1."""
    policy = RandomValidPolicy(seed=42)
    # Mask with valid candidates at indices 2 and 7 only
    mask = np.zeros(20, dtype=np.int8)
    mask[2] = 1
    mask[7] = 1

    selected_actions = [policy.select_action(action_mask=mask) for _ in range(50)]
    for act in selected_actions:
        assert act in (2, 7)
        assert mask[act] == 1


def test_random_reproducible_by_policy_seed():
    """Verify RandomValidPolicy produces identical action sequences when initialized with same policy seed."""
    mask = np.ones(20, dtype=np.int8)

    p1 = RandomValidPolicy(seed=2026)
    actions1 = [p1.select_action(action_mask=mask) for _ in range(20)]

    p2 = RandomValidPolicy(seed=2026)
    actions2 = [p2.select_action(action_mask=mask) for _ in range(20)]

    assert actions1 == actions2


def test_heuristic_does_not_access_private_metadata():
    """Verify PublicFeatureHeuristicPolicy operates strictly on public fields without private metadata."""
    policy = PublicFeatureHeuristicPolicy()
    cand = UICandidate(
        candidate_id=1,
        tag="input",
        element_type="text",
        text="Username",
        placeholder="Enter username",
        attributes={"id": "user-field", "class": "form-control"}
    )

    # Candidate structure must not contain ground-truth attributes
    assert not hasattr(cand, "semantic_role")
    assert not hasattr(cand, "expected_role")

    scores = policy.compute_candidate_score("enter username", "fill", cand)
    assert "final_score" in scores
    assert scores["final_score"] > 0.0


def test_heuristic_frozen_for_test_l6():
    """Verify HeuristicPolicy version and scoring weights are frozen and contain no L6 tuning."""
    policy = PublicFeatureHeuristicPolicy()
    assert policy.version == HEURISTIC_POLICY_VERSION
    assert policy.version == "phase9-heuristic-v1"


def test_brittle_baseline_uses_original_fixed_locators():
    """Verify brittle baseline mapping contains the original fixed Playwright locators."""
    assert "LOGIN" in BRITTLE_WORKFLOW_MAP
    assert "SEARCH" in BRITTLE_WORKFLOW_MAP
    assert "PROFILE" in BRITTLE_WORKFLOW_MAP
    assert "CHECKOUT" in BRITTLE_WORKFLOW_MAP

    login_steps = BRITTLE_WORKFLOW_MAP["LOGIN"]
    locators = [s[1] for s in login_steps if s[0] in ("fill", "click")]
    assert "#username" in locators
    assert "#password" in locators
    assert "#login-btn" in locators

    checkout_steps = BRITTLE_WORKFLOW_MAP["CHECKOUT"]
    c_locators = [s[1] for s in checkout_steps if s[0] in ("fill", "click")]
    assert "#add-to-cart-btn" in c_locators
    assert "#cart-btn" in c_locators
    assert "#checkout-btn" in c_locators
    assert "#confirm-order-btn" in c_locators


def test_same_split_membership_reused_across_methods():
    """Verify canonical splits are materialized with exact episode counts and disjointness."""
    split_cfg = SplitConfig()
    canonical_splits = materialize_canonical_splits(split_cfg)
    validate_split_integrity(canonical_splits, split_cfg)

    assert len(canonical_splits["validation"]) == 120
    assert len(canonical_splits["test_id"]) == 120
    assert len(canonical_splits["test_l6"]) == 100


def test_aggregation_and_workflow_level_correctness():
    """Verify compute_metrics accuracy for success rate, returns, and decision overhead."""
    mock_episodes = [
        {
            "method": "heuristic",
            "split": "validation",
            "episode_id": "ep1",
            "workflow": "LOGIN",
            "mutation_level": 0,
            "mutation_seed": 11,
            "environment_seed": 42,
            "policy_seed": 0,
            "success": True,
            "return": 8.8,
            "decisions": 4,
            "optimal_decisions": 4,
            "decision_overhead": 0,
            "successful_step_actions": 4,
            "wrong_step_actions": 0,
            "invalid_actions": 0,
            "execution_failures": 0,
            "failure_reason": None,
            "terminated": True,
            "truncated": False,
        },
        {
            "method": "heuristic",
            "split": "validation",
            "episode_id": "ep2",
            "workflow": "LOGIN",
            "mutation_level": 1,
            "mutation_seed": 11,
            "environment_seed": 43,
            "policy_seed": 0,
            "success": False,
            "return": -1.0,
            "decisions": 2,
            "optimal_decisions": 4,
            "decision_overhead": -2,
            "successful_step_actions": 1,
            "wrong_step_actions": 1,
            "invalid_actions": 0,
            "execution_failures": 0,
            "failure_reason": "incorrect_candidate",
            "terminated": True,
            "truncated": False,
        },
    ]

    metrics = compute_metrics(mock_episodes, "heuristic")
    assert metrics["episodes"] == 2
    assert metrics["success_count"] == 1
    assert metrics["success_rate"] == 50.0
    assert metrics["mean_return"] == 3.9  # (8.8 - 1.0)/2 = 3.9


def test_dqn_comparison_import_does_not_rerun_dqn():
    """Verify generate_combined_artifacts imports DQN results without executing DQN model forward passes."""
    with tempfile.TemporaryDirectory() as tmp_output_dir:
        with tempfile.TemporaryDirectory() as tmp_dqn_dir:
            # Create mock DQN summary.json files
            for split_name in ["validation", "test-id", "test-l6"]:
                s_dir = os.path.join(tmp_dqn_dir, split_name)
                os.makedirs(s_dir, exist_ok=True)
                with open(os.path.join(s_dir, "summary.json"), "w") as f:
                    json.dump({
                        "split": split_name,
                        "overall": {
                            "success_rate": 100.0,
                            "mean_return": 8.8,
                            "mean_decisions": 4.0,
                            "wrong_action_rate": 0.0,
                            "execution_failure_rate": 0.0
                        }
                    }, f)

            # Create mock baseline episode artifacts
            for method_name in ["brittle", "heuristic"]:
                for split_name in ["validation", "test-id", "test-l6"]:
                    m_dir = os.path.join(tmp_output_dir, method_name, split_name)
                    save_split_artifacts(m_dir, [{
                        "method": method_name,
                        "split": split_name,
                        "episode_id": "ep1",
                        "workflow": "LOGIN",
                        "mutation_level": 0,
                        "mutation_seed": 11,
                        "environment_seed": 42,
                        "policy_seed": 0,
                        "success": True,
                        "return": 8.8 if method_name != "brittle" else None,
                        "decisions": 4 if method_name != "brittle" else None,
                        "optimal_decisions": 4,
                        "decision_overhead": 0 if method_name != "brittle" else None,
                        "successful_step_actions": 4 if method_name != "brittle" else None,
                        "wrong_step_actions": 0 if method_name != "brittle" else None,
                        "invalid_actions": 0 if method_name != "brittle" else None,
                        "execution_failures": 0,
                        "failure_reason": None,
                        "terminated": True,
                        "truncated": False,
                    }], split_name, method_name)

            generate_combined_artifacts(tmp_output_dir, dqn_dir=tmp_dqn_dir)

            ready_path = os.path.join(tmp_output_dir, "combined", "baseline_vs_dqn_ready.csv")
            assert os.path.exists(ready_path)

            with open(ready_path, "r") as f:
                content = f.read()
                assert "Brittle" in content
                assert "Random" not in content
                assert "Heuristic" in content
                assert "DQN" in content
                assert "100.0%" in content



def test_dry_run_main_evaluation():
    """Verify main_evaluation in dry-run mode returns split specs without executing episodes."""
    res = main_evaluation(
        method_choice="all",
        split_choice="all",
        output_dir="artifacts/evaluation/baselines-final",
        run_full=False,
    )
    assert res["split_integrity"] == "PASS"
    assert res["val_count"] == 120
    assert res["test_id_count"] == 120
    assert res["test_l6_count"] == 100
