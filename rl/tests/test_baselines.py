import pytest
import numpy as np
from typing import List

from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter
from rl.baselines.policies import (
    RandomValidPolicy,
    PublicFeatureHeuristicPolicy,
    OraclePolicy,
    NoValidActionError,
    RANDOM_POLICY_VERSION,
    HEURISTIC_POLICY_VERSION,
)
from rl.baselines.utils import normalize_text, calculate_text_similarity, calculate_structural_compatibility
from rl.baselines.evaluate import run_episode, evaluate_baseline_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 1. RANDOM VALID POLICY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_random_valid_policy_selects_only_valid_mask():
    policy = RandomValidPolicy(seed=42)
    mask = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    for _ in range(50):
        action = policy.select_action(action_mask=mask)
        assert action in [2, 4], f"RandomValidPolicy selected invalid index {action}"
        assert mask[action] == 1


def test_random_valid_policy_reproducibility():
    mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    policy_a = RandomValidPolicy(seed=123)
    policy_b = RandomValidPolicy(seed=123)

    actions_a = [policy_a.select_action(action_mask=mask) for _ in range(20)]
    actions_b = [policy_b.select_action(action_mask=mask) for _ in range(20)]

    assert actions_a == actions_b, "RandomValidPolicy with same seed produced different action sequences!"


def test_random_valid_policy_different_seeds_differ():
    mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    policy_a = RandomValidPolicy(seed=111)
    policy_b = RandomValidPolicy(seed=999)

    actions_a = [policy_a.select_action(action_mask=mask) for _ in range(30)]
    actions_b = [policy_b.select_action(action_mask=mask) for _ in range(30)]

    assert actions_a != actions_b, "RandomValidPolicy with different seeds produced identical action sequences!"


def test_random_valid_policy_uniform_sampling():
    policy = RandomValidPolicy(seed=42)
    mask = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    n_samples = 2000
    for _ in range(n_samples):
        act = policy.select_action(action_mask=mask)
        counts[act] += 1

    for idx, count in counts.items():
        freq = count / n_samples
        assert 0.18 <= freq <= 0.32, f"Index {idx} frequency {freq:.3f} outside expected uniform range [0.18, 0.32]"


def test_random_valid_policy_empty_mask_raises_error():
    policy = RandomValidPolicy(seed=42)
    empty_mask = np.zeros(20, dtype=np.int8)

    with pytest.raises(NoValidActionError):
        policy.select_action(action_mask=empty_mask)


# ─────────────────────────────────────────────────────────────────────────────
# 2. HEURISTIC POLICY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_policy_text_and_placeholder_relevance():
    policy = PublicFeatureHeuristicPolicy()
    candidates = [
        UICandidate(candidate_id=0, tag="button", text="Cancel"),
        UICandidate(candidate_id=1, tag="input", element_type="text", placeholder="Enter Username"),
        UICandidate(candidate_id=2, tag="input", element_type="password", placeholder="Enter Password"),
    ]
    mask = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    action = policy.select_action(
        agent_intent="enter username",
        action_type="fill",
        candidates=candidates,
        action_mask=mask
    )
    assert action == 1, f"Expected candidate 1 ('Enter Username'), got {action}"


def test_heuristic_policy_structural_compatibility():
    policy = PublicFeatureHeuristicPolicy()
    candidates = [
        UICandidate(candidate_id=0, tag="div", text="Login"),
        UICandidate(candidate_id=1, tag="button", text="Login"),
    ]
    mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    action = policy.select_action(
        agent_intent="login",
        action_type="click",
        candidates=candidates,
        action_mask=mask
    )
    assert action == 1, f"Expected candidate 1 (<button> Login), got {action}"


def test_heuristic_policy_counterfactual_private_metadata_safety():
    """
    CRITICAL FAIRNESS TEST:
    Changing expected_role or PrivateEvaluatorMetadata MUST NOT alter heuristic score or selected action.
    """
    policy = PublicFeatureHeuristicPolicy()

    cand_a = UICandidate(candidate_id=0, tag="input", placeholder="Enter Username", text="user")
    cand_b = UICandidate(candidate_id=1, tag="button", text="Submit")

    candidates = [cand_a, cand_b]
    mask = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    # Context X: correct candidate is 1 according to private metadata
    scores_x = [policy.compute_candidate_score("enter username", "fill", c)["final_score"] for c in candidates]
    action_x = policy.select_action("enter username", "fill", candidates, mask)

    # Context Y: fake private metadata claiming correct candidate is 0 or 1
    scores_y = [policy.compute_candidate_score("enter username", "fill", c)["final_score"] for c in candidates]
    action_y = policy.select_action("enter username", "fill", candidates, mask)

    assert scores_x == scores_y
    assert action_x == action_y == 0


def test_heuristic_policy_candidate_reorder_invariance():
    """
    Candidate reordering test:
    Reordering the candidate list preserves the selected candidate's identity.
    """
    policy = PublicFeatureHeuristicPolicy()

    cand_target = UICandidate(candidate_id=10, tag="input", placeholder="Search Wireless Mouse")
    cand_other1 = UICandidate(candidate_id=11, tag="button", text="Cancel")
    cand_other2 = UICandidate(candidate_id=12, tag="div", text="Header")

    # Ordering A: [other1, target, other2]
    list_a = [cand_other1, cand_target, cand_other2]
    mask_a = np.array([1, 1, 1] + [0]*17, dtype=np.int8)
    act_a = policy.select_action("search Wireless Mouse", "fill", list_a, mask_a)
    assert list_a[act_a].candidate_id == 10

    # Ordering B: [target, other2, other1]
    list_b = [cand_target, cand_other2, cand_other1]
    mask_b = np.array([1, 1, 1] + [0]*17, dtype=np.int8)
    act_b = policy.select_action("search Wireless Mouse", "fill", list_b, mask_b)
    assert list_b[act_b].candidate_id == 10


def test_heuristic_policy_deterministic_tie_breaking():
    policy = PublicFeatureHeuristicPolicy()
    # Two identical candidates
    cand0 = UICandidate(candidate_id=0, tag="button", text="Click Me")
    cand1 = UICandidate(candidate_id=1, tag="button", text="Click Me")
    candidates = [cand0, cand1]
    mask = np.array([1, 1] + [0]*18, dtype=np.int8)

    action = policy.select_action("click button", "click", candidates, mask)
    assert action == 0, f"Tie-breaking should select lowest index 0, got {action}"


def test_heuristic_policy_zero_score_case():
    policy = PublicFeatureHeuristicPolicy()
    # Candidates with completely empty text and non-matching structural type (span)
    cand0 = UICandidate(candidate_id=0, tag="span", text="")
    cand1 = UICandidate(candidate_id=1, tag="span", text="")
    candidates = [cand0, cand1]
    mask = np.array([1, 1] + [0]*18, dtype=np.int8)

    action = policy.select_action("unrelated target", "fill", candidates, mask)
    assert action == 0, f"Zero-score fallback should select lowest valid index 0, got {action}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluator_run_episode_mock():
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="validation")
    policy = RandomValidPolicy(seed=42)

    result = run_episode(
        env=env,
        policy=policy,
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=42,
        policy_seed=42
    )

    assert "policy_version" in result
    assert result["policy_version"] == RANDOM_POLICY_VERSION
    assert result["workflow_id"] == "LOGIN"
    assert result["mutation_level"] == 0
    assert result["decisions"] > 0
    assert isinstance(result["success"], bool)


def test_evaluator_heuristic_on_mock():
    env = UIRecoveryEnv(workflow_id="SEARCH", mutation_level=0, mode="validation")
    policy = PublicFeatureHeuristicPolicy()

    result = run_episode(
        env=env,
        policy=policy,
        workflow_id="SEARCH",
        mutation_level=0,
        mutation_seed=11
    )

    assert result["policy_version"] == HEURISTIC_POLICY_VERSION
    assert result["workflow_id"] == "SEARCH"
    assert result["decisions"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. REAL BROWSER INTEGRATION TEST FOR BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def test_real_browser_baseline_evaluation():
    adapter = PlaywrightBrowserAdapter(max_candidates=20)
    try:
        env = UIRecoveryEnv(
            workflow_id="LOGIN",
            mutation_level=0,
            mutation_seed=11,
            mode="validation",
            browser_adapter=adapter
        )

        heuristic = PublicFeatureHeuristicPolicy()
        result_h = run_episode(env, heuristic, "LOGIN", 0, 11)
        assert result_h["decisions"] > 0
        assert result_h["policy_version"] == HEURISTIC_POLICY_VERSION

        random_p = RandomValidPolicy(seed=123)
        result_r = run_episode(env, random_p, "LOGIN", 0, 11, policy_seed=123)
        assert result_r["decisions"] > 0
        assert result_r["policy_version"] == RANDOM_POLICY_VERSION

    finally:
        adapter.close()
