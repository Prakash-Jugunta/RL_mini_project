"""
Fast Validation Latency & Evaluator Metadata Regression Tests

Verifies:
1. Fast metadata-based validate_step() execution (<3.0 seconds wall-clock time for wrong actions).
2. Correct action validation and workflow progression.
3. Fast wrong-action validation across all 4 workflows (LOGIN, SEARCH, PROFILE, CHECKOUT).
4. Fast bounded fallback when private evaluator metadata is unavailable.
5. Zero leakage of private evaluator metadata into observation dicts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import pytest
import numpy as np

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter


@pytest.fixture(scope="module")
def real_browser_adapter():
    """Module-level Playwright browser instance for fast test suite execution."""
    adapter = PlaywrightBrowserAdapter(headless=True)
    yield adapter
    adapter.close()


def test_checkout_wrong_action_latency(real_browser_adapter):
    """
    Asserts that selecting a wrong candidate on CHECKOUT workflow completes in < 3.0s,
    yields step_success == False and reward == -0.55.
    """
    env = UIRecoveryEnv(
        workflow_id="CHECKOUT",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=real_browser_adapter,
    )

    obs, info = env.reset()
    candidate_mask = obs["candidate_mask"]
    valid_indices = np.flatnonzero(candidate_mask == 1)
    assert len(valid_indices) > 0

    # Identify a wrong candidate index (an index that is NOT the correct target element)
    # The first step of CHECKOUT is ADD_TO_CART on product detail page
    wrong_candidate_idx = None
    for idx in valid_indices:
        if idx < len(env.agent_candidates):
            cand = env.agent_candidates[idx]
            # Check if this candidate is NOT the add-to-cart button
            meta = env._private_meta[idx] if idx < len(env._private_meta) else None
            if meta and meta.semantic_role != "add-cart-action":
                wrong_candidate_idx = idx
                break

    if wrong_candidate_idx is None:
        wrong_candidate_idx = valid_indices[0]

    t0 = time.time()
    next_obs, reward, terminated, truncated, step_info = env.step(wrong_candidate_idx)
    elapsed = time.time() - t0

    assert step_info["step_success"] is False
    assert step_info["reward_total"] == pytest.approx(-0.55)
    assert step_info["step_index"] == 0  # Workflow did not advance
    assert elapsed < 3.0, f"Wrong action took {elapsed:.2f}s, expected < 3.0s"
    print(f"\nCHECKOUT wrong action completed in {elapsed:.3f}s (validate_step_sec: {env.last_timing_diagnostics['validate_step_sec']:.4f}s)")


def test_checkout_correct_action_progression(real_browser_adapter):
    """Asserts that selecting the correct candidate on CHECKOUT succeeds and advances workflow."""
    env = UIRecoveryEnv(
        workflow_id="CHECKOUT",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=real_browser_adapter,
    )

    obs, info = env.reset()

    # Find the correct candidate for ADD_TO_CART
    correct_candidate_idx = None
    for idx, meta in enumerate(env._private_meta):
        if meta.semantic_role == "add-cart-action":
            correct_candidate_idx = idx
            break

    assert correct_candidate_idx is not None, "Could not find correct candidate for add-cart-action"

    t0 = time.time()
    next_obs, reward, terminated, truncated, step_info = env.step(correct_candidate_idx)
    elapsed = time.time() - t0

    assert step_info["step_success"] is True
    assert reward == pytest.approx(0.95)
    assert step_info["step_index"] == 1  # Advanced to step 1
    assert elapsed < 3.0, f"Correct action took {elapsed:.2f}s, expected < 3.0s"


@pytest.mark.parametrize("workflow_id", ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"])
def test_all_workflows_wrong_action_latency(real_browser_adapter, workflow_id):
    """Asserts that wrong-action evaluation across all 4 workflows completes in < 3.0s."""
    env = UIRecoveryEnv(
        workflow_id=workflow_id,
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=real_browser_adapter,
    )

    obs, info = env.reset()

    # Pick a wrong candidate index
    first_step = env.workflow.steps[0]
    expected_role = first_step.expected_role

    wrong_candidate_idx = 0
    for idx, meta in enumerate(env._private_meta):
        if meta.semantic_role != expected_role:
            wrong_candidate_idx = idx
            break

    t0 = time.time()
    next_obs, reward, terminated, truncated, step_info = env.step(wrong_candidate_idx)
    elapsed = time.time() - t0

    assert elapsed < 3.0, f"Workflow {workflow_id} wrong action took {elapsed:.2f}s"
    assert env.last_timing_diagnostics["validate_step_sec"] < 0.1, f"validate_step_sec took {env.last_timing_diagnostics['validate_step_sec']:.4f}s"
    print(f"\n{workflow_id} wrong action total: {elapsed:.3f}s (validate_step: {env.last_timing_diagnostics['validate_step_sec']:.4f}s)")


def test_metadata_unavailable_fallback(real_browser_adapter):
    """
    Asserts that if private evaluator metadata is cleared, the fallback DOM lookup path
    executes cleanly with bounded timeout without hanging or waiting 30 seconds.
    """
    env = UIRecoveryEnv(
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=real_browser_adapter,
    )

    obs, info = env.reset()

    # Deliberately clear private evaluator metadata
    real_browser_adapter._last_private_meta = []

    t0 = time.time()
    next_obs, reward, terminated, truncated, step_info = env.step(0)
    elapsed = time.time() - t0

    assert elapsed < 3.0, f"Fallback validation took {elapsed:.2f}s"
    assert env.last_timing_diagnostics["validate_step_sec"] < 1.0, f"validate_step_sec fallback took {env.last_timing_diagnostics['validate_step_sec']:.4f}s"


def test_evaluator_metadata_zero_observation_leakage(real_browser_adapter):
    """
    Asserts that altering private evaluator metadata does not change the observation dict
    returned to the RL policy.
    """
    env = UIRecoveryEnv(
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=real_browser_adapter,
    )

    obs1, info1 = env.reset()

    # Create deep copy of observation arrays
    obs1_obj = np.copy(obs1["objective"])
    obs1_ctx = np.copy(obs1["context"])
    obs1_cand = np.copy(obs1["candidates"])
    obs1_mask = np.copy(obs1["candidate_mask"])

    # Mutate private evaluator metadata inside adapter
    real_browser_adapter._last_private_meta = []

    # Construct new observation
    obs2 = env._construct_observation()

    np.testing.assert_array_equal(obs1["objective"], obs2["objective"])
    np.testing.assert_array_equal(obs1["context"], obs2["context"])
    np.testing.assert_array_equal(obs1["candidates"], obs2["candidates"])
    np.testing.assert_array_equal(obs1["candidate_mask"], obs2["candidate_mask"])
