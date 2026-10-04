import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.types import UICandidate, PrivateEvaluatorMetadata, HELD_OUT_MUTATION_LEVEL, sanitize_attributes
from rl.env.workflow import WORKFLOW_REGISTRY, WORKFLOW_LOGIN, WORKFLOW_SEARCH, WORKFLOW_PROFILE, WORKFLOW_CHECKOUT
from rl.env.browser_adapter import MockBrowserAdapter


def test_gymnasium_check_env():
    """1. Gymnasium Environment Checker Compliance Test."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    check_env(env)
    env.close()


def test_frozen_phase12_workflow_definitions():
    """2. Verify Workflow Definitions Match Frozen Phase 1/2 Parameters Exactly."""
    # LOGIN
    login = WORKFLOW_REGISTRY["LOGIN"]
    assert login.start_path == "/login"
    assert login.steps[0].value == "testuser"
    assert login.steps[1].value == "password123"
    assert login.steps[3].value == "Welcome, testuser"

    # SEARCH
    search = WORKFLOW_REGISTRY["SEARCH"]
    assert search.start_path == "/products"
    assert search.steps[0].value == "Wireless Mouse"
    assert search.steps[3].value == "/products/wireless-mouse"

    # PROFILE
    profile = WORKFLOW_REGISTRY["PROFILE"]
    assert profile.start_path == "/profile"
    assert profile.steps[0].value == "Test User"
    assert profile.steps[1].value == "test@example.com"
    assert profile.steps[3].value == "Profile updated successfully"

    # CHECKOUT
    checkout = WORKFLOW_REGISTRY["CHECKOUT"]
    assert checkout.start_path == "/products/wireless-mouse"
    assert checkout.steps[0].expected_role == "add-cart-action"
    assert checkout.steps[1].expected_role == "nav-cart"
    assert checkout.steps[2].expected_role == "checkout-action"
    assert checkout.steps[3].expected_role == "confirm-order-action"
    assert checkout.steps[4].value == "Order placed successfully"


def test_reset():
    """3. Reset Contract Test."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=1, mutation_seed=11, mode="train")
    obs, info = env.reset(seed=42)

    assert env.observation_space.contains(obs) is True
    assert isinstance(obs, dict)
    assert "objective" in obs and "candidates" in obs and "candidate_mask" in obs and "context" in obs

    assert info["workflow_id"] == "LOGIN"
    assert info["step_index"] == 0
    assert info["episode_steps"] == 0
    assert info["wrong_actions"] == 0
    assert info["mutation_level"] == 1
    assert info["mutation_seed"] == 11
    assert info["mode"] == "train"

    env.close()


def test_step_and_completion():
    """4. Step Transitions & Episode Completion Test."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    obs, info = env.reset(seed=42)

    num_steps = len(env.workflow.steps)
    total_reward = 0.0

    for _ in range(num_steps):
        correct_idx = None
        current_step = env.workflow.steps[env.current_step_index]
        for idx, cand in enumerate(env.agent_candidates):
            meta = env.browser_adapter.private_metadata_map.get(cand.candidate_id)
            if meta and meta.semantic_role == current_step.expected_role:
                correct_idx = idx
                break

        assert correct_idx is not None, "Target candidate must be present in candidates list"

        obs, reward, terminated, truncated, info = env.step(correct_idx)
        total_reward += reward
        assert info["action_success"] is True

    assert terminated is True
    assert truncated is False
    assert info["workflow_completed"] is True
    # Phase 8 Reward: 3 * (+0.95) + (+5.95) = +8.80
    assert total_reward == pytest.approx(8.80)

    env.close()


def test_terminal_info_step_id():
    """5. Terminal Transition Info Test (step_id is None when workflow_completed is True)."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    env.reset(seed=42)

    step_info = None
    for _ in range(len(env.workflow.steps)):
        current_step = env.workflow.steps[env.current_step_index]
        correct_idx = [
            idx for idx, c in enumerate(env.agent_candidates)
            if env.browser_adapter.private_metadata_map[c.candidate_id].semantic_role == current_step.expected_role
        ][0]
        _, _, terminated, _, step_info = env.step(correct_idx)

    assert terminated is True
    assert step_info["workflow_completed"] is True
    assert step_info["step_id"] is None, "step_id must be None on terminal transition"

    env.close()


def test_truncation():
    """6. Truncation Limit Test."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, max_episode_steps=3, mode="train")
    env.reset(seed=42)

    terminated = False
    truncated = False
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action=9)

    assert truncated is True
    assert terminated is False
    assert info["episode_steps"] == 3

    env.close()


def test_determinism_and_seed_clarification():
    """7. Determinism & Seed Separation Test (Environment Reset Seed vs Mutation Seed)."""
    # Test A: Same reset seed -> Same candidate ordering
    env1 = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=1, mutation_seed=11, mode="train")
    obs1, _ = env1.reset(seed=42)
    cands1 = [c.candidate_id for c in env1.agent_candidates]

    env2 = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=1, mutation_seed=11, mode="train")
    obs2, _ = env2.reset(seed=42)
    cands2 = [c.candidate_id for c in env2.agent_candidates]

    assert np.array_equal(obs1["objective"], obs2["objective"])
    assert np.array_equal(obs1["candidates"], obs2["candidates"])
    assert np.array_equal(obs1["candidate_mask"], obs2["candidate_mask"])
    assert np.array_equal(obs1["context"], obs2["context"])
    assert cands1 == cands2
    assert cands1 == cands2

    # Test B: Different reset seeds -> Different candidate ordering
    env3 = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=1, mutation_seed=11, mode="train")
    env3.reset(seed=999)
    cands3 = [c.candidate_id for c in env3.agent_candidates]
    assert cands1 != cands3, "Candidate ordering should change when environment reset seed changes"

    # Test C: Changing mutation_seed changes mock text representation deterministically
    adapter_s11 = MockBrowserAdapter()
    adapter_s11.reset(start_path="/login", level=1, seed=11)
    cands_s11, _ = adapter_s11.get_candidates("ENTER_USERNAME", "username-input")
    target_s11_text = [c.text for c in cands_s11 if "Mutated" in c.text][0]

    adapter_s22 = MockBrowserAdapter()
    adapter_s22.reset(start_path="/login", level=1, seed=22)
    cands_s22, _ = adapter_s22.get_candidates("ENTER_USERNAME", "username-input")
    target_s22_text = [c.text for c in cands_s22 if "Mutated" in c.text][0]

    assert target_s11_text != target_s22_text, "Mock mutation text must change when mutation_seed changes"

    env1.close()
    env2.close()
    env3.close()


def test_hardened_attribute_leakage_protection():
    """8. Production Attribute Sanitization Guard Test."""
    dirty_attributes = {
        "data-semantic-role": "username-input",
        "semantic_role": "secret-role",
        "expected_role": "password-input",
        "is_correct": "true",
        "ground_truth": "hidden",
        "class": "btn-primary",
        "id": "input-123"
    }

    # 1. Direct function check
    clean = sanitize_attributes(dirty_attributes)
    assert "data-semantic-role" not in clean
    assert "semantic_role" not in clean
    assert "expected_role" not in clean
    assert "is_correct" not in clean
    assert "ground_truth" not in clean
    assert clean == {"class": "btn-primary", "id": "input-123"}

    # 2. Production UICandidate __post_init__ check
    cand = UICandidate(candidate_id=1, tag="input", attributes=dirty_attributes)
    assert "data-semantic-role" not in cand.attributes
    assert "semantic_role" not in cand.attributes
    assert cand.attributes == {"class": "btn-primary", "id": "input-123"}


def test_reset_options_validation():
    """9. Strict reset(options=...) Validation Test."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    env.reset(seed=42)

    # Invalid workflow_id in options
    with pytest.raises(ValueError, match="Invalid workflow_id"):
        env.reset(options={"workflow_id": "NON_EXISTENT_WORKFLOW"})

    # Invalid mode in options
    with pytest.raises(ValueError, match="Invalid mode"):
        env.reset(options={"mode": "invalid_mode"})

    # Invalid mutation_level in options
    with pytest.raises(ValueError, match="Invalid mutation_level"):
        env.reset(options={"mutation_level": 99})

    # TRAIN + Level 6 in options
    with pytest.raises(ValueError, match="HELD OUT"):
        env.reset(options={"mutation_level": 6, "mode": "train"})

    # TEST + Level 6 in options -> Succeeded
    obs, info = env.reset(options={"mutation_level": 6, "mode": "test"})
    assert info["mutation_level"] == 6
    assert info["mode"] == "test"

    env.close()


    env.close()


def _get_correct_idx(env):
    current_step = env.workflow.steps[env.current_step_index]
    for idx, cand in enumerate(env.agent_candidates):
        meta = env.browser_adapter.private_metadata_map.get(cand.candidate_id)
        if meta and meta.semantic_role == current_step.expected_role:
            return idx
    return 0


def test_page_state_verification_automatic_transition():
    """Verify VERIFY_PRODUCT_PAGE (url_equals) is automatically evaluated without candidate decisions."""
    env = UIRecoveryEnv(workflow_id="SEARCH", mutation_level=0, mode="train")
    obs, info = env.reset(seed=42)

    # SEARCH steps: ENTER_SEARCH_QUERY (0), CLICK_SEARCH (1), SELECT_PRODUCT (2), VERIFY_PRODUCT_PAGE (3)
    assert info["step_index"] == 0

    # Step 0: ENTER_SEARCH_QUERY
    obs, reward, terminated, truncated, info = env.step(_get_correct_idx(env))
    assert info["step_index"] == 1

    # Step 1: CLICK_SEARCH
    obs, reward, terminated, truncated, info = env.step(_get_correct_idx(env))
    assert info["step_index"] == 2

    # Step 2: SELECT_PRODUCT -> advancing should automatically evaluate VERIFY_PRODUCT_PAGE (3) and complete workflow (4)
    obs, reward, terminated, truncated, info = env.step(_get_correct_idx(env))

    assert info["action_success"] is True
    assert info["step_index"] == 4  # Advanced past step 3 automatically
    assert terminated is True
    assert info["workflow_completed"] is True

    env.close()


def test_element_verification_requires_candidates():
    """Verify element verification steps (VERIFY_DASHBOARD, VERIFY_SUCCESS, VERIFY_ORDER_SUCCESS) produce candidates."""
    # LOGIN workflow: VERIFY_DASHBOARD is step 3 (text_contains)
    env_login = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    env_login.reset(seed=42)
    env_login.step(_get_correct_idx(env_login))  # ENTER_USERNAME
    env_login.step(_get_correct_idx(env_login))  # ENTER_PASSWORD
    obs, reward, terminated, truncated, info = env_login.step(_get_correct_idx(env_login))  # CLICK_LOGIN -> advances to VERIFY_DASHBOARD (step 3)

    assert info["step_index"] == 3
    assert terminated is False
    assert len(env_login.agent_candidates) > 0  # Candidates extracted for element verification
    env_login.close()


def test_url_verification_failure():
    """Verify validate_page_state returns False when URL does not match."""
    adapter = MockBrowserAdapter()
    adapter.reset(start_path="/products", level=0, seed=11)
    # Path is /products, but expecting /products/wireless-mouse
    assert adapter.validate_page_state("url_equals", "/products/wireless-mouse") is False
    # Set path to /products/wireless-mouse
    adapter.current_path = "/products/wireless-mouse"
    assert adapter.validate_page_state("url_equals", "/products/wireless-mouse") is True
    adapter.close()


