import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from rl.env.workflow import WORKFLOW_LOGIN, WORKFLOW_SEARCH, WORKFLOW_PROFILE, WORKFLOW_CHECKOUT
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.reward import RewardConfig, RewardCalculator, RewardBreakdown, REWARD_VERSION


def test_reward_config_defaults():
    config = RewardConfig()
    assert config.step_cost == -0.05
    assert config.progress_reward == 1.00
    assert config.wrong_action_penalty == -0.50
    assert config.invalid_action_penalty == -1.00
    assert config.completion_bonus == 5.00
    assert config.failure_penalty == -2.00


def test_reward_breakdown_sum():
    bd = RewardBreakdown(
        step_cost=-0.05,
        progress=1.00,
        wrong_action=0.0,
        invalid_action=0.0,
        completion=5.00,
        failure=0.0
    )
    assert bd.total == pytest.approx(5.95)
    d = bd.to_dict()
    assert d["total"] == pytest.approx(5.95)
    assert d["step_cost"] == -0.05
    assert d["completion"] == 5.00


def test_correct_nonterminal_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=True, step_success=True, workflow_completed=False, truncated=False)
    assert bd.total == pytest.approx(0.95)
    assert bd.step_cost == -0.05
    assert bd.progress == 1.00
    assert bd.wrong_action == 0.0
    assert bd.completion == 0.0


def test_wrong_real_candidate_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=True, step_success=False, workflow_completed=False, truncated=False)
    assert bd.total == pytest.approx(-0.55)
    assert bd.step_cost == -0.05
    assert bd.wrong_action == -0.50


def test_invalid_action_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=False, step_success=False, workflow_completed=False, truncated=False)
    assert bd.total == pytest.approx(-1.05)
    assert bd.step_cost == -0.05
    assert bd.invalid_action == -1.00


def test_final_success_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=True, step_success=True, workflow_completed=True, truncated=False)
    assert bd.total == pytest.approx(5.95)
    assert bd.step_cost == -0.05
    assert bd.progress == 1.00
    assert bd.completion == 5.00


def test_wrong_final_budget_action_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=True, step_success=False, workflow_completed=False, truncated=True)
    assert bd.total == pytest.approx(-2.55)
    assert bd.step_cost == -0.05
    assert bd.wrong_action == -0.50
    assert bd.failure == -2.00


def test_invalid_final_budget_action_reward():
    calc = RewardCalculator()
    bd = calc.calculate(action_valid=False, step_success=False, workflow_completed=False, truncated=True)
    assert bd.total == pytest.approx(-3.05)
    assert bd.step_cost == -0.05
    assert bd.invalid_action == -1.00
    assert bd.failure == -2.00


def test_completion_and_failure_mutually_exclusive():
    calc = RewardCalculator()
    # Even if max steps reached on the final completing step, completion bonus applies and failure penalty is 0.0
    bd = calc.calculate(action_valid=True, step_success=True, workflow_completed=True, truncated=True)
    assert bd.completion == 5.00
    assert bd.failure == 0.0
    assert bd.total == pytest.approx(5.95)


def test_no_reward_during_reset():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    assert "reward_total" not in info or info.get("reward_total") is None


def test_optimal_returns_per_workflow():
    expected_returns = {
        "LOGIN": 8.80,     # 3 * 0.95 + 5.95
        "SEARCH": 7.85,    # 2 * 0.95 + 5.95
        "PROFILE": 8.80,   # 3 * 0.95 + 5.95
        "CHECKOUT": 9.75,  # 4 * 0.95 + 5.95
    }

    for w_id, exp_ret in expected_returns.items():
        env = UIRecoveryEnv(workflow_id=w_id, max_candidates=20)
        obs, info = env.reset()

        total_return = 0.0
        while True:
            curr_step = env.workflow.steps[env.current_step_index]
            corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
            obs, reward, term, trunc, info = env.step(corr_idx)
            total_return += reward
            if term or trunc:
                break

        assert total_return == pytest.approx(exp_ret)
        env.close()


def test_one_wrong_action_lowers_return():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()

    # Step 0: Take wrong action first
    expected_role = env.workflow.steps[0].expected_role
    wrong_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role != expected_role][0]
    
    obs, r_wrong, term, trunc, info = env.step(wrong_idx)
    assert r_wrong == pytest.approx(-0.55)

    # Now complete workflow optimally
    total_return = r_wrong
    while True:
        curr_step = env.workflow.steps[env.current_step_index]
        corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
        obs, reward, term, trunc, info = env.step(corr_idx)
        total_return += reward
        if term or trunc:
            break

    # 8.80 - 0.55 = 8.25
    assert total_return == pytest.approx(8.25)


def test_one_invalid_action_lowers_return():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()

    # Restrict to 10 candidates to test padded action 15
    env.agent_candidates = env.agent_candidates[:10]
    env._private_meta = env._private_meta[:10]

    obs, r_invalid, term, trunc, info = env.step(15)
    assert r_invalid == pytest.approx(-1.05)

    # Now complete workflow optimally
    total_return = r_invalid
    while True:
        curr_step = env.workflow.steps[env.current_step_index]
        corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
        obs, reward, term, trunc, info = env.step(corr_idx)
        total_return += reward
        if term or trunc:
            break

    # 8.80 - 1.05 = 7.75
    assert total_return == pytest.approx(7.75)


def test_efficiency_property():
    """
    Optimal 4-decision completion return (8.80) > 6-decision completion return with mistakes.
    """
    env_opt = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env_opt.reset()
    ret_opt = 0.0
    while True:
        curr_step = env_opt.workflow.steps[env_opt.current_step_index]
        corr = [i for i, m in enumerate(env_opt._private_meta) if m.semantic_role == curr_step.expected_role][0]
        obs, r, term, trunc, info = env_opt.step(corr)
        ret_opt += r
        if term or trunc:
            break

    env_slow = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env_slow.reset()
    
    # 2 wrong actions before proceeding
    ret_slow = 0.0
    for _ in range(2):
        curr_step = env_slow.workflow.steps[env_slow.current_step_index]
        wrong_idx = [i for i, m in enumerate(env_slow._private_meta) if m.semantic_role != curr_step.expected_role][0]
        obs, r, _, _, _ = env_slow.step(wrong_idx)
        ret_slow += r

    while True:
        curr_step = env_slow.workflow.steps[env_slow.current_step_index]
        corr = [i for i, m in enumerate(env_slow._private_meta) if m.semantic_role == curr_step.expected_role][0]
        obs, r, term, trunc, info = env_slow.step(corr)
        ret_slow += r
        if term or trunc:
            break

    assert ret_opt > ret_slow
    assert ret_slow == pytest.approx(8.80 - 1.10)


def test_completion_vs_partial_failure():
    """
    Complete workflow return (+8.80) > partial progress followed by truncation failure.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20, max_episode_steps=5)
    obs, info = env.reset()

    # Complete Step 0 & 1
    corr0 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[0].expected_role][0]
    obs, r0, _, _, _ = env.step(corr0)  # +0.95
    
    corr1 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[1].expected_role][0]
    obs, r1, _, _, _ = env.step(corr1)  # +0.95

    # Now fail remaining decisions until truncated at step 5
    total_ret = r0 + r1
    while True:
        wrong = [i for i, m in enumerate(env._private_meta) if m.semantic_role != env.workflow.steps[2].expected_role][0]
        obs, r, term, trunc, info = env.step(wrong)
        total_ret += r
        if term or trunc:
            break

    # Step 2: -0.55, Step 3: -0.55, Step 4 (truncation): -2.55
    # Total = 0.95 + 0.95 - 0.55 - 0.55 - 2.55 = -1.75
    assert total_ret < 8.80


def test_wrong_clickable_element_gets_negative_reward():
    """
    Even if Playwright successfully clicks a wrong element (execution_success = True),
    the reward must be negative (-0.55) because step_success = False.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()

    wrong_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role != env.workflow.steps[0].expected_role][0]
    obs, reward, term, trunc, info = env.step(wrong_idx)

    assert info["execution_success"] is True
    assert info["step_success"] is False
    assert reward == pytest.approx(-0.55)


def test_info_exposes_breakdown():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()

    corr0 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[0].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr0)

    assert "reward_total" in info
    assert "reward_breakdown" in info
    assert "reward_version" in info
    assert info["reward_version"] == REWARD_VERSION
    assert info["reward_total"] == pytest.approx(0.95)
    assert info["reward_breakdown"]["progress"] == 1.00


def test_transition_consistency_on_page_state_failure():
    """
    Test transition consistency when SELECT_PRODUCT candidate validation passes
    BUT mandatory VERIFY_PRODUCT_PAGE url_equals assertion fails.

    Verifies:
    - reward = -0.55 (step cost -0.05 + wrong action -0.50)
    - progress reward = 0
    - completion bonus = 0
    - terminated = False
    - workflow restarts at step 0 (ENTER_SEARCH_QUERY)
    - browser URL, candidates, and observation objective are mutually consistent for step 0
    - agent can successfully recover and continue after restoration.
    """
    env = UIRecoveryEnv(workflow_id="SEARCH", max_candidates=20)
    obs, info = env.reset(seed=42)

    # Step 0: ENTER_SEARCH_QUERY
    corr0 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[0].expected_role][0]
    obs, r0, term, trunc, info = env.step(corr0)
    assert info["step_index"] == 1

    # Step 1: CLICK_SEARCH
    corr1 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[1].expected_role][0]
    obs, r1, term, trunc, info = env.step(corr1)
    assert info["step_index"] == 2

    # Override validate_page_state to simulate mandatory url_equals failure
    original_validate = env.browser_adapter.validate_page_state
    env.browser_adapter.validate_page_state = lambda success_condition, expected_value: False

    # Step 2: SELECT_PRODUCT (candidate matches expected_role, but URL validation will fail)
    corr2 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[2].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr2)

    # Restore original validate method
    env.browser_adapter.validate_page_state = original_validate

    # Verify reward contract on page-state assertion failure
    assert reward == pytest.approx(-0.55)
    assert info["reward_breakdown"]["progress"] == 0.0
    assert info["reward_breakdown"]["completion"] == 0.0
    assert info["reward_breakdown"]["wrong_action"] == -0.50
    assert term is False
    assert info["invalid_reason"] == "verification_failed"

    # CRITICAL INVARIANT VERIFICATION
    # Step index restarted to 0 (ENTER_SEARCH_QUERY)
    assert env.current_step_index == 0
    assert env.workflow.steps[env.current_step_index].step_id == "ENTER_SEARCH_QUERY"

    # Browser URL restored to /products
    assert env.browser_adapter.current_path == "/products"

    # info step_id corresponds to ENTER_SEARCH_QUERY
    assert info["step_id"] == "ENTER_SEARCH_QUERY"

    # Observation objective and candidate metadata match step 0 (search-input)
    assert env.observation_space.contains(obs) is True
    assert any(meta.semantic_role == "search-input" for meta in env._private_meta)

    # PROVE RECOVERABILITY: Execute correct search input action on restored step 0
    corr0_new = [i for i, m in enumerate(env._private_meta) if m.semantic_role == "search-input"][0]
    obs_rec, r_rec, term_rec, trunc_rec, info_rec = env.step(corr0_new)

    assert r_rec == pytest.approx(0.95)
    assert info_rec["step_index"] == 1
    assert info_rec["step_id"] == "CLICK_SEARCH"


def test_real_browser_page_state_failure_restoration():
    """
    Real Playwright browser integration test verifying page-state failure restoration
    and recoverability on real DOM.
    """
    from rl.env.playwright_browser_adapter import PlaywrightBrowserAdapter, PLAYWRIGHT_AVAILABLE
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")

    adapter = PlaywrightBrowserAdapter(base_url="http://localhost:3000", max_candidates=20)
    try:
        env = UIRecoveryEnv(workflow_id="SEARCH", max_candidates=20, browser_adapter=adapter)
        obs, info = env.reset(seed=42)

        # Step 0: ENTER_SEARCH_QUERY
        corr0 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == "search-input"][0]
        obs, r0, term, trunc, info = env.step(corr0)
        assert info["step_index"] == 1

        # Step 1: CLICK_SEARCH
        corr1 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == "search-action"][0]
        obs, r1, term, trunc, info = env.step(corr1)
        assert info["step_index"] == 2

        # Override page-state validation to fail
        orig_validate = adapter.validate_page_state
        adapter.validate_page_state = lambda success_condition, expected_value: False

        corr2 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == "product-result"][0]
        obs, reward, term, trunc, info = env.step(corr2)

        adapter.validate_page_state = orig_validate

        # Verify reward and step 0 restoration
        assert reward == pytest.approx(-0.55)
        assert env.current_step_index == 0
        assert info["step_id"] == "ENTER_SEARCH_QUERY"

        # Verify real browser page contains required step 0 candidate (search-input)
        assert any(m.semantic_role == "search-input" for m in env._private_meta)

        # Prove recoverability: re-execute search input action on restored real page
        corr0_rec = [i for i, m in enumerate(env._private_meta) if m.semantic_role == "search-input"][0]
        obs_rec, r_rec, term_rec, trunc_rec, info_rec = env.step(corr0_rec)

        assert r_rec == pytest.approx(0.95)
        assert info_rec["step_index"] == 1
        assert info_rec["step_id"] == "CLICK_SEARCH"
    finally:
        env.close()


def test_gymnasium_check_env():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    check_env(env, skip_render_check=True)


def test_held_out_protection():
    with pytest.raises(ValueError, match="EXPERIMENTAL INTEGRITY VIOLATION"):
        UIRecoveryEnv(workflow_id="LOGIN", mutation_level=6, mode="train")
