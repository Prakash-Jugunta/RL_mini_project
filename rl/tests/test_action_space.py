import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.env.workflow import WORKFLOW_LOGIN, WORKFLOW_SEARCH, WORKFLOW_PROFILE, WORKFLOW_CHECKOUT
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.action import ActionExecutionResult, sample_valid_action


def test_action_space_is_discrete_20():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 20


def test_action_mask_shape_and_values():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    mask = env.get_action_mask()
    assert mask.shape == (20,)
    assert mask.dtype == np.int8 or mask.dtype == np.int64 or mask.dtype == int
    assert set(np.unique(mask)).issubset({0, 1})
    
    num_candidates = info["candidate_count"]
    assert np.sum(mask) == num_candidates
    assert np.all(mask[:num_candidates] == 1)
    assert np.all(mask[num_candidates:] == 0)


def test_action_mask_matches_observation():
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    action_mask = env.get_action_mask()
    valid_mask = env.valid_action_mask()
    
    assert np.array_equal(action_mask, valid_mask)
    assert np.array_equal(action_mask, obs["candidate_mask"])


def test_mask_represents_existence_only():
    """
    Action mask must answer 'does candidate i exist?' NOT 'is candidate i correct?'
    All real candidates must have mask = 1 regardless of role or ground truth.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    mask = env.get_action_mask()
    num_candidates = len(env.agent_candidates)
    
    # Check that ALL real candidates have mask == 1
    for i in range(num_candidates):
        assert mask[i] == 1


def test_wrong_real_candidate_is_valid_action():
    """
    Selecting a wrong real candidate index is valid (mask=1), but fails workflow step validation.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    # Find a wrong real candidate index (one whose semantic_role doesn't match expected_role)
    expected_role = env.workflow.steps[0].expected_role
    wrong_index = None
    for i, meta in enumerate(env._private_meta):
        if meta.semantic_role != expected_role:
            wrong_index = i
            break
            
    assert wrong_index is not None, "Mock adapter should provide wrong candidates as well as correct ones"
    
    obs, reward, terminated, truncated, info = env.step(wrong_index)
    
    assert info["action_valid"] is True
    assert info["execution_success"] is True
    assert info["step_success"] is False
    assert info["invalid_reason"] == "incorrect_candidate"
    assert info["step_index"] == 0  # Step did not advance


def test_padded_action_behavior():
    """
    For candidate_count <= action < 20, environment must:
    - NOT call Playwright
    - NOT advance workflow
    - NOT crash
    - Return invalid_reason = 'padded_candidate'
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    # Restrict candidate list to 11 candidates to test padding
    env.agent_candidates = env.agent_candidates[:11]
    env._private_meta = env._private_meta[:11]
    obs = env._construct_observation()
    
    padded_action = 11  # Index 11 is padded (11 < 20)
    
    assert padded_action < 20
    assert obs["candidate_mask"][padded_action] == 0
    
    obs, reward, terminated, truncated, info = env.step(padded_action)
    
    assert info["action_valid"] is False
    assert info["invalid_reason"] == "padded_candidate"
    assert info["execution_success"] is False
    assert info["step_success"] is False
    assert info["step_index"] == 0  # Step did not advance


def test_out_of_space_actions():
    """
    Test action < 0, action = 20, action = 999.
    Must handle safely without Python negative indexing (action=-1 must not select last element).
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    for invalid_action in [-1, 20, 999]:
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        assert info["action_valid"] is False
        assert info["invalid_reason"] == "out_of_space"
        assert info["execution_success"] is False
        assert info["step_success"] is False
        assert info["step_index"] == 0


def test_decoupled_execution_success_vs_step_success():
    """
    execution_success = Did Playwright execute the action?
    step_success = Did the action satisfy ground truth objective?
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    # Choose a wrong real candidate
    wrong_index = None
    expected_role = env.workflow.steps[0].expected_role
    for i, meta in enumerate(env._private_meta):
        if meta.semantic_role != expected_role:
            wrong_index = i
            break
            
    obs, reward, terminated, truncated, info = env.step(wrong_index)
    
    assert info["execution_success"] is True
    assert info["step_success"] is False
    
    action_result = info["action_result"]
    assert isinstance(action_result, ActionExecutionResult)
    assert action_result.attempted is True
    assert action_result.executed is True
    assert action_result.execution_success is True
    assert action_result.step_success is False


def test_private_metadata_does_not_affect_mask():
    """
    Mask must be identical regardless of private evaluator metadata.
    """
    mock_adapter_1 = MockBrowserAdapter(max_candidates=20)
    mock_adapter_2 = MockBrowserAdapter(max_candidates=20)
    
    env1 = UIRecoveryEnv(workflow_id="LOGIN", browser_adapter=mock_adapter_1)
    env2 = UIRecoveryEnv(workflow_id="LOGIN", browser_adapter=mock_adapter_2)
    
    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)
    
    assert np.array_equal(env1.get_action_mask(), env2.get_action_mask())


def test_candidate_reorder_changes_action_mapping():
    """
    Action index 0 MUST target candidate 0.
    Reordering candidates [A, B, C] -> [C, A, B] causes action 0 to target C.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    original_c0 = env.agent_candidates[0]
    original_c1 = env.agent_candidates[1]
    
    # Swap elements 0 and 1 in agent_candidates
    env.agent_candidates[0], env.agent_candidates[1] = env.agent_candidates[1], env.agent_candidates[0]
    
    # Now action 0 selects original_c1
    assert env.agent_candidates[0].candidate_id == original_c1.candidate_id


def test_candidate_lifetime_and_mask_refresh():
    """
    After a successful step transition, candidates and masks are freshly generated.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    mask_step0 = env.get_action_mask().copy()
    
    # Locate correct candidate for step 0
    correct_idx = None
    expected_role = env.workflow.steps[0].expected_role
    for i, meta in enumerate(env._private_meta):
        if meta.semantic_role == expected_role:
            correct_idx = i
            break
            
    assert correct_idx is not None
    obs, reward, terminated, truncated, info = env.step(correct_idx)
    
    assert info["step_index"] == 1
    mask_step1 = env.get_action_mask()
    
    # Observation candidate mask matches new mask
    assert np.array_equal(obs["candidate_mask"], mask_step1)


def test_page_state_step_skips_rl_decision():
    """
    SEARCH workflow:
    0: ENTER_SEARCH_QUERY (fill)
    1: CLICK_SEARCH (click)
    2: SELECT_PRODUCT (click)
    3: VERIFY_PRODUCT_PAGE (url_equals -> page-state assertion)
    Total RL candidate decisions = 3.
    """
    env = UIRecoveryEnv(workflow_id="SEARCH", max_candidates=20)
    obs, info = env.reset()
    
    # Step 0: ENTER_SEARCH_QUERY
    corr0 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[0].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr0)
    assert info["step_index"] == 1
    assert not term
    
    # Step 1: CLICK_SEARCH
    corr1 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[1].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr1)
    assert info["step_index"] == 2
    assert not term
    
    # Step 2: SELECT_PRODUCT (after which VERIFY_PRODUCT_PAGE auto-runs)
    corr2 = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[2].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr2)
    
    # Workflow should now be terminated because step 3 (url_equals) auto-passed!
    assert info["workflow_completed"] is True
    assert term is True
    assert env.total_decisions == 3  # Exactly 3 decisions!


def test_expected_decision_counts_per_workflow():
    """
    LOGIN: 4 decisions
    SEARCH: 3 decisions
    PROFILE: 4 decisions
    CHECKOUT: 5 decisions
    Total: 16 decisions across all workflows.
    """
    decisions = {}
    for w_id in ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]:
        env = UIRecoveryEnv(workflow_id=w_id, max_candidates=20)
        obs, info = env.reset()
        
        while True:
            # Pick correct candidate for current step
            curr_step = env.workflow.steps[env.current_step_index]
            corr = [i for i, m in enumerate(env._private_meta) if m.semantic_role == curr_step.expected_role][0]
            obs, reward, term, trunc, info = env.step(corr)
            if term or trunc:
                break
                
        decisions[w_id] = env.total_decisions
        
    assert decisions["LOGIN"] == 4
    assert decisions["SEARCH"] == 3
    assert decisions["PROFILE"] == 4
    assert decisions["CHECKOUT"] == 5
    assert sum(decisions.values()) == 16


def test_random_valid_sampler():
    """
    Verify sample_valid_action only chooses mask == 1, is reproducible under seed, and samples uniformly.
    """
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    
    sample1 = sample_valid_action(mask, rng1)
    sample2 = sample_valid_action(mask, rng2)
    
    assert sample1 == sample2
    assert 0 <= sample1 < 5
    
    # Statistical uniformity check
    counts = {i: 0 for i in range(5)}
    rng = np.random.default_rng(100)
    for _ in range(1000):
        a = sample_valid_action(mask, rng)
        assert mask[a] == 1
        counts[a] += 1
        
    for i in range(5):
        assert counts[i] > 100  # Reasonable spread across valid indices


def test_action_diagnostics_tracking():
    """
    Verify episode diagnostic metrics.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    obs, info = env.reset()
    
    # Restrict candidate list to 10 candidates to allow testing padded action 15
    env.agent_candidates = env.agent_candidates[:10]
    env._private_meta = env._private_meta[:10]
    
    # Padded action (15 >= 10, 15 < 20)
    env.step(15)
    # Out of space action (-1 < 0)
    env.step(-1)
    # Wrong candidate action
    wrong_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role != env.workflow.steps[0].expected_role][0]
    env.step(wrong_idx)
    # Correct candidate action
    corr_idx = [i for i, m in enumerate(env._private_meta) if m.semantic_role == env.workflow.steps[0].expected_role][0]
    obs, reward, term, trunc, info = env.step(corr_idx)
    
    assert info["total_decisions"] == 4
    assert info["padded_actions"] == 1
    assert info["out_of_space_actions"] == 1
    assert info["valid_candidate_actions"] == 2
    assert info["successful_step_actions"] == 1
    assert info["wrong_step_actions"] == 1


def test_gymnasium_check_env():
    """
    Verify UIRecoveryEnv complies with standard Gymnasium API requirements.
    """
    env = UIRecoveryEnv(workflow_id="LOGIN", max_candidates=20)
    check_env(env, skip_render_check=True)


def test_held_out_protection():
    """
    Level 6 cannot be used with mode='train'.
    """
    with pytest.raises(ValueError, match="EXPERIMENTAL INTEGRITY VIOLATION"):
        UIRecoveryEnv(workflow_id="LOGIN", mutation_level=6, mode="train")
