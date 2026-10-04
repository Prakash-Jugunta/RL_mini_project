import pytest
import numpy as np
import torch

from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.env.workflow import WORKFLOW_REGISTRY
from rl.state.encoder import StateEncoder
from rl.state.feature_schema import StateFeatureSchema
from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.network import apply_action_mask


def test_previous_candidate_permutation_invariance():
    """
    Part 8 Test: Verify that permuting candidate order between steps does NOT alter
    the stable public structural representation of the previously selected candidate.
    """
    encoder = StateEncoder(max_candidates=20)
    cand_prev = UICandidate(candidate_id=4, tag="input", element_type="text", text="", attributes={"id": "username"})

    # Context when candidate was selected at slot index 4
    ctx_slot_4 = encoder.encode_context_features(
        workflow_id="LOGIN",
        action_type="fill",
        current_step_index=1,
        total_steps=4,
        previous_action=4,
        previous_success=1.0,
        has_previous_action=True,
        previous_candidate=cand_prev,
    )

    # Context when same candidate was selected at slot index 10
    ctx_slot_10 = encoder.encode_context_features(
        workflow_id="LOGIN",
        action_type="fill",
        current_step_index=1,
        total_steps=4,
        previous_action=10,
        previous_success=1.0,
        has_previous_action=True,
        previous_candidate=cand_prev,
    )

    # Permutation invariance: context feature vectors MUST be identical (L2 distance = 0.0)
    assert np.allclose(ctx_slot_4, ctx_slot_10, atol=1e-6)
    assert np.linalg.norm(ctx_slot_4 - ctx_slot_10) == pytest.approx(0.0, abs=1e-6)


def test_raw_previous_slot_is_ignored_without_public_candidate():
    """Raw candidate indices must not change policy context."""
    encoder = StateEncoder(max_candidates=20)
    ctx_a = encoder.encode_context_features(
        "LOGIN", "fill", 1, 4, previous_action=4, has_previous_action=True
    )
    ctx_b = encoder.encode_context_features(
        "LOGIN", "fill", 1, 4, previous_action=10, has_previous_action=True
    )
    assert np.array_equal(ctx_a, ctx_b)


def test_current_step_distinguishability():
    """
    Part 8 Test: Verify that observations with identical candidate sets are strongly distinguished
    between ENTER_USERNAME vs ENTER_PASSWORD and ADD_TO_CART vs CLICK_CART.
    """
    encoder = StateEncoder(max_candidates=20)
    cands = [
        UICandidate(0, "input", "text", "", attributes={"id": "username"}),
        UICandidate(1, "input", "password", "", attributes={"id": "password"}),
        UICandidate(2, "button", "submit", "Login", attributes={"id": "login-btn"}),
    ]

    login_w = WORKFLOW_REGISTRY["LOGIN"]
    obs_user = encoder.encode("LOGIN", login_w.steps[0], 0, 4, cands, has_previous_action=False)
    obs_pass = encoder.encode("LOGIN", login_w.steps[1], 1, 4, cands, has_previous_action=True, previous_candidate=cands[0])

    # Context vectors must differ (step progress, previous candidate features)
    assert not np.allclose(obs_user["context"], obs_pass["context"])

    checkout_w = WORKFLOW_REGISTRY["CHECKOUT"]
    checkout_cands = [
        UICandidate(0, "button", "button", "Add to Cart", attributes={"id": "add-to-cart-btn"}),
        UICandidate(1, "button", "button", "Cart", attributes={"id": "cart-btn"}),
    ]
    obs_add_cart = encoder.encode("CHECKOUT", checkout_w.steps[0], 0, 5, checkout_cands, has_previous_action=False)
    obs_click_cart = encoder.encode("CHECKOUT", checkout_w.steps[1], 1, 5, checkout_cands, has_previous_action=True, previous_candidate=checkout_cands[0])

    assert not np.allclose(obs_add_cart["context"], obs_click_cart["context"])


def test_no_evaluator_metadata_leakage():
    """
    Part 8 Test: Audit state observations to verify zero private metadata leakage.
    """
    encoder = StateEncoder(max_candidates=20)
    cand_public = UICandidate(0, "input", "text", "", attributes={"id": "username"})

    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    obs = encoder.encode("LOGIN", step, 0, 4, [cand_public])

    # Convert observation arrays to raw text representations
    obs_str = str(obs)
    assert "data-semantic-role" not in obs_str
    assert "expected_role" not in obs_str
    assert "username-input" not in obs_str


def test_dqn_forward_pass_dimensions():
    """
    Part 8 Test: Verify DQN agent forward pass dimensions with current StateFeatureSchema.
    """
    schema = StateFeatureSchema()
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)

    dummy_obs = {
        "objective": torch.zeros((1, schema.objective_dim), dtype=torch.float32),
        "context": torch.zeros((1, schema.context_dim), dtype=torch.float32),
        "candidates": torch.zeros((1, schema.max_candidates, schema.candidate_feature_dim), dtype=torch.float32),
    }

    with torch.no_grad():
        q_vals = agent.online_network(dummy_obs)

    assert q_vals.shape == (1, schema.max_candidates)


def test_environment_decision_counts_and_oracle_returns():
    """
    Part 8 Test: Assert exact frozen RL decision counts and optimal oracle returns per workflow.
    """
    expected_decisions = {
        "LOGIN": 4,
        "SEARCH": 3,
        "PROFILE": 4,
        "CHECKOUT": 5,
    }

    expected_returns = {
        "LOGIN": 8.80,
        "SEARCH": 7.85,
        "PROFILE": 8.80,
        "CHECKOUT": 9.75,
    }

    for w_id, target_decisions in expected_decisions.items():
        env = UIRecoveryEnv(workflow_id=w_id, mutation_level=0, mode="train")
        obs, info = env.reset(seed=42)

        decisions = 0
        total_return = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated) and decisions < 20:
            current_step = env.workflow.steps[env.current_step_index]
            correct_idx = None
            for idx, cand in enumerate(env.agent_candidates):
                meta = env.browser_adapter.private_metadata_map.get(cand.candidate_id)
                if meta and meta.semantic_role == current_step.expected_role:
                    correct_idx = idx
                    break

            assert correct_idx is not None
            obs, reward, terminated, truncated, info = env.step(correct_idx)
            total_return += reward
            decisions += 1

        assert decisions == target_decisions, f"{w_id} decision count mismatch"
        assert total_return == pytest.approx(expected_returns[w_id]), f"{w_id} return mismatch"
        env.close()


def test_candidate_permutation_equivariance():
    """
    Part 8 Test: Verify candidate permutation equivariance of DQN forward pass.
    Q(s_permuted)[perm[i]] must equal Q(s_original)[i].
    """
    encoder = StateEncoder(max_candidates=20)
    config = DQNConfig(seed=42)
    agent = DQNAgent(config)

    cand_a = UICandidate(0, "input", "text", "User", attributes={"id": "user"})
    cand_b = UICandidate(1, "input", "password", "Pass", attributes={"id": "pass"})
    cands_orig = [cand_a, cand_b]
    cands_perm = [cand_b, cand_a]

    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    obs_orig = encoder.encode("LOGIN", step, 0, 4, cands_orig)
    obs_perm = encoder.encode("LOGIN", step, 0, 4, cands_perm)

    def to_tensors(obs):
        return {
            "objective": torch.tensor(obs["objective"], dtype=torch.float32).unsqueeze(0),
            "context": torch.tensor(obs["context"], dtype=torch.float32).unsqueeze(0),
            "candidates": torch.tensor(obs["candidates"], dtype=torch.float32).unsqueeze(0),
        }

    with torch.no_grad():
        q_orig = agent.online_network(to_tensors(obs_orig)).squeeze(0).numpy()
        q_perm = agent.online_network(to_tensors(obs_perm)).squeeze(0).numpy()

    # cand_a was slot 0 in orig, slot 1 in perm
    assert q_orig[0] == pytest.approx(q_perm[1], abs=1e-5)
    # cand_b was slot 1 in orig, slot 0 in perm
    assert q_orig[1] == pytest.approx(q_perm[0], abs=1e-5)

