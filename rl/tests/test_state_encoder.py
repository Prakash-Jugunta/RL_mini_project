import pytest
import numpy as np
import gymnasium as gym
from dataclasses import replace

from rl.env.types import UICandidate, PrivateEvaluatorMetadata, HELD_OUT_MUTATION_LEVEL
from rl.env.workflow import WORKFLOW_REGISTRY
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.state.encoder import StateEncoder
from rl.state.feature_schema import StateFeatureSchema


def test_state_output_shape_and_containment():
    """1. Test observation dictionary shapes and Gymnasium space containment."""
    encoder = StateEncoder(max_candidates=20)
    space = encoder.get_observation_space()

    workflow = WORKFLOW_REGISTRY["LOGIN"]
    step = workflow.steps[0]

    cand = UICandidate(candidate_id=1, tag="input", element_type="text", placeholder="Username")
    obs = encoder.encode(
        workflow_id="LOGIN",
        step=step,
        current_step_index=0,
        total_steps=4,
        candidates=[cand],
        previous_action=-1,
        previous_success=0.0,
        has_previous_action=False
    )

    assert obs["objective"].shape == (64,)
    assert obs["candidates"].shape == (20, 91)
    assert obs["candidate_mask"].shape == (20,)
    assert obs["context"].shape == (30,)

    assert obs["objective"].dtype == np.float32
    assert obs["candidates"].dtype == np.float32
    assert obs["candidate_mask"].dtype == np.int8
    assert obs["context"].dtype == np.float32

    assert space.contains(obs) is True


def test_candidate_padding_and_mask():
    """2. Test that N candidates populate rows 0..N-1 and rows N..19 are exact zero vectors with mask 0."""
    encoder = StateEncoder(max_candidates=20)
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]

    cands = [
        UICandidate(candidate_id=1, tag="input", text="User"),
        UICandidate(candidate_id=2, tag="button", text="Submit"),
    ]

    obs = encoder.encode("LOGIN", step, 0, 4, cands)

    mask = obs["candidate_mask"]
    matrix = obs["candidates"]

    # First 2 mask entries must be 1
    assert mask[0] == 1
    assert mask[1] == 1
    # Remaining 18 mask entries must be 0
    assert np.all(mask[2:] == 0)

    # First 2 rows must be non-zero
    assert np.any(matrix[0] != 0)
    assert np.any(matrix[1] != 0)
    # Remaining 18 rows must be exact zeros
    assert np.all(matrix[2:] == 0.0)


def test_candidate_order_preservation():
    """3. Test candidate ordering is preserved and NOT re-sorted by semantic similarity."""
    encoder = StateEncoder(max_candidates=20)
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]

    c1 = UICandidate(candidate_id=1, tag="input", placeholder="Username", x=10.0)
    c2 = UICandidate(candidate_id=2, tag="button", text="Login", x=100.0)

    obs1 = encoder.encode("LOGIN", step, 0, 4, [c1, c2])
    obs2 = encoder.encode("LOGIN", step, 0, 4, [c2, c1])

    # Row 0 of obs1 must match Row 1 of obs2
    assert np.array_equal(obs1["candidates"][0], obs2["candidates"][1])
    assert np.array_equal(obs1["candidates"][1], obs2["candidates"][0])


def test_determinism():
    """4. Test identical inputs produce numerically identical observation vectors."""
    encoder = StateEncoder(max_candidates=20)
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    cand = UICandidate(candidate_id=1, tag="input", placeholder="Username")

    obs1 = encoder.encode("LOGIN", step, 0, 4, [cand], previous_action=2, previous_success=1.0, has_previous_action=True)
    obs2 = encoder.encode("LOGIN", step, 0, 4, [cand], previous_action=2, previous_success=1.0, has_previous_action=True)

    assert np.array_equal(obs1["objective"], obs2["objective"])
    assert np.array_equal(obs1["candidates"], obs2["candidates"])
    assert np.array_equal(obs1["candidate_mask"], obs2["candidate_mask"])
    assert np.array_equal(obs1["context"], obs2["context"])


def test_different_objective_encoding():
    """5. Test ENTER_USERNAME vs ENTER_PASSWORD produces different objective embeddings."""
    encoder = StateEncoder(max_candidates=20)
    step_user = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    step_pass = WORKFLOW_REGISTRY["LOGIN"].steps[1]

    obs_user = encoder.encode("LOGIN", step_user, 0, 4, [])
    obs_pass = encoder.encode("LOGIN", step_pass, 1, 4, [])

    assert not np.array_equal(obs_user["objective"], obs_pass["objective"])


def test_different_public_dom_encoding():
    """6. Test different observable DOM candidate attributes change the candidate representation."""
    encoder = StateEncoder(max_candidates=20)
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]

    cand_a = UICandidate(candidate_id=1, tag="input", placeholder="Username")
    cand_b = UICandidate(candidate_id=1, tag="input", placeholder="Account Number")

    obs_a = encoder.encode("LOGIN", step, 0, 4, [cand_a])
    obs_b = encoder.encode("LOGIN", step, 0, 4, [cand_b])

    assert not np.array_equal(obs_a["candidates"][0], obs_b["candidates"][0])


def test_private_metadata_counterfactual_invariance():
    """7. Counterfactual test: varying private ground truth (expected_role) MUST NOT alter the observation."""
    encoder = StateEncoder(max_candidates=20)

    step_a = WORKFLOW_REGISTRY["LOGIN"].steps[0]  # expected_role="username-input"
    step_b = WORKFLOW_REGISTRY["LOGIN"].steps[1]  # expected_role="password-input"
    # Override step_b's intent to match step_a's intent so only ground truth differs
    step_b_same_intent = replace(
        WORKFLOW_REGISTRY["LOGIN"].steps[1],
        agent_intent=step_a.agent_intent,
    )

    cand = UICandidate(candidate_id=1, tag="input", placeholder="Enter username")

    obs_a = encoder.encode("LOGIN", step_a, 0, 4, [cand])
    obs_b = encoder.encode("LOGIN", step_b_same_intent, 0, 4, [cand])

    assert np.array_equal(obs_a["objective"], obs_b["objective"])
    assert np.array_equal(obs_a["candidates"], obs_b["candidates"])
    assert np.array_equal(obs_a["context"], obs_b["context"])


def test_mutation_metadata_counterfactual_invariance():
    """8. Counterfactual test: mutation_level/seed metadata MUST NOT enter policy state."""
    env1 = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mutation_seed=11)
    env2 = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=3, mutation_seed=99)

    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)

    # Context vectors for same workflow index 0 must be identical
    assert np.array_equal(obs1["context"], obs2["context"])

    env1.close()
    env2.close()


def test_leakage_audit():
    """9. Audit that prohibited strings never appear in state dictionary or schema."""
    encoder = StateEncoder(max_candidates=20)
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    cand = UICandidate(candidate_id=1, tag="input", placeholder="Username")
    obs = encoder.encode("LOGIN", step, 0, 4, [cand])

    forbidden = [
        "data-semantic-role", "semantic_role", "expected_role",
        "target_role", "ground_truth", "is_correct", "target_index",
        "candidate_recall", "mutation_level", "mutation_seed", "PrivateEvaluatorMetadata"
    ]

    for key in obs.keys():
        for f in forbidden:
            assert f not in key.lower(), f"Forbidden string '{f}' found in observation key '{key}'"


def test_previous_candidate_context_representation():
    """10. Test workflow/operation context and slot-index independence."""
    encoder = StateEncoder(max_candidates=20)

    # LOGIN (idx 0), fill (idx 0)
    step_fill = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    obs1 = encoder.encode("LOGIN", step_fill, 0, 4, [], previous_action=-1, has_previous_action=False)

    ctx1 = obs1["context"]
    assert ctx1[0] == 1.0  # LOGIN one-hot
    assert ctx1[4] == 1.0  # fill one-hot
    assert ctx1[8] == 0.0  # has_previous_action=False
    assert np.all(ctx1[9:29] == 0.0)  # no previous public candidate

    # SEARCH (idx 1), click (idx 1), raw previous slot is ignored
    step_click = WORKFLOW_REGISTRY["SEARCH"].steps[1]
    obs2 = encoder.encode("SEARCH", step_click, 1, 4, [], previous_action=3, previous_success=1.0, has_previous_action=True)

    ctx2 = obs2["context"]
    assert ctx2[1] == 1.0  # SEARCH one-hot
    assert ctx2[5] == 1.0  # click one-hot
    assert ctx2[8] == 1.0  # has_previous_action=True
    assert np.all(ctx2[9:29] == 0.0)  # no public candidate supplied
    assert ctx2[29] == 1.0  # prev_success=1.0


def test_normalized_bounding_boxes():
    """11. Test visual bounding box features are normalized in [0, 1]."""
    encoder = StateEncoder(max_candidates=20, viewport_width=1280.0, viewport_height=800.0)
    cand = UICandidate(candidate_id=1, tag="button", x=640.0, y=400.0, width=128.0, height=40.0)

    vis = encoder.encode_visual_features(cand)
    assert vis[0] == pytest.approx(0.5)      # x_norm
    assert vis[1] == pytest.approx(0.5)      # y_norm
    assert vis[2] == pytest.approx(0.1)      # width_norm
    assert vis[3] == pytest.approx(0.05)     # height_norm
    assert np.all((vis >= 0.0) & (vis <= 1.0))


def test_page_state_verification_no_decision_state():
    """12. Test VERIFY_PRODUCT_PAGE auto-skips without requesting RL candidate decisions."""
    env = UIRecoveryEnv(workflow_id="SEARCH", mutation_level=0)
    env.reset(seed=42)

    def _get_correct_idx(e):
        cs = e.workflow.steps[e.current_step_index]
        for i, c in enumerate(e.agent_candidates):
            meta = e.browser_adapter.private_metadata_map.get(c.candidate_id)
            if meta and meta.semantic_role == cs.expected_role:
                return i
        return 0

    env.step(_get_correct_idx(env))  # ENTER_SEARCH_QUERY
    env.step(_get_correct_idx(env))  # CLICK_SEARCH
    obs, reward, terminated, truncated, info = env.step(_get_correct_idx(env))  # SELECT_PRODUCT

    assert info["step_index"] == 4
    assert terminated is True
    assert info["workflow_completed"] is True
    env.close()


def test_gymnasium_check_env_phase6():
    """13. Gymnasium Environment Checker Compliance Test for Phase 6."""
    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=0, mode="train")
    gym.utils.env_checker.check_env(env)
    env.close()


def test_l6_held_out_training_rejection():
    """14. Test Level 6 is allowed for encoder inspection but rejected for RL training."""
    encoder = StateEncoder()
    cand = UICandidate(candidate_id=1, tag="input", text="Test L6")
    step = WORKFLOW_REGISTRY["LOGIN"].steps[0]
    obs = encoder.encode("LOGIN", step, 0, 4, [cand])
    assert obs["objective"].shape == (64,)

    env = UIRecoveryEnv(workflow_id="LOGIN", mutation_level=6, mode="test")
    obs_test, _ = env.reset()
    assert env.observation_space.contains(obs_test)
    env.close()

    with pytest.raises(ValueError, match="HELD OUT"):
        UIRecoveryEnv(workflow_id="LOGIN", mutation_level=6, mode="train")
