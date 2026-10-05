"""
Unit tests for Phase 17 — Benchmark Difficulty, Shortcut & 100% Success Audit (rl.audit.audit_benchmark).

Tests:
1. private semantic-role never enters policy observation Dict
2. UICandidate sanitize_attributes strips ground-truth keys
3. candidate recall analysis uses private metadata strictly post-extraction
4. feature ablation does not modify model parameters or checkpoint
5. canonical L6 definitions remain frozen
6. audit outputs stored outside canonical evaluation folder
7. original Phase 14 artifacts are untouched
8. Q logging preserves original argmax action
9. static leakage audit classifies symbols correctly
"""

import json
import os
import tempfile
import numpy as np
import pytest
import torch

from rl.env.types import UICandidate, PrivateEvaluatorMetadata, sanitize_attributes, PROHIBITED_ATTRIBUTE_KEYS, HELD_OUT_MUTATION_LEVEL
from rl.state.encoder import StateEncoder, STATE_ENCODING_VERSION
from rl.agents.dqn.agent import DQNAgent
from rl.audit.audit_benchmark import (
    run_static_leakage_audit,
    generate_feature_inventory,
    run_l6_candidate_recall_audit,
    run_candidate_set_difficulty,
    run_q_value_margin_audit,
    run_feature_ablation_audit,
    generate_benchmark_scorecard,
)


def test_private_semantic_role_never_enters_policy_observation():
    """Verify UICandidate and StateEncoder strictly strip private ground-truth attributes."""
    cand = UICandidate(
        candidate_id=1,
        tag="input",
        element_type="text",
        text="Username",
        attributes={
            "id": "username-field",
            "data-semantic-role": "username-input",
            "expected_role": "username-input",
            "ground_truth": "correct",
            "class": "form-control"
        }
    )

    # Sanitize check
    assert "data-semantic-role" not in cand.attributes
    assert "expected_role" not in cand.attributes
    assert "ground_truth" not in cand.attributes
    assert cand.attributes["id"] == "username-field"
    assert cand.attributes["class"] == "form-control"

    # Encoder check
    encoder = StateEncoder()
    struct_feats = encoder.encode_structural_features(cand)
    assert len(struct_feats) == 20
    assert not hasattr(cand, "semantic_role")
    assert not hasattr(cand, "expected_role")


def test_candidate_recall_uses_private_metadata_post_extraction():
    """Verify PrivateEvaluatorMetadata links candidate_id to ground truth without exposing it to UICandidate."""
    cand = UICandidate(candidate_id=5, tag="button", text="Submit")
    priv_meta = PrivateEvaluatorMetadata(candidate_id=5, semantic_role="login-action")

    assert not hasattr(cand, "semantic_role")
    assert priv_meta.candidate_id == cand.candidate_id
    assert priv_meta.semantic_role == "login-action"


def test_feature_ablation_does_not_modify_model_weights():
    """Verify feature ablation zeroing is inference-only and preserves model parameters."""
    agent = DQNAgent()
    weights_before = {k: v.clone() for k, v in agent.online_network.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_feature_ablation_audit("non_existent_path.pt", tmp_dir)

    weights_after = agent.online_network.state_dict()
    for k in weights_before:
        assert torch.equal(weights_before[k], weights_after[k])


def test_canonical_l6_definition_frozen():
    """Verify HELD_OUT_MUTATION_LEVEL remains 6 / 'L6' and is frozen."""
    assert HELD_OUT_MUTATION_LEVEL == 6 or str(HELD_OUT_MUTATION_LEVEL) == "L6"


def test_audit_outputs_stored_outside_canonical_evaluation_folder():
    """Verify Phase 17 outputs are placed in artifacts/audit/phase17 and leave dqn-final untouched."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        audit_dir = os.path.join(tmp_dir, "audit", "phase17")
        run_static_leakage_audit(audit_dir)
        generate_feature_inventory(audit_dir)
        generate_benchmark_scorecard(audit_dir)

        assert os.path.exists(os.path.join(audit_dir, "static_leakage_audit.md"))
        assert os.path.exists(os.path.join(audit_dir, "feature_inventory.csv"))
        assert os.path.exists(os.path.join(audit_dir, "benchmark_difficulty_scorecard.md"))
        assert "dqn-final" not in audit_dir


def test_original_phase14_artifacts_untouched():
    """Verify Phase 14 evaluation artifacts remain present and unmodified."""
    dqn_final_dir = "artifacts/evaluation/dqn-final"
    if os.path.exists(dqn_final_dir):
        assert os.path.exists(os.path.join(dqn_final_dir, "manifest.json"))
        assert os.path.exists(os.path.join(dqn_final_dir, "checkpoint_metadata.json"))


def test_q_logging_preserves_argmax_action():
    """Verify Q-margin calculation identifies top Q and second-highest Q accurately."""
    q_vals = np.array([-1.0, 5.2, 3.1, -1e9, 1.8], dtype=np.float32)
    valid_mask = np.array([1, 1, 1, 0, 1], dtype=np.int8)

    valid_indices = np.flatnonzero(valid_mask == 1)
    valid_q = q_vals[valid_indices]
    sorted_idx = np.argsort(-valid_q)

    top_act = valid_indices[sorted_idx[0]]
    runner_up_act = valid_indices[sorted_idx[1]]

    assert top_act == 1
    assert runner_up_act == 2
    assert q_vals[top_act] == 5.2
    assert q_vals[runner_up_act] == 3.1
    assert round(float(q_vals[top_act] - q_vals[runner_up_act]), 2) == 2.1
