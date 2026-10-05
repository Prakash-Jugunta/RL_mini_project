"""
Unit tests for Phase 16 — Final DQN Results Analysis (rl.analysis.load_dqn_results & rl.analysis.analyze_dqn).

Tests:
1. normalized loader concatenates splits correctly
2. success-rate computation & Wilson CI boundaries
3. split separation & strict input validation
4. L6 leakage detection in validation/test-id splits
5. workflow aggregation correctness
6. mutation level and type aggregation
7. failure taxonomy fallback and category assignment
8. missing optional metric handling (e.g. latency NA fallback)
9. summary artifacts serialization (final_metrics.json, presentation_summary.md, viva_defence.md)
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from rl.analysis.load_dqn_results import (
    compute_wilson_ci,
    validate_input_artifacts,
    load_normalized_dataset,
    save_normalized_dataset,
)
from rl.analysis.analyze_dqn import (
    compute_split_overall_metrics,
    compute_generalization_gaps,
    generate_workflow_analysis,
    generate_mutation_analysis,
    generate_failure_taxonomy,
    generate_presentation_summary,
    generate_viva_defence,
    generate_phase16_report,
    parse_numeric,
)


def create_mock_evaluation_dir(tmp_dir: str, num_val: int = 120, num_id: int = 120, num_l6: int = 100) -> str:
    """Helper to create a fully valid mock Phase 14 evaluation directory structure."""
    with open(os.path.join(tmp_dir, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint_path": "artifacts/dqn/phase13-v1/latest_checkpoint.pt",
            "sha256": "1ad8de13df71acb5f9ca1f3bfea25b050c1f0bbc901f56235af1ba60e3c01753",
            "training_episode": 15000,
            "global_step": 68459,
        }, f)

    with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_version": "phase14-dqn-eval-v1",
            "protocol": {
                "epsilon": 0.0,
                "reward_version": "phase8-v1",
                "state_encoding_version": "phase6-public-prev-candidate-v2"
            }
        }, f)

    split_configs = [
        ("validation", num_val, ["0", "1", "2", "3", "4", "5"]),
        ("test-id", num_id, ["0", "1", "2", "3", "4", "5"]),
        ("test-l6", num_l6, ["6"]),
    ]

    ep_id_counter = 0
    for split_name, count, levels in split_configs:
        s_dir = os.path.join(tmp_dir, split_name)
        os.makedirs(s_dir, exist_ok=True)
        rows = []
        for i in range(count):
            ep_id_counter += 1
            lvl = levels[i % len(levels)]
            rows.append({
                "split": split_name,
                "episode_id": ep_id_counter,
                "workflow": ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"][i % 4],
                "mutation_level": lvl,
                "mutation_seed": 11,
                "environment_seed": 1000 + i,
                "success": True,
                "return": 8.8,
                "decisions": 4,
                "optimal_decisions": 4,
                "decision_overhead": 0,
                "successful_step_actions": 4,
                "wrong_step_actions": 0,
                "invalid_actions": 0,
                "execution_failures": 0,
                "terminated": True,
                "truncated": False,
                "failure_reason": "",
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(s_dir, "episodes.csv"), index=False)

    return tmp_dir


def test_wilson_ci_computation():
    """Verify Wilson binomial confidence interval boundaries."""
    # 100% success (120/120)
    lower, upper = compute_wilson_ci(120, 120)
    assert lower > 95.0
    assert upper == 100.0

    # 50% success (50/100)
    lower_50, upper_50 = compute_wilson_ci(50, 100)
    assert 40.0 < lower_50 < 50.0
    assert 50.0 < upper_50 < 60.0


def test_input_validation_pass_and_fail():
    """Verify validate_input_artifacts correctly flags valid vs missing/invalid artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_evaluation_dir(tmp_dir)
        report = validate_input_artifacts(tmp_dir, strict=True)
        assert report["status"] == "PASS"
        assert report["l6_leakage_detected"] == False

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Missing test-l6
        create_mock_evaluation_dir(tmp_dir, num_val=120, num_id=120, num_l6=100)
        import shutil
        shutil.rmtree(os.path.join(tmp_dir, "test-l6"))
        report = validate_input_artifacts(tmp_dir, strict=False)
        assert "test-l6" in report["missing_splits"]


def test_l6_leakage_detection():
    """Verify validate_input_artifacts detects L6 contamination in validation or test-id."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_evaluation_dir(tmp_dir)
        # Contaminate validation split with L6
        val_csv = os.path.join(tmp_dir, "validation", "episodes.csv")
        df = pd.read_csv(val_csv)
        df["mutation_level"] = df["mutation_level"].astype(str)
        df.loc[0, "mutation_level"] = "L6"
        df.to_csv(val_csv, index=False)

        report = validate_input_artifacts(tmp_dir, strict=True)
        assert report["status"] == "FAIL"
        assert report["l6_leakage_detected"] == True



def test_normalized_loader():
    """Verify load_normalized_dataset concatenates all episode CSVs and preserves required schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_evaluation_dir(tmp_dir, num_val=120, num_id=120, num_l6=100)
        df_all = load_normalized_dataset(tmp_dir)

        assert len(df_all) == 340
        assert set(df_all["split"].unique()) == {"validation", "test-id", "test-l6"}
        assert "latency" in df_all.columns
        assert (df_all["latency"] == "NA").all()


def test_overall_metrics_and_gaps():
    """Verify compute_split_overall_metrics and generalization gap computations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_evaluation_dir(tmp_dir, num_val=120, num_id=120, num_l6=100)
        df_all = load_normalized_dataset(tmp_dir)

        val_m = compute_split_overall_metrics(df_all[df_all["split"] == "validation"], "validation")
        id_m = compute_split_overall_metrics(df_all[df_all["split"] == "test-id"], "test-id")
        l6_m = compute_split_overall_metrics(df_all[df_all["split"] == "test-l6"], "test-l6")

        assert val_m["success_rate"] == 100.0
        assert id_m["success_rate"] == 100.0
        assert l6_m["success_rate"] == 100.0

        gaps = compute_generalization_gaps(val_m, id_m, l6_m)
        assert gaps["val_to_test_id"]["abs_success_drop_pp"] == 0.0
        assert gaps["val_to_test_l6"]["abs_success_drop_pp"] == 0.0


def test_workflow_and_mutation_aggregation():
    """Verify workflow and mutation breakdown generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")
        create_mock_evaluation_dir(tmp_dir)
        df_all = load_normalized_dataset(tmp_dir)

        wf_df = generate_workflow_analysis(df_all, out_dir)
        assert len(wf_df) > 0
        assert set(wf_df["workflow"].unique()) == {"LOGIN", "SEARCH", "PROFILE", "CHECKOUT"}

        lvl_df, type_df = generate_mutation_analysis(df_all, out_dir)
        assert len(lvl_df) > 0
        assert any("HELD-OUT" in str(label) for label in lvl_df["mutation_level"])


def test_failure_taxonomy_fallback():
    """Verify failure taxonomy categorization when failure episodes are present."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        df_mock = pd.DataFrame([
            {
                "split": "validation",
                "episode_id": 1,
                "workflow": "LOGIN",
                "mutation_level": "L1",
                "mutation_seed": 11,
                "success": False,
                "return": -1.0,
                "wrong_step_actions": 1,
                "failure_reason": "incorrect_candidate",
                "truncated": False,
            },
            {
                "split": "test-l6",
                "episode_id": 2,
                "workflow": "SEARCH",
                "mutation_level": "L6",
                "mutation_seed": 11,
                "success": False,
                "return": -1.0,
                "wrong_step_actions": 0,
                "failure_reason": "max_steps_exceeded",
                "truncated": True,
            }
        ])

        tax_df, summary = generate_failure_taxonomy(df_mock, out_dir)
        assert len(tax_df) == 2
        assert summary["total_failures"] == 2
        assert summary["category_counts"]["Target present but DQN selected wrong candidate"] == 1
        assert summary["category_counts"]["Maximum-step truncation"] == 1


def test_missing_optional_metrics_handling():
    """Verify parse_numeric safely returns default value for NA / None / empty string."""
    assert parse_numeric("NA", default=0.0) == 0.0
    assert parse_numeric(None, default=-1.0) == -1.0
    assert parse_numeric("", default=5.0) == 5.0
    assert parse_numeric("8.8", default=0.0) == 8.8


def test_summary_artifacts_generation():
    """Verify serialization of presentation_summary.md, viva_defence.md, and phase16_report.md."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_mock_evaluation_dir(tmp_dir)
        df_all = load_normalized_dataset(tmp_dir)

        val_m = compute_split_overall_metrics(df_all[df_all["split"] == "validation"], "validation")
        id_m = compute_split_overall_metrics(df_all[df_all["split"] == "test-id"], "test-id")
        l6_m = compute_split_overall_metrics(df_all[df_all["split"] == "test-l6"], "test-l6")
        gaps = compute_generalization_gaps(val_m, id_m, l6_m)

        metrics = {
            "checkpoint": {"path": "chk.pt", "sha256": "abc"},
            "splits": {"validation": val_m, "test_id": id_m, "test_l6": l6_m},
            "generalization": gaps,
        }

        p_path = generate_presentation_summary(metrics, tmp_dir)
        v_path = generate_viva_defence(tmp_dir)
        r_path = generate_phase16_report(metrics, os.path.join(tmp_dir, "report.md"))

        assert os.path.exists(p_path)
        assert os.path.exists(v_path)
        assert os.path.exists(r_path)

        with open(p_path, "r", encoding="utf-8") as f:
            p_content = f.read()
            assert "Final DQN Evaluation Summary" in p_content
            assert "100.00%" in p_content

        with open(v_path, "r", encoding="utf-8") as f:
            v_content = f.read()
            assert "Why was DQN chosen?" in v_content
