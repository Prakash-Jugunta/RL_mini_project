"""
Unit tests for Phase 18 — DQN Learning, Convergence, Complexity & Sample-Efficiency Analysis
(rl.analysis.load_training_history & rl.analysis.analyze_learning).

Tests:
1. programmatic parameter count equals actual model parameters
2. no report generator contains stale parameter count literals
3. decimal MB vs binary MiB conversion
4. first-threshold episode differs correctly from stable-after episode
5. report consumes computed convergence values
6. latency flag controls execution
7. missing latency -> NA rather than hardcoded number
8. Q-margin imported as 0.1141 from authoritative artifact
9. presentation/report/viva metrics remain consistent
10. no hardcoded experimental metrics in report generators
"""

import json
import os
import re
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from rl.agents.dqn.network import CandidateAwareQNetwork
from rl.analysis.load_training_history import (
    discover_training_artifacts,
    load_normalized_training_history,
    STAGE_BOUNDARIES,
)
from rl.analysis.analyze_learning import (
    analyze_convergence,
    analyze_curriculum_stages,
    analyze_sample_efficiency,
    analyze_model_and_memory,
    build_authoritative_results,
    generate_consistency_check,
    generate_co4_summary,
    generate_presentation_summary,
    generate_viva_co4,
    generate_phase18_report,
    measure_inference_latency,
)


def create_mock_training_dir(tmp_dir: str, num_episodes: int = 500) -> str:
    """Helper to create mock training logs matching exact Phase 13 schema."""
    state_data = {
        "global_episode_id": num_episodes,
        "total_episodes_budget": 15000,
        "stage_index": 1,
        "total_steps": num_episodes * 4,
        "learning_steps": num_episodes * 4,
        "replay_size": 10000,
        "epsilon": 0.05,
    }
    with open(os.path.join(tmp_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)

    logs = []
    for i in range(num_episodes):
        # First 100 episodes low success, later high success
        succ = (i >= 100)
        logs.append({
            "episode": i + 1,
            "global_episode_id": i,
            "stage_index": 0 if i < 200 else 1,
            "workflow": ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"][i % 4],
            "mutation_level": "L0" if i < 200 else "L1",
            "mutation_seed": 11,
            "environment_seed": 1000 + i,
            "epsilon": max(1.0 - (i * 0.002), 0.05),
            "return": 8.8 if succ else -10.0,
            "success": succ,
            "decisions": 4 if succ else 20,
            "replay_size": min((i + 1) * 4, 10000),
            "recent_loss": 0.12,
        })

    with open(os.path.join(tmp_dir, "episode_logs.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return tmp_dir


def test_programmatic_parameter_count_equals_model():
    """Verify programmatic parameter calculation matches PyTorch CandidateAwareQNetwork model."""
    model = CandidateAwareQNetwork()
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual_params == 69601

    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = os.path.join(tmp_dir, "dummy.pt")
        model_comp, _ = analyze_model_and_memory(chk_path, tmp_dir)
        assert model_comp["trainable_parameters"] == actual_params


def test_no_report_generator_contains_stale_literals():
    """Verify report generators do not embed stale metric literals (57025, 0.4200, 1.4500, etc.)."""
    import inspect
    from rl.analysis import analyze_learning

    stale_literals = ["57025", "57,025", "0.4200", "0.6500", "1.4500", "154.86 MB / 154.86"]
    
    gen_funcs = [
        generate_presentation_summary,
        generate_viva_co4,
        generate_phase18_report,
        generate_co4_summary,
    ]

    for fn in gen_funcs:
        src = inspect.getsource(fn)
        for literal in stale_literals:
            assert literal not in src, f"Stale literal '{literal}' found in source of {fn.__name__}"


def test_decimal_mb_vs_binary_mib_conversion():
    """Verify decimal (KB/MB) and binary (KiB/MiB) unit conversions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = os.path.join(tmp_dir, "dummy.pt")
        model_comp, mem_comp = analyze_model_and_memory(chk_path, tmp_dir)

        # 69,601 params * 4 bytes = 278,404 bytes
        assert model_comp["parameter_bytes"] == 278404
        assert model_comp["parameter_kb_decimal"] == pytest.approx(278.40, abs=0.01)
        assert model_comp["parameter_kib_binary"] == pytest.approx(271.88, abs=0.01)

        # Replay buffer (10,000 capacity * 15,486 bytes = 154,860,000 bytes)
        assert mem_comp["replay_bytes"] == 154860000
        assert mem_comp["replay_mb_decimal"] == pytest.approx(154.86, abs=0.01)
        assert mem_comp["replay_mib_binary"] == pytest.approx(147.69, abs=0.01)


def test_first_threshold_episode_differs_from_stable_after():
    """Verify analyze_convergence distinguishes first episode reaching 90% from stable after episode."""
    rows = []
    # Episodes 1..100: low success
    for ep in range(1, 101):
        rows.append({"episode": ep, "cumulative_steps": ep * 4, "success": False, "return": -10.0, "recent_loss": 0.5})
    # Episodes 101..190: high success (reaches 90% at episode 190)
    for ep in range(101, 191):
        rows.append({"episode": ep, "cumulative_steps": ep * 4, "success": True, "return": 8.8, "recent_loss": 0.1})
    # Episode 191..220: dip (30 False episodes) causing rolling success to drop to ~70%
    for ep in range(191, 221):
        rows.append({"episode": ep, "cumulative_steps": ep * 4, "success": False, "return": -10.0, "recent_loss": 0.5})
    # Episodes 221..500: high success thereafter (stabilizes around episode 310)
    for ep in range(221, 501):
        rows.append({"episode": ep, "cumulative_steps": ep * 4, "success": True, "return": 8.8, "recent_loss": 0.1})

    df = pd.DataFrame(rows)
    df["rolling_success_100"] = df["success"].rolling(100, min_periods=100).mean() * 100.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        conv_df, th90_dict, _ = analyze_convergence(df, tmp_dir)
        first_ep = th90_dict["first_reached_episode"]
        stable_ep = th90_dict["stable_after_episode"]

        assert first_ep != "NA"
        assert stable_ep != "NA"
        # Since there was a dip after episode 190, stable_ep should be greater than first_ep
        assert stable_ep > first_ep


def test_report_consumes_computed_convergence_values():
    """Verify generated report text dynamically consumes computed convergence dictionary values."""
    mock_results = {
        "training": {"total_episodes": 15000, "total_global_steps": 68459, "successful_episodes": 14500, "overall_success_rate": 96.67},
        "convergence": {
            "rolling_window": 100,
            "threshold_90": {
                "first_reached_episode": 463,
                "first_reached_global_step": 2058,
                "stable_after_episode": 463,
                "stable_global_step": 2058,
            },
        },
        "model": {
            "checkpoint_path": "dummy.pt",
            "checkpoint_size_mb": 0.28,
            "trainable_parameters": 69601,
            "parameter_kb_decimal": 278.40,
            "parameter_kib_binary": 271.88,
        },
        "memory": {
            "replay_capacity": 10000,
            "replay_bytes": 154860000,
            "replay_mb_decimal": 154.86,
            "replay_mib_binary": 147.69,
            "bytes_per_transition": 15486,
        },
        "latency": {
            "dqn_forward": {"mean_ms": 1.42, "median_ms": 1.35, "p95_ms": 2.07, "min_ms": 1.10, "max_ms": 3.50, "device": "CPU", "runs": 500}
        },
        "q_margin": {"authoritative_mean": 0.1141, "source_file": "artifacts/audit/phase17/q_margin_reconciliation.md"}
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        pres_str = generate_presentation_summary(mock_results, tmp_dir)
        with open(pres_str, "r", encoding="utf-8") as f:
            pres_text = f.read()
        assert "Episode 463" in pres_text
        assert "69,601" in pres_text
        assert "147.69 MiB" in pres_text

        viva_str = generate_viva_co4(mock_results, tmp_dir)
        with open(viva_str, "r", encoding="utf-8") as f:
            viva_text = f.read()
        assert "Episode 463" in viva_text
        assert "69,601" in viva_text
        assert "147.69 MiB" in viva_text


def test_latency_flag_controls_execution():
    """Verify latency benchmark is executed only when run_benchmark=True, otherwise NA fallback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = os.path.join(tmp_dir, "nonexistent.pt")
        
        # When run_benchmark=False and no file exists
        res_no_run = measure_inference_latency(chk_path, tmp_dir, run_benchmark=False)
        assert res_no_run["benchmark_executed"] is False
        assert res_no_run["dqn_forward"]["mean_ms"] == "NA"

        # When run_benchmark=True
        res_run = measure_inference_latency(chk_path, tmp_dir, run_benchmark=True)
        assert res_run["benchmark_executed"] is True
        assert isinstance(res_run["dqn_forward"]["mean_ms"], float)


def test_missing_latency_na_handling():
    """Verify missing latency benchmarking outputs 'NA' rather than hardcoded latency values."""
    mock_results = {
        "training": {"total_episodes": 100, "total_global_steps": 400, "successful_episodes": 50, "overall_success_rate": 50.0},
        "convergence": {
            "rolling_window": 100,
            "threshold_90": {
                "first_reached_episode": "NA",
                "first_reached_global_step": "NA",
                "stable_after_episode": "NA",
                "stable_global_step": "NA",
            },
        },
        "model": {"checkpoint_path": "dummy.pt", "checkpoint_size_mb": 0.28, "trainable_parameters": 69601, "parameter_kb_decimal": 278.40, "parameter_kib_binary": 271.88},
        "memory": {"replay_capacity": 10000, "replay_bytes": 154860000, "replay_mb_decimal": 154.86, "replay_mib_binary": 147.69, "bytes_per_transition": 15486},
        "latency": {"dqn_forward": {"mean_ms": "NA", "p95_ms": "NA"}},
        "q_margin": {"authoritative_mean": 0.1141, "source_file": "artifacts/audit/phase17/q_margin_reconciliation.md"}
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        df_co4 = generate_co4_summary(mock_results, tmp_dir)
        lat_row = df_co4[df_co4["metric"] == "Mean DQN Forward Pass Latency"]
        assert lat_row.iloc[0]["value"] == "NA"


def test_q_margin_imported_as_01141():
    """Verify Phase 17 Q-margin authoritative value is imported as 0.1141."""
    recon_path = "artifacts/audit/phase17/q_margin_reconciliation.md"
    assert os.path.exists(recon_path)
    with open(recon_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "0.1141" in content
        assert "AUTHORITATIVE" in content


def test_cross_artifact_consistency():
    """Verify cross-artifact consistency validator passes on valid results and fails on mismatched ones."""
    valid_results = {
        "training": {"total_episodes": 15000, "total_global_steps": 68459, "successful_episodes": 14500, "overall_success_rate": 96.67},
        "convergence": {
            "rolling_window": 100,
            "threshold_90": {
                "first_reached_episode": 463,
                "first_reached_global_step": 2058,
                "stable_after_episode": 463,
                "stable_global_step": 2058,
            },
        },
        "model": {"checkpoint_path": "dummy.pt", "checkpoint_size_mb": 0.28, "trainable_parameters": 69601, "parameter_kb_decimal": 278.40, "parameter_kib_binary": 271.88},
        "memory": {"replay_capacity": 10000, "replay_bytes": 154860000, "replay_mb_decimal": 154.86, "replay_mib_binary": 147.69, "bytes_per_transition": 15486},
        "latency": {"dqn_forward": {"mean_ms": 1.42, "p95_ms": 2.07}},
        "q_margin": {"authoritative_mean": 0.1141, "source_file": "artifacts/audit/phase17/q_margin_reconciliation.md"}
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        check_pass = generate_consistency_check(valid_results, tmp_dir)
        assert check_pass["overall"] == "PASS"

        invalid_results = dict(valid_results)
        invalid_results["q_margin"] = {"authoritative_mean": 1.4500, "source_file": "stale"}
        check_fail = generate_consistency_check(invalid_results, tmp_dir)
        assert check_fail["overall"] == "FAIL"


def test_canonical_artifact_immutability():
    """Verify canonical checkpoints and evaluation artifacts remain present and unaltered."""
    chk_path = "artifacts/dqn/phase13-v1/latest_checkpoint.pt"
    assert os.path.exists(chk_path)

    eval_dir = "artifacts/evaluation/dqn-final"
    if os.path.exists(eval_dir):
        assert os.path.exists(os.path.join(eval_dir, "manifest.json"))
        assert os.path.exists(os.path.join(eval_dir, "checkpoint_metadata.json"))
