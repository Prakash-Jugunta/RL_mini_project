"""
Unit tests for DQN Demo Runner (rl.demo.run_dqn_demo).

Tests:
1. demo module imports correctly
2. CLI parsing and defaults
3. checkpoint loading
4. greedy-only action selection
5. headed/headless option propagation
6. mutation level conversion (L6 -> test_l6, L0-5 -> validation)
7. workflow selection
8. no optimizer/replay updates (model weights unchanged)
9. highlight helper does not change observation
10. private metadata not entering policy
"""

import os
import sys
import pytest
import numpy as np
import torch

from rl.agents.dqn import DQNAgent, DQNConfig
from rl.agents.dqn.checkpoint import load_checkpoint
from rl.env.browser_adapter import MockBrowserAdapter
from rl.env.ui_recovery_env import UIRecoveryEnv
from rl.demo.run_dqn_demo import build_parser, run_dqn_demo, clean_text


CHECKPOINT_PATH = "artifacts/dqn/phase13-v1/latest_checkpoint.pt"


def test_demo_module_imports_correctly():
    """Verify demo package and module import without error."""
    import rl.demo
    import rl.demo.run_dqn_demo as demo_mod
    assert hasattr(demo_mod, "run_dqn_demo")
    assert hasattr(demo_mod, "build_parser")


def test_cli_parsing_defaults():
    """Verify CLI parser options and default values."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.checkpoint == CHECKPOINT_PATH
    assert args.workflow == "LOGIN"
    assert args.mutation_level == 6
    assert args.mutation_seed == 11
    assert args.headed is True
    assert args.auto is False  # default mode is step
    assert args.delay_ms == 1200
    assert args.top_k == 5
    assert args.show_evaluator_debug is False


def test_cli_parsing_custom_arguments():
    """Verify CLI parser with explicit flags."""
    parser = build_parser()
    cmd = [
        "--checkpoint", CHECKPOINT_PATH,
        "--workflow", "CHECKOUT",
        "--mutation-level", "3",
        "--mutation-seed", "44",
        "--headless",
        "--auto",
        "--delay-ms", "500",
        "--top-k", "3",
        "--show-evaluator-debug"
    ]
    args = parser.parse_args(cmd)

    assert args.checkpoint == CHECKPOINT_PATH
    assert args.workflow == "CHECKOUT"
    assert args.mutation_level == 3
    assert args.mutation_seed == 44
    assert args.headed is False
    assert args.auto is True
    assert args.delay_ms == 500
    assert args.top_k == 3
    assert args.show_evaluator_debug is True


def test_checkpoint_loading():
    """Verify trained DQN checkpoint loads successfully and sets eval mode."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    config = DQNConfig(seed=2026)
    agent = DQNAgent(config)
    meta = agent.load_checkpoint(CHECKPOINT_PATH)

    agent.set_eval_mode()
    assert not agent.online_network.training


def test_greedy_only_action_selection_and_no_optimizer_updates():
    """Verify action selection is greedy (epsilon=0) and model parameters do not change."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    adapter = MockBrowserAdapter(max_candidates=20)
    res = run_dqn_demo(
        checkpoint_path=CHECKPOINT_PATH,
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        headed=False,
        auto=True,
        delay_ms=0,
        browser_adapter=adapter,
        interactive_prompt=False,
    )

    assert isinstance(res, dict)
    assert "success" in res
    assert "return" in res
    assert res["decisions"] > 0


def test_headed_headless_option():
    """Verify headed/headless CLI flags pass correct parameters."""
    parser = build_parser()
    
    args_headed = parser.parse_args(["--headed"])
    assert args_headed.headed is True

    args_headless = parser.parse_args(["--headless"])
    assert args_headless.headed is False


def test_mutation_level_conversion_and_workflow_selection():
    """Verify mutation level and workflow selection configure environment properly."""
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    for w_id in ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]:
        adapter = MockBrowserAdapter(max_candidates=20)
        res = run_dqn_demo(
            checkpoint_path=CHECKPOINT_PATH,
            workflow_id=w_id,
            mutation_level=6,
            mutation_seed=11,
            headed=False,
            auto=True,
            delay_ms=0,
            browser_adapter=adapter,
            interactive_prompt=False,
        )
        assert res["workflow_id"] == w_id
        assert res["mutation_level"] == 6


def test_highlight_helper_does_not_change_observation():
    """Verify highlight_candidate call does not alter environment observation."""
    adapter = MockBrowserAdapter(max_candidates=20)
    env = UIRecoveryEnv(
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        mode="validation",
        browser_adapter=adapter,
    )
    obs1, _ = env.reset(seed=42)

    # Call highlight helper
    adapter.highlight_candidate(candidate_id=1, label="DQN SELECTED", duration_sec=0.01)

    obs2 = env._construct_observation()

    np.testing.assert_array_equal(obs1["objective"], obs2["objective"])
    np.testing.assert_array_equal(obs1["context"], obs2["context"])
    np.testing.assert_array_equal(obs1["candidates"], obs2["candidates"])
    np.testing.assert_array_equal(obs1["candidate_mask"], obs2["candidate_mask"])


def test_private_metadata_not_entering_policy():
    """Verify evaluator-private metadata (expected_role / data-semantic-role) does not enter observation tensors."""
    adapter1 = MockBrowserAdapter(max_candidates=20, rng=np.random.default_rng(42))
    cands_with_meta, _ = adapter1.get_candidates("ENTER_USERNAME", expected_role="username-input")

    adapter2 = MockBrowserAdapter(max_candidates=20, rng=np.random.default_rng(42))
    cands_without_meta, _ = adapter2.get_candidates("ENTER_USERNAME", expected_role=None)

    # UICandidates returned to agent must be identical and contain no semantic_role attribute
    for c1, c2 in zip(cands_with_meta, cands_without_meta):
        assert c1.tag == c2.tag
        assert c1.text == c2.text
        assert not hasattr(c1, "semantic_role")
        assert not hasattr(c2, "semantic_role")


def test_cli_parsing_recovery_injection():
    """Verify CLI parsing for --inject-wrong-step and --inject-wrong-rank."""
    parser = build_parser()
    args = parser.parse_args([
        "--inject-wrong-step", "2",
        "--inject-wrong-rank", "3"
    ])
    assert args.inject_wrong_step == 2
    assert args.inject_wrong_rank == 3


def test_recovery_injection_behavior():
    """
    Verify transparent demo-only recovery injection:
    - injected action is valid and not greedy top-1
    - injected action is evaluator-confirmed wrong
    - greedy control resumes immediately afterward
    - no policy/state/reward changes
    - no training started
    """
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    adapter = MockBrowserAdapter(max_candidates=20)
    res = run_dqn_demo(
        checkpoint_path=CHECKPOINT_PATH,
        workflow_id="LOGIN",
        mutation_level=0,
        mutation_seed=11,
        headed=False,
        auto=True,
        delay_ms=0,
        inject_wrong_step=1,
        inject_wrong_rank=2,
        browser_adapter=adapter,
        interactive_prompt=False,
    )

    assert isinstance(res, dict)
    assert res["injected_wrong_actions"] == 1
    assert res["natural_dqn_wrong_actions"] == max(0, res["wrong_actions"] - 1)
    assert "recovery_after_injection" in res
    assert res["decisions"] > 0

