"""
Phase 6 — State Inspection Utility
===================================

Run with:
    python -m rl.state.inspect_state
"""

import time
import numpy as np

from rl.env.workflow import WORKFLOW_REGISTRY
from rl.env.browser_adapter import MockBrowserAdapter
from rl.state.encoder import StateEncoder


def inspect_all_workflows():
    print("=" * 70)
    print("PHASE 6 STATE ENCODER INSPECTION")
    print("=" * 70)

    encoder = StateEncoder(max_candidates=20)
    space = encoder.get_observation_space()

    adapter = MockBrowserAdapter(max_candidates=10)

    total_latency_ms = 0.0
    runs = 0

    for w_id, workflow in WORKFLOW_REGISTRY.items():
        step = workflow.steps[0]
        adapter.reset(workflow.start_path, level=0, seed=42)
        candidates, _ = adapter.get_candidates(
            step_id=step.step_id,
            expected_role=step.expected_role,
            action_type=step.action_type
        )

        t0 = time.perf_counter()
        obs = encoder.encode(
            workflow_id=w_id,
            step=step,
            current_step_index=0,
            total_steps=len(workflow.steps),
            candidates=candidates,
            previous_action=-1,
            previous_success=0.0,
            has_previous_action=False
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        total_latency_ms += latency_ms
        runs += 1

        is_valid = space.contains(obs)

        print(f"\nWorkflow        : {w_id}")
        print(f"Start Path      : {workflow.start_path}")
        print(f"First Step ID   : {step.step_id}")
        print(f"Agent Intent    : {step.agent_intent!r}")
        print(f"Objective Shape : {obs['objective'].shape} (norm={np.linalg.norm(obs['objective']):.4f})")
        print(f"Candidates      : {int(obs['candidate_mask'].sum())} / {encoder.max_candidates}")
        print(f"Matrix Shape    : {obs['candidates'].shape}")
        print(f"Mask Shape      : {obs['candidate_mask'].shape}")
        print(f"Context Shape   : {obs['context'].shape}")
        print(f"Space Contains  : {is_valid}")
        print(f"Encoding Latency: {latency_ms:.3f} ms")

        # Inspect Candidate 0
        if int(obs['candidate_mask'].sum()) > 0:
            c0 = candidates[0]
            text_str = encoder.text_encoder.serialize_candidate(c0)
            c0_feat = obs['candidates'][0]
            print(f"  Candidate 0 String    : {text_str!r}")
            print(f"  Candidate 0 Text Norm : {np.linalg.norm(c0_feat[:64]):.4f}")
            print(f"  Candidate 0 Struct Dim: {c0_feat[64:84].shape}")
            print(f"  Candidate 0 Visual Dim: {c0_feat[84:].shape}")

    avg_latency = total_latency_ms / max(runs, 1)
    print("\n" + "=" * 70)
    print(f"Average State Encoding Latency: {avg_latency:.3f} ms")
    print("=" * 70)


if __name__ == "__main__":
    inspect_all_workflows()
