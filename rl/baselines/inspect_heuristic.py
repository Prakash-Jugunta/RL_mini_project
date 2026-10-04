"""
CLI inspection utility for PublicFeatureHeuristicPolicy candidate scoring breakdown.
Does NOT display private ground-truth semantic roles or answer keys.
"""

import sys
import numpy as np
from typing import List, Optional

from rl.env.types import UICandidate
from rl.baselines.policies import PublicFeatureHeuristicPolicy


def inspect_heuristic_decision(
    agent_intent: str,
    action_type: str,
    candidates: List[UICandidate],
    action_mask: Optional[np.ndarray] = None,
) -> int:
    policy = PublicFeatureHeuristicPolicy()
    num_cand = len(candidates)
    if action_mask is None:
        action_mask = np.zeros(20, dtype=np.int8)
        action_mask[:min(num_cand, 20)] = 1

    print(f"Intent: {agent_intent}")
    print(f"Action type: {action_type}")
    print("-" * 50)

    scores_list = []
    valid_indices = np.where(action_mask == 1)[0]

    for idx in valid_indices:
        if idx >= len(candidates):
            continue
        cand = candidates[idx]
        score_dict = policy.compute_candidate_score(agent_intent, action_type, cand)
        scores_list.append((idx, cand, score_dict))

        print(f"Candidate {idx}")
        print(f"  tag: {cand.tag}")
        print(f"  text: {cand.text}")
        print(f"  placeholder: {cand.placeholder}")
        print(f"  aria-label: {cand.aria_label}")
        print(f"  semantic score:   {score_dict['semantic_score']:.4f}")
        print(f"  structural score: {score_dict['structural_score']:.4f}")
        print(f"  class score:      {score_dict['class_score']:.4f}")
        print(f"  final score:      {score_dict['final_score']:.4f}")
        print()

    selected_action = policy.select_action(
        agent_intent=agent_intent,
        action_type=action_type,
        candidates=candidates,
        action_mask=action_mask
    )

    print("-" * 50)
    print(f"Selected action: {selected_action}")
    return selected_action


if __name__ == "__main__":
    # Example diagnostic demo run
    sample_candidates = [
        UICandidate(candidate_id=0, tag="button", text="Cancel", attributes={"class": "btn-secondary"}),
        UICandidate(candidate_id=1, tag="input", element_type="text", placeholder="Enter Username", attributes={"name": "username", "id": "user-input"}),
        UICandidate(candidate_id=2, tag="button", text="Sign In / Login", attributes={"class": "btn-primary", "id": "btn-login"}),
    ]
    inspect_heuristic_decision(
        agent_intent="enter username",
        action_type="fill",
        candidates=sample_candidates
    )
