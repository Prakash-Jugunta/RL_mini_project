import numpy as np
from typing import Optional, List, Dict, Any

from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.action.sampler import sample_valid_action
from rl.baselines.utils import (
    normalize_text,
    calculate_text_similarity,
    calculate_structural_compatibility,
)

RANDOM_POLICY_VERSION = "phase9-random-v1"
HEURISTIC_POLICY_VERSION = "phase9-heuristic-v1"
ORACLE_POLICY_VERSION = "phase9-oracle-v1"


class NoValidActionError(Exception):
    """Raised when baseline policy encounters an action mask with zero valid candidates."""
    pass


class RandomValidPolicy:
    """
    Baseline A — Random Valid Policy.
    Uniformly samples valid candidate indices where action_mask == 1 using Phase 7 sampler.
    Excludes padded candidates. Completely leak-free.
    """

    def __init__(self, seed: Optional[int] = None):
        self.version = RANDOM_POLICY_VERSION
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: Optional[Dict[str, np.ndarray]] = None,
        action_mask: Optional[np.ndarray] = None,
        **context: Any,
    ) -> int:
        mask = action_mask
        if mask is None and observation is not None:
            mask = observation.get("candidate_mask")

        if mask is None or not np.any(mask == 1):
            raise NoValidActionError("No valid candidate actions available in action_mask.")

        return sample_valid_action(mask, self.rng)


class PublicFeatureHeuristicPolicy:
    """
    Baseline B — Public-Feature Heuristic Policy.
    Deterministic, non-learning policy that ranks candidates using ONLY public candidate features
    and workflow intent/action_type.
    """

    def __init__(self):
        self.version = HEURISTIC_POLICY_VERSION

    def compute_candidate_score(
        self,
        agent_intent: str,
        action_type: str,
        candidate: UICandidate,
    ) -> Dict[str, float]:
        """
        Computes the weighted score breakdown for a single UICandidate based solely on public fields.
        """
        norm_intent = normalize_text(agent_intent)

        # Public string similarity components
        text_sim = calculate_text_similarity(norm_intent, candidate.text)
        placeholder_sim = calculate_text_similarity(norm_intent, candidate.placeholder)
        aria_sim = calculate_text_similarity(norm_intent, candidate.aria_label)

        # Attribute similarities
        name_attr = candidate.attributes.get("name") or candidate.attributes.get("title") or ""
        name_sim = calculate_text_similarity(norm_intent, name_attr)

        value_attr = candidate.attributes.get("value") or ""
        value_sim = calculate_text_similarity(norm_intent, value_attr)

        id_attr = candidate.attributes.get("id") or ""
        id_sim = calculate_text_similarity(norm_intent, id_attr)

        class_attr = candidate.attributes.get("class") or ""
        class_sim = calculate_text_similarity(norm_intent, class_attr)

        semantic_score = max(text_sim, placeholder_sim, aria_sim, name_sim, value_sim, 0.8 * id_sim)
        structural_score = calculate_structural_compatibility(action_type, candidate)
        class_score = class_sim

        final_score = (
            0.75 * semantic_score +
            0.20 * structural_score +
            0.05 * class_score
        )

        return {
            "semantic_score": float(semantic_score),
            "structural_score": float(structural_score),
            "class_score": float(class_score),
            "final_score": float(final_score),
        }

    def select_action(
        self,
        agent_intent: str,
        action_type: str,
        candidates: List[UICandidate],
        action_mask: np.ndarray,
        **context: Any,
    ) -> int:
        """
        Ranks valid candidates and returns the index of the highest-scoring candidate.
        Deterministic tie-breaking picks the lowest candidate index.
        """
        valid_indices = np.where(action_mask == 1)[0]
        if len(valid_indices) == 0 or len(candidates) == 0:
            raise NoValidActionError("No valid candidate actions available for heuristic policy.")

        best_score = -1.0
        best_index = -1

        for idx in valid_indices:
            if idx >= len(candidates):
                continue
            cand = candidates[idx]
            scores = self.compute_candidate_score(agent_intent, action_type, cand)
            score = scores["final_score"]

            if score > best_score:
                best_score = score
                best_index = int(idx)

        # Fallback to lowest valid candidate index if tie or all zero
        if best_index < 0:
            best_index = int(valid_indices[0])

        return best_index


class OraclePolicy:
    """
    DIAGNOSTIC ONLY — EVALUATOR-ASSISTED ORACLE.
    ENVIRONMENT SANITY CHECK ONLY. NOT A REAL BASELINE.
    Uses private evaluator metadata to pick ground-truth target candidate.
    """

    def __init__(self):
        self.version = ORACLE_POLICY_VERSION

    def select_action(
        self,
        expected_role: str,
        private_meta: List[PrivateEvaluatorMetadata],
        action_mask: np.ndarray,
        **context: Any,
    ) -> int:
        valid_indices = np.where(action_mask == 1)[0]
        if len(valid_indices) == 0:
            raise NoValidActionError("No valid candidate actions available for oracle policy.")

        for idx in valid_indices:
            if idx < len(private_meta):
                meta = private_meta[idx]
                if meta.semantic_role == expected_role:
                    return int(idx)

        # Fallback if target not found in candidate list
        return int(valid_indices[0])
