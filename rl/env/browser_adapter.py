from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from rl.env.types import UICandidate, PrivateEvaluatorMetadata


class BrowserAdapter(ABC):
    """
    Abstract interface decoupling the Gymnasium environment from Playwright/DOM details.

    Allows mock adapter testing during Phase 4 and real Playwright browser
    interaction in future phases.
    """

    @abstractmethod
    def reset(self, start_path: str, level: int, seed: int) -> None:
        """Reset browser to initial page path under given mutation level and seed."""
        pass

    @abstractmethod
    def get_candidates(
        self,
        step_id: str,
        expected_role: Optional[str],
        action_type: str = "click",
    ) -> Tuple[List[UICandidate], List[PrivateEvaluatorMetadata]]:
        """
        Returns (agent_candidates, private_metadata).

        agent_candidates: AGENT-VISIBLE candidates (NO ground-truth semantic roles).
        private_metadata: PRIVATE EVALUATOR ground truth (used strictly by environment).

        action_type controls which extraction mode is used:
            "fill"   -> interactive (inputs, buttons, links)
            "click"  -> interactive
            "verify" -> verification (headings, alerts, text elements)

        The mode MUST be derived from action_type, not from expected_role.
        """
        pass

    @abstractmethod
    def execute_action(
        self,
        candidate_id: int,
        action_type: str,
        value: Optional[str] = None
    ) -> bool:
        """Executes an action (fill/click) on the target candidate_id."""
        pass

    @abstractmethod
    def validate_step(self, expected_role: Optional[str]) -> bool:
        """Privately checks if the last action satisfied the step's expected semantic role."""
        pass

    @abstractmethod
    def validate_page_state(self, success_condition: str, expected_value: str) -> bool:
        """Evaluates a page-state assertion (e.g. url_equals) directly against browser state."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up browser resources."""
        pass

    def highlight_candidate(
        self,
        candidate_id: int,
        label: str = "DQN SELECTED",
        duration_sec: float = 1.0,
    ) -> bool:
        """Visually outline the selected candidate in the browser for demo presentation."""
        return True



class MockBrowserAdapter(BrowserAdapter):
    """
    Deterministic mock browser adapter for Phase 4 unit testing and Gymnasium env checking.

    SEED SEPARATION CONCEPTS:
      - Environment / Policy Reset Seed (via self.rng): Controls candidate permutation & ordering.
      - Mutation Level & Seed (via self.current_level, self.current_seed): Controls mock element attributes,
        text mutations, and distractor text deterministically.
    """

    def __init__(self, max_candidates: int = 10, rng: Optional[np.random.Generator] = None):
        self.max_candidates = max_candidates
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.current_path = "/"
        self.current_level = 0
        self.current_seed = 42
        self.last_executed_candidate_id: Optional[int] = None
        self.private_metadata_map: Dict[int, PrivateEvaluatorMetadata] = {}

    def set_rng(self, rng: np.random.Generator) -> None:
        """Set the Gymnasium environment policy-side RNG (controls candidate ordering)."""
        self.rng = rng

    def reset(self, start_path: str, level: int, seed: int) -> None:
        """Sets deterministic mutation configuration (level & seed)."""
        self.current_path = start_path
        self.current_level = level
        self.current_seed = seed
        self.last_executed_candidate_id = None
        self.private_metadata_map.clear()

    def _get_mutation_rng(self) -> np.random.Generator:
        """
        Generates a deterministic RNG specifically for mock element attribute/text mutations
        based strictly on (mutation_level, mutation_seed).
        """
        seed_value = int(abs(hash((self.current_level, self.current_seed))) % (2**31))
        return np.random.default_rng(seed_value)

    def get_candidates(
        self,
        step_id: str,
        expected_role: Optional[str],
        action_type: str = "click",
    ) -> Tuple[List[UICandidate], List[PrivateEvaluatorMetadata]]:
        """
        Generates max_candidates UICandidates.
        Candidate text/attribute mutations are determined by (mutation_level, mutation_seed).
        Candidate ORDERING is permuted using self.rng (the Gymnasium reset seed).
        """
        candidates: List[UICandidate] = []
        private_meta: List[PrivateEvaluatorMetadata] = []

        mutation_rng = self._get_mutation_rng()
        candidate_ids = list(range(1, self.max_candidates + 1))

        # Target candidate
        target_id = candidate_ids[0]
        is_input = "ENTER" in step_id or "INPUT" in step_id
        if self.current_level == 0:
            target_text = step_id.replace("_", " ").title()
            target_placeholder = f"Enter {step_id.split('_')[-1].lower()}"
            css_class = "form-control"
        else:
            # Deterministic text mutation based on mutation_level & mutation_seed
            hex_suffix = hex(mutation_rng.integers(0x1000, 0xFFFF))[2:]
            target_text = f"Mutated {step_id.replace('_', ' ').title()} - {hex_suffix}"
            target_placeholder = f"Mutated {step_id.split('_')[-1].lower()} - {hex_suffix}"
            css_class = f"form-control-mutated-{self.current_level}"

        target_candidate = UICandidate(
            candidate_id=target_id,
            tag="input" if is_input else "button",
            element_type="text" if is_input else "submit",
            text=target_text,
            placeholder=target_placeholder,
            visible=True,
            enabled=True,
            x=100.0,
            y=200.0,
            width=150.0,
            height=40.0,
            attributes={"class": css_class}
        )
        candidates.append(target_candidate)
        private_meta.append(PrivateEvaluatorMetadata(candidate_id=target_id, semantic_role=expected_role))

        # Distractor candidates
        for i in range(1, self.max_candidates):
            cid = candidate_ids[i]
            d_suffix = hex(mutation_rng.integers(0x100, 0xFFF))[2:]
            distractor_candidate = UICandidate(
                candidate_id=cid,
                tag="button" if i % 2 == 0 else "a",
                element_type="button" if i % 2 == 0 else None,
                text=f"Distractor Element {i} ({d_suffix})",
                placeholder=None,
                visible=True,
                enabled=True,
                x=50.0 + i * 20.0,
                y=100.0 + i * 30.0,
                width=100.0,
                height=30.0,
                attributes={"class": "btn-distractor"}
            )
            candidates.append(distractor_candidate)
            private_meta.append(PrivateEvaluatorMetadata(candidate_id=cid, semantic_role=f"distractor-{i}"))

        # Permute candidates using environment reset RNG (self.rng)
        perm = self.rng.permutation(len(candidates))
        shuffled_candidates = [candidates[idx] for idx in perm]
        shuffled_meta = [private_meta[idx] for idx in perm]

        # Update private lookup map
        self.private_metadata_map = {meta.candidate_id: meta for meta in shuffled_meta}

        return shuffled_candidates, shuffled_meta

    def execute_action(
        self,
        candidate_id: int,
        action_type: str,
        value: Optional[str] = None
    ) -> bool:
        self.last_executed_candidate_id = candidate_id
        if self.current_path == "/products" and action_type == "click":
            meta = self.private_metadata_map.get(candidate_id)
            if meta and meta.semantic_role == "product-result":
                self.current_path = "/products/wireless-mouse"
        return True

    def validate_step(self, expected_role: Optional[str]) -> bool:
        if self.last_executed_candidate_id is None:
            return False
        meta = self.private_metadata_map.get(self.last_executed_candidate_id)
        if meta and expected_role is not None and meta.semantic_role == expected_role:
            return True
        return False

    def validate_page_state(self, success_condition: str, expected_value: str) -> bool:
        if success_condition == "url_equals":
            return self.current_path == expected_value or expected_value in self.current_path
        return False

    def close(self) -> None:
        pass
