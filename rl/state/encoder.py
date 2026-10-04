from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if TYPE_CHECKING:
    from rl.env.types import UICandidate
    from rl.env.workflow import WorkflowStep

from rl.state.feature_schema import StateFeatureSchema
from rl.state.text_encoder import DeterministicTextEncoder

logger = logging.getLogger(__name__)

STATE_ENCODING_VERSION = "phase6-public-prev-candidate-v2"

# Known tag categories for one-hot encoding (8 dims)
TAG_CATEGORIES = ["input", "button", "a", "select", "textarea", "div", "text_node", "other"]
# Known element types for one-hot encoding (6 dims)
TYPE_CATEGORIES = ["text", "password", "submit", "button", "checkbox_radio", "other"]
# Known workflows for one-hot encoding (4 dims)
WORKFLOW_KEYS = ["LOGIN", "SEARCH", "PROFILE", "CHECKOUT"]
# Known action types for one-hot encoding (3 dims)
ACTION_TYPES = ["fill", "click", "verify"]


class StateEncoder:
    """
    Phase 6 State Encoder converting agent-visible UI state into fixed-size numeric observations.

    Key Constraints & Privacy Enforcement:
    --------------------------------------
    - EXCLUDES evaluator ground-truth (data-semantic-role, expected_role, PrivateEvaluatorMetadata, target_index).
    - EXCLUDES experiment metadata (mutation_level, mutation_seed).
    - Preserves Phase 5 candidate ordering.
    - Uses deterministic lightweight feature hashing for text embeddings.
    - Uses fixed capacity MAX_CANDIDATES=20 with explicit zero-padding and candidate mask.
    """

    def __init__(
        self,
        max_candidates: int = 20,
        viewport_width: float = 1280.0,
        viewport_height: float = 800.0,
    ):
        self.schema = StateFeatureSchema(max_candidates=max_candidates)
        self.max_candidates = max_candidates
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.text_encoder = DeterministicTextEncoder(n_features=self.schema.candidate_text_dim)

    def get_observation_space(self) -> spaces.Dict:
        """
        Defines the Gymnasium Dict observation space for Phase 6.
        """
        return spaces.Dict({
            "objective": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.schema.objective_dim,),
                dtype=np.float32,
            ),
            "candidates": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self.schema.candidate_matrix_shape,
                dtype=np.float32,
            ),
            "candidate_mask": spaces.Box(
                low=0,
                high=1,
                shape=self.schema.candidate_mask_shape,
                dtype=np.int8,
            ),
            "context": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.schema.context_dim,),
                dtype=np.float32,
            ),
        })

    def encode_structural_features(self, candidate: UICandidate) -> np.ndarray:
        """
        Encodes candidate tag, element type, visibility, and attribute flags into a 20-dim float32 vector.
        """
        features = np.zeros(self.schema.candidate_structural_dim, dtype=np.float32)

        # 1. Tag One-Hot (8 dims: 0..7)
        tag = (candidate.tag or "").lower()
        if tag == "input":
            features[0] = 1.0
        elif tag == "button":
            features[1] = 1.0
        elif tag == "a":
            features[2] = 1.0
        elif tag == "select":
            features[3] = 1.0
        elif tag == "textarea":
            features[4] = 1.0
        elif tag == "div":
            features[5] = 1.0
        elif tag in ("p", "span", "h1", "h2", "h3", "h4", "h5", "h6"):
            features[6] = 1.0  # text_node
        else:
            features[7] = 1.0  # other

        # 2. Type One-Hot (6 dims: 8..13)
        elem_type = (candidate.element_type or "").lower()
        if elem_type == "text":
            features[8] = 1.0
        elif elem_type == "password":
            features[9] = 1.0
        elif elem_type == "submit":
            features[10] = 1.0
        elif elem_type == "button":
            features[11] = 1.0
        elif elem_type in ("checkbox", "radio"):
            features[12] = 1.0
        else:
            features[13] = 1.0

        # 3. Boolean flags & attributes (6 dims: 14..19)
        features[14] = 1.0 if candidate.visible else 0.0
        features[15] = 1.0 if candidate.enabled else 0.0
        features[16] = 1.0 if bool(candidate.text and candidate.text.strip()) else 0.0
        features[17] = 1.0 if bool(candidate.placeholder and candidate.placeholder.strip()) else 0.0
        features[18] = 1.0 if bool(candidate.attributes.get("aria-label")) else 0.0
        features[19] = min(float(len(candidate.attributes)), 10.0) / 10.0

        return features

    def encode_visual_features(self, candidate: UICandidate) -> np.ndarray:
        """
        Encodes bounding box geometry normalized by viewport dimensions (1280x800) into a 7-dim float32 vector.
        """
        w = max(self.viewport_width, 1.0)
        h = max(self.viewport_height, 1.0)

        cx = float(candidate.x) if candidate.x is not None else 0.0
        cy = float(candidate.y) if candidate.y is not None else 0.0
        cw = float(candidate.width) if candidate.width is not None else 0.0
        ch = float(candidate.height) if candidate.height is not None else 0.0

        x_norm = float(np.clip(cx / w, 0.0, 1.0))
        y_norm = float(np.clip(cy / h, 0.0, 1.0))
        width_norm = float(np.clip(cw / w, 0.0, 1.0))
        height_norm = float(np.clip(ch / h, 0.0, 1.0))

        center_x_norm = float(np.clip((cx + cw / 2.0) / w, 0.0, 1.0))
        center_y_norm = float(np.clip((cy + ch / 2.0) / h, 0.0, 1.0))
        area_ratio = float(np.clip((cw * ch) / (w * h), 0.0, 1.0))

        return np.array([
            x_norm,
            y_norm,
            width_norm,
            height_norm,
            center_x_norm,
            center_y_norm,
            area_ratio,
        ], dtype=np.float32)

    def encode_context_features(
        self,
        workflow_id: str,
        action_type: str,
        current_step_index: int,
        total_steps: int,
        previous_action: int = -1,
        previous_success: float = 0.0,
        has_previous_action: bool = False,
        previous_candidate: Optional[UICandidate] = None,
    ) -> np.ndarray:
        """
        Encodes workflow identity, action operation type, progress, previous selected candidate public representation,
        and previous execution success into a 30-dim float32 vector.

        Permutation-Invariant Public Context Encoding:
        - Dims 0..3   : Workflow ID One-Hot (4 dims)
        - Dims 4..6   : Operation Type One-Hot (3 dims)
        - Dim 7       : Normalized Step Progress (1 dim)
        - Dim 8       : Has Previous Action flag (1 dim)
        - Dims 9..28  : Public Structural Features of Previously Selected Candidate (20 dims)
                        (Permutation-invariant: depends ONLY on candidate tag, type, and public attributes)
        - Dim 29      : Previous Action Execution Success (1 dim)
        """
        context = np.zeros(self.schema.context_dim, dtype=np.float32)

        # 1. Workflow ID One-Hot (4 dims: 0..3)
        if workflow_id in WORKFLOW_KEYS:
            w_idx = WORKFLOW_KEYS.index(workflow_id)
            context[w_idx] = 1.0

        # 2. Operation Type One-Hot (3 dims: 4..6)
        if action_type in ACTION_TYPES:
            a_idx = ACTION_TYPES.index(action_type)
            context[4 + a_idx] = 1.0

        # 3. Normalized Step Progress (1 dim: 7)
        denom = max(float(total_steps), 1.0)
        context[7] = float(current_step_index) / denom

        # 4. Has Previous Action (1 dim: 8)
        context[8] = 1.0 if has_previous_action else 0.0

        # 5. Stable Public Representation of Previously Selected Candidate (20 dims: 9..28)
        if has_previous_action and previous_candidate is not None:
            prev_struct = self.encode_structural_features(previous_candidate)
            context[9:29] = prev_struct
        # A raw previous candidate slot is intentionally ignored. Candidate slots
        # are transient and are permuted between decisions.

        # 6. Previous Action Success (1 dim: 29)
        context[29] = float(previous_success)

        return context

    def encode(
        self,
        workflow_id: str,
        step: WorkflowStep,
        current_step_index: int,
        total_steps: int,
        candidates: List[UICandidate],
        previous_action: int = -1,
        previous_success: float = 0.0,
        has_previous_action: bool = False,
        previous_candidate: Optional[UICandidate] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Constructs the full Phase 6 RL observation dict.
        """
        # 1. Objective Representation (64 dims)
        intent_text = step.agent_intent or step.step_id.replace("_", " ").lower()
        objective_vec = self.text_encoder.encode_text(intent_text)

        # 2. Candidate Matrix (20, 91) and Candidate Mask (20,)
        candidate_matrix = np.zeros(self.schema.candidate_matrix_shape, dtype=np.float32)
        mask = np.zeros(self.schema.candidate_mask_shape, dtype=np.int8)

        num_candidates = min(len(candidates), self.max_candidates)
        if num_candidates > 0:
            # Batch text serialization & encoding for speed
            candidate_texts = [
                self.text_encoder.serialize_candidate(c)
                for c in candidates[:num_candidates]
            ]
            text_embeddings = self.text_encoder.encode_batch(candidate_texts)

            for i in range(num_candidates):
                cand = candidates[i]
                text_vec = text_embeddings[i]
                struct_vec = self.encode_structural_features(cand)
                vis_vec = self.encode_visual_features(cand)

                # Concatenate 64 + 20 + 7 = 91 dims
                cand_feature = np.concatenate([text_vec, struct_vec, vis_vec], axis=0)
                candidate_matrix[i] = cand_feature
                mask[i] = 1

        # 3. Context Representation (30 dims)
        context_vec = self.encode_context_features(
            workflow_id=workflow_id,
            action_type=step.action_type,
            current_step_index=current_step_index,
            total_steps=total_steps,
            previous_action=previous_action,
            previous_success=previous_success,
            has_previous_action=has_previous_action,
            previous_candidate=previous_candidate,
        )

        obs = {
            "objective": objective_vec,
            "candidates": candidate_matrix,
            "candidate_mask": mask,
            "context": context_vec,
        }

        # Verify Gym observation space contract
        obs_space = self.get_observation_space()
        if not obs_space.contains(obs):
            logger.warning("Observation failed Gymnasium observation_space.contains check!")

        return obs
