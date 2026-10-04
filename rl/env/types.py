from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

TRAIN_MUTATION_LEVELS: List[int] = [0, 1, 2, 3, 4, 5]
HELD_OUT_MUTATION_LEVEL: int = 6
VALID_MODES: set[str] = {"train", "validation", "test"}

PROHIBITED_ATTRIBUTE_KEYS: set[str] = {
    "semantic_role",
    "data_semantic_role",
    "data-semantic-role",
    "expected_role",
    "is_correct",
    "ground_truth",
    "target_role"
}


def sanitize_attributes(attrs: Dict[str, Any]) -> Dict[str, str]:
    """
    Sanitizes candidate element attributes dictionary to permanently prevent
    accidental ground-truth leakage into agent-visible observation/candidate structures.
    """
    if not attrs:
        return {}
    clean_attrs = {}
    for key, val in attrs.items():
        k_str = str(key)
        k_lower = k_str.lower().replace("-", "_")
        if k_lower in PROHIBITED_ATTRIBUTE_KEYS or "semantic_role" in k_lower or "ground_truth" in k_lower:
            continue
        clean_attrs[k_str] = str(val)
    return clean_attrs


@dataclass
class UICandidate:
    """
    AGENT-VISIBLE Candidate UI Element.

    CRITICAL GROUND-TRUTH LEAKAGE PROTECTION:
    This structure is visible to the RL agent policy.
    It MUST NOT contain any ground-truth semantic labels (e.g. data-semantic-role,
    expected_role, is_correct, ground_truth, target_role).
    Ground truth information is strictly private to the evaluator/environment.
    """
    candidate_id: int
    tag: Optional[str] = None
    element_type: Optional[str] = None
    text: Optional[str] = None
    placeholder: Optional[str] = None
    aria_label: Optional[str] = None
    role: Optional[str] = None

    visible: bool = True
    enabled: bool = True

    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None

    attributes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Production code guard: Automatically sanitize attributes upon instantiation."""
        self.attributes = sanitize_attributes(self.attributes)


@dataclass
class PrivateEvaluatorMetadata:
    """
    EVALUATOR ONLY — PRIVATE GROUND TRUTH.

    This metadata links a candidate_id to its ground-truth data-semantic-role.
    This structure is strictly PRIVATE to the Environment Evaluator for reward
    and step success validation. It is NEVER placed in observations or agent-visible info.
    """
    candidate_id: int
    semantic_role: Optional[str] = None
