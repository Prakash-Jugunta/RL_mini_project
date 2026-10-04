import re
from difflib import SequenceMatcher
from typing import Optional, Set
from rl.env.types import UICandidate


def normalize_text(text: Optional[str]) -> str:
    """
    Public text normalization utility.
    Converts to lowercase, converts hyphens/underscores/punctuation to spaces,
    collapses multiple spaces, and strips whitespace.
    """
    if not text:
        return ""
    text_str = str(text).lower()
    # Replace hyphens, underscores, and non-alphanumeric chars with space
    cleaned = re.sub(r'[^a-z0-9]', ' ', text_str)
    # Collapse multiple whitespaces
    collapsed = re.sub(r'\s+', ' ', cleaned).strip()
    return collapsed


def _get_tokens(norm_text: str) -> Set[str]:
    """Tokenize normalized text into a set of non-empty word tokens."""
    return set(norm_text.split()) if norm_text else set()


def calculate_text_similarity(target_intent: str, candidate_field: Optional[str]) -> float:
    """
    Calculates lexical similarity between normalized target intent and a public candidate field.
    Combines Jaccard token overlap, difflib SequenceMatcher ratio, and substring containment.
    Returns float in range [0.0, 1.0].
    """
    norm_intent = normalize_text(target_intent)
    norm_field = normalize_text(candidate_field)

    if not norm_intent or not norm_field:
        return 0.0

    # Exact normalized match
    if norm_intent == norm_field:
        return 1.0

    # Token overlap (Jaccard similarity)
    intent_tokens = _get_tokens(norm_intent)
    field_tokens = _get_tokens(norm_field)
    
    if not intent_tokens or not field_tokens:
        jaccard = 0.0
    else:
        intersection = intent_tokens.intersection(field_tokens)
        union = intent_tokens.union(field_tokens)
        jaccard = len(intersection) / len(union) if union else 0.0

    # SequenceMatcher edit distance ratio
    seq_ratio = SequenceMatcher(None, norm_intent, norm_field).ratio()

    # Substring containment bonus
    containment = 0.0
    if norm_intent in norm_field or norm_field in norm_intent:
        containment = 0.8
    else:
        # Check token containment
        if intent_tokens and intent_tokens.issubset(field_tokens):
            containment = 0.7
        elif field_tokens and field_tokens.issubset(intent_tokens):
            containment = 0.6

    # Combined score
    score = max(jaccard, seq_ratio, containment)
    return float(min(1.0, max(0.0, score)))


def calculate_structural_compatibility(action_type: str, candidate: UICandidate) -> float:
    """
    Calculates structural compatibility score between workflow action_type ("fill", "click", "verify")
    and public properties of UICandidate.
    """
    act_type = (action_type or "").lower().strip()
    tag = (candidate.tag or "").lower().strip()
    elem_type = (candidate.element_type or "").lower().strip()
    role = (candidate.role or "").lower().strip()
    attr_type = (candidate.attributes.get("type", "") or "").lower().strip()

    if act_type == "fill":
        if tag in {"input", "textarea"} or elem_type in {"text", "password", "email", "search", "number", "tel"} or attr_type in {"text", "password", "email", "search", "number", "tel"} or role in {"textbox", "searchbox"}:
            return 1.0
        return 0.0

    elif act_type == "click":
        if tag in {"button", "a"} or elem_type in {"submit", "button"} or attr_type in {"submit", "button"} or role in {"button", "link", "menuitem", "tab"}:
            return 1.0
        elif candidate.visible and candidate.enabled:
            return 0.5
        return 0.0

    elif act_type == "verify":
        if candidate.visible:
            return 1.0
        return 0.0

    return 0.5
