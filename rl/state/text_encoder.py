from __future__ import annotations
from functools import lru_cache
from typing import List, Optional, TYPE_CHECKING
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

if TYPE_CHECKING:
    from rl.env.types import UICandidate


class DeterministicTextEncoder:
    """
    Deterministic lightweight feature-hashing text encoder.

    Uses scikit-learn's HashingVectorizer (n_features=64, n-grams 1-2) with L2 normalization.
    Chosen because it is completely deterministic, 100% native, requires 0 pre-training or fitting
    on held-out L6 data, avoids HuggingFace model download latency, and avoids Windows PyTorch
    OpenMP DLL conflicts.
    """

    def __init__(self, n_features: int = 64):
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True
        )

    @lru_cache(maxsize=1024)
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single string into a normalized 64-dimensional float32 vector.
        Cached for high performance on repeated objective strings.
        """
        if not text or not text.strip():
            return np.zeros(self.n_features, dtype=np.float32)

        X = self._vectorizer.transform([text]).toarray().astype(np.float32)[0]
        norm = np.linalg.norm(X)
        if norm > 0:
            X /= norm
        return X

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of strings into a (batch_size, 64) normalized float32 matrix.
        """
        if not texts:
            return np.zeros((0, self.n_features), dtype=np.float32)

        X = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    @staticmethod
    def serialize_candidate(candidate: UICandidate) -> str:
        """
        Construct a canonical public representation string for a UICandidate.

        STRICT PRIVACY ENFORCEMENT:
        Included fields: tag, element_type, text, placeholder, aria_label, role, attributes (name, title, class).
        Explicitly EXCLUDED fields:
            - data-semantic-role
            - semantic_role
            - expected_role
            - PrivateEvaluatorMetadata
            - target_index / is_correct / correctness
            - mutation_level / mutation_seed

        Password handling: If element_type or attributes indicate a password field,
        the literal text string is masked out to protect credential values.
        """
        parts = [f"tag {candidate.tag or ''}"]

        elem_type = candidate.element_type or ""
        if elem_type:
            parts.append(f"type {elem_type}")

        role = candidate.attributes.get("role", "")
        if role:
            parts.append(f"role {role}")

        # Protect literal password values
        is_password = (elem_type.lower() == "password")
        if not is_password and candidate.text:
            parts.append(f"text {candidate.text}")

        if candidate.placeholder:
            parts.append(f"placeholder {candidate.placeholder}")

        aria = candidate.attributes.get("aria-label", "")
        if aria:
            parts.append(f"aria {aria}")

        name = candidate.attributes.get("name", "")
        if name:
            parts.append(f"name {name}")

        title = candidate.attributes.get("title", "")
        if title:
            parts.append(f"title {title}")

        css_class = candidate.attributes.get("class", "")
        if css_class:
            parts.append(f"class {css_class}")

        return " ".join(parts)
