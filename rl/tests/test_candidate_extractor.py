"""
Phase 5 — RealDOMCandidateExtractor Unit Tests
===============================================

Tests all properties of the Phase 5 extraction pipeline WITHOUT
requiring a running browser (uses a mock page object).

Test coverage:
  1.  Allowlist enforcement: non-allowed attrs are stripped at extraction time
  2.  Prohibited key guard: data-semantic-role never appears in UICandidate
  3.  Filter: hidden elements are dropped
  4.  Filter: disabled elements are dropped
  5.  Filter: zero-area elements are dropped
  6.  max_candidates truncation is respected
  7.  candidate_id is re-assigned as 0-based position after permutation
  8.  Deterministic permutation: same seed → same ordering
  9.  Different seeds → different orderings (probabilistic check)
  10. PrivateEvaluatorMetadata has no ground truth in UICandidate attributes
  11. Text is truncated to max_text_length
  12. Empty page returns empty candidate list
  13. _apply_allowlist drops data-* attributes
  14. UICandidate.attributes passes through sanitize_attributes (second layer)
"""

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np

from rl.candidates.extractor import (
    ALLOWED_ATTRIBUTE_KEYS,
    CandidateExtractionConfig,
    RealDOMCandidateExtractor,
)
from rl.env.types import PROHIBITED_ATTRIBUTE_KEYS, UICandidate, sanitize_attributes


# ── Mock Page Factory ─────────────────────────────────────────────────────────

def _make_page(raw_elements: List[Dict[str, Any]]) -> MagicMock:
    """
    Build a mock Playwright page that returns ``raw_elements`` from page.evaluate().
    The JS code evaluation is fully bypassed — we inject raw elements directly.
    """
    page = MagicMock()
    page.url = "http://localhost:3000/login"
    page.evaluate.return_value = raw_elements
    return page


def _make_element(
    tag: str = "input",
    el_type: str = "text",
    text: str = "Click me",
    placeholder: str = "Enter value",
    aria_label: str = "Field",
    role: Optional[str] = None,
    visible: bool = True,
    enabled: bool = True,
    x: float = 100.0,
    y: float = 200.0,
    width: float = 150.0,
    height: float = 40.0,
    attrs: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return {
        "tag": tag,
        "type": el_type,
        "text": text,
        "placeholder": placeholder,
        "ariaLabel": aria_label,
        "role": role,
        "visible": visible,
        "enabled": enabled,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "attrs": attrs or {"id": "field-1", "class": "form-input", "type": "text"},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCandidateExtractionConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = CandidateExtractionConfig()
        self.assertEqual(cfg.max_candidates, 20)
        self.assertTrue(cfg.require_visible)
        self.assertTrue(cfg.require_enabled)


class TestAllowlistEnforcement(unittest.TestCase):
    """Test 1-2: Attribute allowlist and prohibited key guard."""

    def setUp(self):
        self.extractor = RealDOMCandidateExtractor(
            config=CandidateExtractionConfig(max_candidates=10),
            rng=np.random.default_rng(42),
        )

    def test_non_allowed_attrs_stripped(self):
        """Non-allowlisted attributes must not appear in UICandidate.attributes."""
        el = _make_element(attrs={
            "id": "usr",
            "class": "form-input",
            "data-custom": "should-be-dropped",
            "onclick": "should-be-dropped",
            "style": "should-be-dropped",
        })
        page = _make_page([el])
        candidates, _ = self.extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        attrs = candidates[0].attributes
        self.assertIn("id", attrs)
        self.assertIn("class", attrs)
        self.assertNotIn("data-custom", attrs)
        self.assertNotIn("onclick", attrs)
        self.assertNotIn("style", attrs)

    def test_data_semantic_role_never_in_candidate(self):
        """data-semantic-role must NEVER appear in UICandidate attributes — Test 2."""
        el = _make_element(attrs={
            "id": "usr",
            "data-semantic-role": "username-input",  # Must be stripped
            "class": "form-input",
        })
        page = _make_page([el])
        candidates, _ = self.extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        attrs = candidates[0].attributes
        self.assertNotIn("data-semantic-role", attrs)
        self.assertNotIn("data_semantic_role", attrs)
        self.assertNotIn("semantic_role", attrs)

    def test_all_allowed_keys_survive(self):
        """All keys in ALLOWED_ATTRIBUTE_KEYS must pass through if present."""
        all_allowed = {k: f"val-{k}" for k in ALLOWED_ATTRIBUTE_KEYS}
        el = _make_element(attrs=all_allowed)
        page = _make_page([el])
        candidates, _ = self.extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        for key in ALLOWED_ATTRIBUTE_KEYS:
            self.assertIn(key, candidates[0].attributes, f"Expected allowed key '{key}' to survive.")


class TestFilteringBehavior(unittest.TestCase):
    """Tests 3-5: Hidden, disabled, and zero-area element filtering."""

    def setUp(self):
        self.config = CandidateExtractionConfig(max_candidates=10)
        self.rng = np.random.default_rng(42)

    def test_hidden_elements_filtered(self):
        """Test 3: Hidden elements (visible=False) must be dropped."""
        extractor = RealDOMCandidateExtractor(config=self.config, rng=np.random.default_rng(42))
        visible_el = _make_element(visible=True)
        hidden_el = _make_element(tag="button", text="Hidden", visible=False)
        page = _make_page([visible_el, hidden_el])
        candidates, _ = extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].visible)

    def test_disabled_elements_filtered(self):
        """Test 4: Disabled elements must be dropped when require_enabled=True."""
        extractor = RealDOMCandidateExtractor(config=self.config, rng=np.random.default_rng(42))
        enabled_el = _make_element(enabled=True)
        disabled_el = _make_element(tag="button", text="Disabled Btn", enabled=False)
        page = _make_page([enabled_el, disabled_el])
        candidates, _ = extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].enabled)

    def test_zero_area_elements_filtered(self):
        """Test 5: Elements with width=0, height=0 must be filtered out."""
        extractor = RealDOMCandidateExtractor(config=self.config, rng=np.random.default_rng(42))
        good_el = _make_element(width=100.0, height=40.0)
        zero_el = _make_element(tag="button", text="Zero Area", width=0.0, height=0.0)
        page = _make_page([good_el, zero_el])
        candidates, _ = extractor.extract(page)
        self.assertEqual(len(candidates), 1)


class TestMaxCandidatesTruncation(unittest.TestCase):
    """Test 6: max_candidates truncation."""

    def test_truncation_respected(self):
        config = CandidateExtractionConfig(max_candidates=5)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(42))
        elements = [_make_element(tag="button", text=f"Btn {i}", x=float(i * 10)) for i in range(20)]
        page = _make_page(elements)
        candidates, meta = extractor.extract(page)
        self.assertLessEqual(len(candidates), 5)
        self.assertEqual(len(candidates), len(meta))


class TestCandidateIdReassignment(unittest.TestCase):
    """Test 7: candidate_id must be 0-based position index after permutation."""

    def test_candidate_ids_are_sequential(self):
        config = CandidateExtractionConfig(max_candidates=10)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(7))
        elements = [_make_element(tag="button", text=f"El {i}", x=float(i * 5)) for i in range(5)]
        page = _make_page(elements)
        candidates, _ = extractor.extract(page)
        ids = [c.candidate_id for c in candidates]
        self.assertEqual(sorted(ids), list(range(len(candidates))))
        # IDs must equal their index position
        for i, c in enumerate(candidates):
            self.assertEqual(c.candidate_id, i)


class TestDeterministicPermutation(unittest.TestCase):
    """Tests 8-9: Permutation determinism and seed-sensitivity."""

    def _extract_order(self, seed: int) -> List[str]:
        config = CandidateExtractionConfig(max_candidates=10)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(seed))
        elements = [_make_element(tag="button", text=f"El{i}", x=float(i * 5)) for i in range(8)]
        page = _make_page(elements)
        candidates, _ = extractor.extract(page)
        return [c.text for c in candidates]

    def test_same_seed_same_order(self):
        """Test 8: Same seed must produce identical ordering."""
        order_a = self._extract_order(seed=123)
        order_b = self._extract_order(seed=123)
        self.assertEqual(order_a, order_b)

    def test_different_seeds_different_orders(self):
        """Test 9: Different seeds should produce different orderings (probabilistic)."""
        order_a = self._extract_order(seed=1)
        order_b = self._extract_order(seed=9999)
        # With 8 elements, the probability of an identical permutation is 1/8! ≈ 0.00025
        self.assertNotEqual(order_a, order_b)


class TestPrivateMetadataLeakage(unittest.TestCase):
    """Test 10: PrivateEvaluatorMetadata content never appears in UICandidate attributes."""

    def test_private_meta_not_in_candidate(self):
        config = CandidateExtractionConfig(max_candidates=10)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(42))
        el = _make_element()
        page = _make_page([el])
        candidates, private_meta = extractor.extract(page, expected_role="username-input")
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        # semantic_role must not appear in UICandidate.attributes
        for key in cand.attributes:
            self.assertNotIn("semantic_role", key.lower())
            self.assertNotIn("ground_truth", key.lower())
        # Private meta has no info-leak into agent structure
        self.assertEqual(len(private_meta), 1)
        self.assertIsInstance(private_meta[0].candidate_id, int)


class TestTextTruncation(unittest.TestCase):
    """Test 11: Text is truncated to max_text_length."""

    def test_long_text_truncated(self):
        max_len = 30
        config = CandidateExtractionConfig(max_candidates=10, max_text_length=max_len)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(42))
        long_text = "A" * 200
        el = _make_element(text=long_text)
        page = _make_page([el])
        candidates, _ = extractor.extract(page)
        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(len(candidates[0].text or ""), max_len)


class TestEmptyPage(unittest.TestCase):
    """Test 12: Empty page returns empty candidate list."""

    def test_empty_page(self):
        config = CandidateExtractionConfig(max_candidates=10)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(42))
        page = _make_page([])
        candidates, meta = extractor.extract(page)
        self.assertEqual(candidates, [])
        self.assertEqual(meta, [])


class TestApplyAllowlist(unittest.TestCase):
    """Tests 13-14: _apply_allowlist static method."""

    def test_drops_data_star_attributes(self):
        """Test 13: _apply_allowlist must drop all data-* attributes."""
        dirty = {
            "id": "usr",
            "data-semantic-role": "username-input",
            "data-testid": "login-field",
            "class": "form-input",
        }
        clean = RealDOMCandidateExtractor._apply_allowlist(dirty)
        self.assertIn("id", clean)
        self.assertIn("class", clean)
        self.assertNotIn("data-semantic-role", clean)
        self.assertNotIn("data-testid", clean)

    def test_sanitize_attributes_second_layer(self):
        """Test 14: sanitize_attributes (second layer) also removes prohibited keys."""
        dirty = {
            "id": "x",
            "semantic_role": "should-be-dropped",
            "ground_truth": "should-be-dropped",
            "is_correct": "true",
        }
        clean = sanitize_attributes(dirty)
        self.assertIn("id", clean)
        self.assertNotIn("semantic_role", clean)
        self.assertNotIn("ground_truth", clean)
        self.assertNotIn("is_correct", clean)

    def test_ui_candidate_post_init_sanitizes(self):
        """UICandidate.__post_init__ must silently sanitize prohibited keys."""
        cand = UICandidate(
            candidate_id=0,
            tag="input",
            attributes={
                "id": "usr",
                "data-semantic-role": "username-input",  # Must be stripped
                "class": "form-input",
            }
        )
        self.assertNotIn("data-semantic-role", cand.attributes)
        self.assertIn("id", cand.attributes)
        self.assertIn("class", cand.attributes)


class TestSetRng(unittest.TestCase):
    """set_rng() correctly replaces the extractor's RNG."""

    def test_set_rng_changes_permutation(self):
        config = CandidateExtractionConfig(max_candidates=10)
        extractor = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(1))
        elements = [_make_element(tag="button", text=f"El{i}", x=float(i * 5)) for i in range(8)]
        page = _make_page(elements)

        candidates_before, _ = extractor.extract(page)
        order_before = [c.text for c in candidates_before]

        extractor.set_rng(np.random.default_rng(9999))
        page2 = _make_page(elements)
        candidates_after, _ = extractor.extract(page2)
        order_after = [c.text for c in candidates_after]

        self.assertNotEqual(order_before, order_after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
