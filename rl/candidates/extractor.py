"""
Phase 5 — Real DOM Candidate Extraction Pipeline
=================================================

Converts the current live browser page into a bounded collection of
agent-visible UICandidate objects.

Pipeline:
    Real browser page
          ↓
    DOM inspection (via Playwright page.evaluate — arrow-function API)
          ↓
    Filter: interactive element tags only
          ↓
    Filter: unusable elements (hidden, disabled, zero-area)
          ↓
    Extract safe observable features (attribute allowlist enforced)
          ↓
    Deterministic permutation (respects max_candidates + env reset seed)
          ↓
    List[UICandidate]  →  later Phase 6 state encoder

This module does NOT determine which element is correct.
It only defines the action choices available to the RL policy.

PLAYWRIGHT page.evaluate() API CONTRACT
----------------------------------------
Playwright's Python sync API is:

    page.evaluate(expression, arg=None)

where ``expression`` is a JavaScript *function string* of the form::

    "(params) => { ... return result; }"

Playwright calls it as ``fn(arg)``, where ``arg`` is passed as the
first (and only) parameter.  Do NOT use ``arguments[0]`` — arrow
functions do not bind the ``arguments`` object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rl.env.types import (
    PROHIBITED_ATTRIBUTE_KEYS,
    PrivateEvaluatorMetadata,
    UICandidate,
    sanitize_attributes,
)

logger = logging.getLogger(__name__)


# ─── Custom Exception ─────────────────────────────────────────────────────────

class DOMExtractionError(RuntimeError):
    """
    Raised when the browser-side JavaScript fails during candidate extraction.

    A true empty page and a broken extractor are NOT the same experimental
    result.  Callers must NOT catch this exception silently.
    """


# ─── Constants ────────────────────────────────────────────────────────────────

# Interactive element CSS selector — Phase 5 targets exactly these tags/roles.
# 'select' and 'textarea' included for future workflows; 'a[href]' for nav links.
INTERACTIVE_SELECTOR: str = (
    "input:not([type='hidden']), button, select, textarea, a[href], "
    "[role='button'], [role='link'], [role='checkbox'], [role='radio'], "
    "[role='combobox'], [role='listbox'], [role='option'], [role='menuitem'], "
    "[role='tab']"
)

# Verification element CSS selector — visible content elements for verification steps.
VERIFICATION_SELECTOR: str = (
    "div, p, span, h1, h2, h3, h4, h5, h6, main, section, "
    "[role='alert'], [role='status']"
)

# Strict attribute allowlist — ONLY these attributes may be extracted.
# Any key NOT in this set is silently dropped at extraction time,
# providing a stronger first-line defense against ground-truth leakage.
ALLOWED_ATTRIBUTE_KEYS: frozenset = frozenset({
    "id",
    "class",
    "name",
    "type",
    "placeholder",
    "title",
    "href",
    "autocomplete",
    "aria-label",
    "aria-labelledby",
    "role",
})

# Minimum visible bounding box area (px²) to consider an element usable.
MIN_ELEMENT_AREA_PX: float = 1.0

# Maximum text length kept for UICandidate.text (prevents embedding explosion).
MAX_TEXT_LENGTH: int = 120

# Absolute maximum candidates fetched from DOM before permutation/truncation.
DOM_FETCH_LIMIT: int = 200

# ---------------------------------------------------------------------------
# Playwright-compatible arrow-function expression for DOM extraction.
#
# Playwright calls page.evaluate(expression, arg) as:  fn(arg)
# The expression MUST be an arrow-function string.
# NEVER use arguments[0] — arrow functions do not bind `arguments`.
# ---------------------------------------------------------------------------
_DOM_EXTRACTION_JS = (
    # Playwright calls this function as fn(params) where params = js_arg dict.
    "(params) => {"
    "  const ALLOWED = new Set(params.allowedKeys);"
    "  const SEL = params.selector;"
    "  const LIMIT = params.fetchLimit;"
    "  const results = [];"
    "  const seen = new WeakSet();"
    "  const nodes = document.querySelectorAll(SEL);"
    "  for (let i = 0; i < nodes.length && results.length < LIMIT; i++) {"
    "    const el = nodes[i];"
    "    if (seen.has(el)) continue;"
    "    seen.add(el);"
    "    const rect = el.getBoundingClientRect();"
    # Collect ONLY allowed attributes. data-* keys are not in ALLOWED so they
    # are silently skipped here — belt-and-suspenders against leakage.
    "    const attrs = {};"
    "    for (let a = 0; a < el.attributes.length; a++) {"
    "      const attr = el.attributes[a];"
    "      const k = attr.name.toLowerCase();"
    "      if (ALLOWED.has(k)) { attrs[k] = attr.value; }"
    "    }"
    "    const txt = (el.innerText || el.textContent || '').trim().slice(0, 200);"
    "    const st = window.getComputedStyle(el);"
    "    const hidden = ("
    "      st.display === 'none' ||"
    "      st.visibility === 'hidden' ||"
    "      st.opacity === '0' ||"
    "      (rect.width === 0 && rect.height === 0)"
    "    );"
    "    const disabled = (el.disabled === true || el.getAttribute('aria-disabled') === 'true');"
    "    results.push({"
    "      tag: el.tagName.toLowerCase(),"
    "      type: el.type || null,"
    "      text: txt,"
    "      placeholder: el.placeholder || null,"
    "      ariaLabel: el.getAttribute('aria-label') || null,"
    "      role: el.getAttribute('role') || null,"
    "      visible: !hidden,"
    "      enabled: !disabled,"
    "      x: rect.left,"
    "      y: rect.top,"
    "      width: rect.width,"
    "      height: rect.height,"
    "      attrs: attrs"
    "    });"
    "  }"
    "  return results;"
    "}"
)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CandidateExtractionConfig:
    """
    Configuration for RealDOMCandidateExtractor.

    max_candidates: Upper bound on candidates returned to the agent.
    min_area_px: Minimum bounding-box area (px²) to keep an element.
    max_text_length: Maximum text characters kept per candidate.
    require_visible: If True, zero-area or hidden elements are filtered.
    require_enabled: If True, disabled elements are filtered.
    """
    max_candidates: int = 20
    min_area_px: float = MIN_ELEMENT_AREA_PX
    max_text_length: int = MAX_TEXT_LENGTH
    require_visible: bool = True
    require_enabled: bool = True


# ─── Extractor ───────────────────────────────────────────────────────────────

class RealDOMCandidateExtractor:
    """
    Extracts a bounded, deterministically-ordered list of UICandidate objects
    from the current live browser page using Playwright's page.evaluate().

    Usage::

        extractor = RealDOMCandidateExtractor(config, rng=np.random.default_rng(seed))
        candidates, private_meta = extractor.extract(page, expected_role="username-input")

    The extractor is stateless between calls except for the RNG, which is
    advanced once per extract() call (for deterministic candidate permutation).

    Raises
    ------
    DOMExtractionError
        If the browser-side JavaScript throws.  This is a hard failure —
        a broken extractor must NOT be treated as zero candidates.
    """

    def __init__(
        self,
        config: Optional[CandidateExtractionConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or CandidateExtractionConfig()
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def set_rng(self, rng: np.random.Generator) -> None:
        """Replace the RNG (called by the environment on each reset())."""
        self.rng = rng

    # ── Public API ──────────────────────────────────────────────────────────

    def extract(
        self,
        page: Any,  # playwright.sync_api.Page
        expected_role: Optional[str] = None,
        mode: str = "interactive",
    ) -> Tuple[List[UICandidate], List[PrivateEvaluatorMetadata]]:
        """
        Extract candidates from the current page.

        Parameters
        ----------
        page : playwright.sync_api.Page
            Live Playwright page object.
        expected_role : str, optional
            Ground-truth semantic role for the current workflow step.
            Used ONLY to populate PrivateEvaluatorMetadata.
            NEVER stored in UICandidate.
        mode : str, optional
            "interactive" (default, for fill/click) or "verification" (for verify steps).

        Returns
        -------
        candidates : List[UICandidate]
            Agent-visible candidates. No ground-truth information.
        private_meta : List[PrivateEvaluatorMetadata]
            Evaluator-private ground truth, parallel to candidates.

        Raises
        ------
        DOMExtractionError
            If the browser-side JavaScript evaluation fails.
        """
        selector = (
            VERIFICATION_SELECTOR if mode == "verification" else INTERACTIVE_SELECTOR
        )
        raw_elements = self._fetch_dom_elements(page, selector=selector)
        filtered = self._filter_elements(raw_elements, mode=mode)
        raw_candidates = self._build_raw_candidates(filtered)

        # Deterministic permutation using env-reset RNG
        candidates, private_meta = self._permute_and_truncate(
            raw_candidates, expected_role
        )

        logger.debug(
            "extract(): url=%s mode=%s role=%s dom_fetched=%d filtered=%d returned=%d",
            getattr(page, "url", "?"),
            mode,
            expected_role,
            len(raw_elements),
            len(filtered),
            len(candidates),
        )
        return candidates, private_meta

    # ── DOM Fetching ────────────────────────────────────────────────────────

    def _fetch_dom_elements(
        self,
        page: Any,
        selector: str = INTERACTIVE_SELECTOR,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate JavaScript in the live page to collect raw element descriptors.

        Uses the module-level ``_DOM_EXTRACTION_JS`` arrow-function expression,
        which is the ONLY correct form for Playwright's page.evaluate(expr, arg)
        API.  See module docstring for the API contract.

        Raises
        ------
        DOMExtractionError
            Wraps the original Playwright exception so callers cannot silently
            interpret a broken extractor as empty candidates.
        """
        js_arg = {
            "allowedKeys": list(ALLOWED_ATTRIBUTE_KEYS),
            "selector": selector,
            "fetchLimit": DOM_FETCH_LIMIT,
        }

        try:
            raw = page.evaluate(_DOM_EXTRACTION_JS, js_arg)
        except Exception as exc:
            raise DOMExtractionError(
                f"DOM extraction failed on page '{getattr(page, 'url', '?')}': {exc}"
            ) from exc

        return raw if isinstance(raw, list) else []

    # ── Filtering ───────────────────────────────────────────────────────────

    def _filter_elements(
        self,
        elements: List[Dict[str, Any]],
        mode: str = "interactive",
    ) -> List[Dict[str, Any]]:
        """
        Remove elements that are unusable by the agent:
          - Not visible (if require_visible)
          - Disabled (if require_enabled)
          - Zero-area bounding box (below min_area_px threshold)
          - Empty text (if mode == "verification")
        """
        kept = []
        for el in elements:
            if self.config.require_visible and not el.get("visible", True):
                continue
            if self.config.require_enabled and not el.get("enabled", True):
                continue
            w = el.get("width", 0.0) or 0.0
            h = el.get("height", 0.0) or 0.0
            if self.config.require_visible and (w * h) < self.config.min_area_px:
                continue
            if mode == "verification":
                txt = (el.get("text") or "").strip()
                if not txt:
                    continue
            kept.append(el)
        return kept

    # ── Candidate Building ───────────────────────────────────────────────────

    def _build_raw_candidates(self, elements: List[Dict[str, Any]]) -> List[UICandidate]:
        """
        Convert filtered raw DOM descriptors into UICandidate objects.

        LEAKAGE PREVENTION (belt-and-suspenders):
          1. The JS payload never contains data-semantic-role (ALLOWED set guard).
          2. _apply_allowlist() strips any non-allowed attribute.
          3. UICandidate.__post_init__() calls sanitize_attributes() which
             drops any remaining PROHIBITED_ATTRIBUTE_KEYS.
        """
        candidates = []
        for i, el in enumerate(elements):
            safe_attrs = self._apply_allowlist(el.get("attrs", {}))

            text_raw = el.get("text") or ""
            text = text_raw[: self.config.max_text_length] if text_raw else None
            if not text:
                text = None

            candidate = UICandidate(
                candidate_id=i,           # Temporary index; re-assigned after permutation
                tag=el.get("tag"),
                element_type=el.get("type"),
                text=text,
                placeholder=el.get("placeholder") or None,
                aria_label=el.get("ariaLabel") or None,
                role=el.get("role") or None,
                visible=bool(el.get("visible", True)),
                enabled=bool(el.get("enabled", True)),
                x=float(el.get("x") or 0.0),
                y=float(el.get("y") or 0.0),
                width=float(el.get("width") or 0.0),
                height=float(el.get("height") or 0.0),
                attributes=safe_attrs,
            )
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _apply_allowlist(attrs: Dict[str, Any]) -> Dict[str, str]:
        """
        Phase 5 first-line extraction-time allowlist guard.

        Drops any attribute key not in ALLOWED_ATTRIBUTE_KEYS.
        Also drops any key matching PROHIBITED_ATTRIBUTE_KEYS patterns.
        (UICandidate.__post_init__ provides the second layer via sanitize_attributes.)
        """
        clean: Dict[str, str] = {}
        for key, val in (attrs or {}).items():
            k_lower = str(key).lower().strip()
            if k_lower not in ALLOWED_ATTRIBUTE_KEYS:
                continue
            k_norm = k_lower.replace("-", "_")
            if k_norm in PROHIBITED_ATTRIBUTE_KEYS or "semantic_role" in k_norm:
                continue
            clean[k_lower] = str(val)
        return clean

    # ── Permutation & Truncation ──────────────────────────────────────────────

    def _permute_and_truncate(
        self,
        candidates: List[UICandidate],
        expected_role: Optional[str],
    ) -> Tuple[List[UICandidate], List[PrivateEvaluatorMetadata]]:
        """
        Deterministically permute candidates using the environment RNG,
        then truncate to max_candidates.

        candidate_id is re-assigned post-permutation so it reflects
        the agent's index into the observation rather than DOM position.

        Private metadata (ground truth semantic role) is built in parallel.
        The ``expected_role`` is stored ONLY in PrivateEvaluatorMetadata,
        never in UICandidate.
        """
        n = len(candidates)
        if n == 0:
            return [], []

        if n > 1:
            perm = self.rng.permutation(n)
        else:
            perm = np.array([0])

        permuted = [candidates[idx] for idx in perm]
        truncated = permuted[: self.config.max_candidates]

        final_candidates: List[UICandidate] = []
        private_meta: List[PrivateEvaluatorMetadata] = []

        for new_id, cand in enumerate(truncated):
            cand.candidate_id = new_id
            private_meta.append(
                PrivateEvaluatorMetadata(candidate_id=new_id, semantic_role=None)
            )
            final_candidates.append(cand)

        return final_candidates, private_meta
