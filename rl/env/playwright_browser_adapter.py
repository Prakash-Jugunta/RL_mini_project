"""
Phase 5 — PlaywrightBrowserAdapter
====================================

Real browser adapter implementing the BrowserAdapter interface using
Playwright (synchronous API). This replaces the MockBrowserAdapter for
integration testing and real training runs.

Key design constraints
----------------------
* Seed separation is preserved:
    - env reset seed  → controls candidate permutation (via RealDOMCandidateExtractor.rng)
    - mutation_level + mutation_seed → controls DOM mutations injected into the app
      via the Phase 3 /api/set-mutation endpoint (NOT query parameters).
* Ground-truth leakage is prevented at two layers:
    1. RealDOMCandidateExtractor enforces the attribute allowlist during extraction.
    2. validate_step() reads ``data-semantic-role`` from the DOM ONLY to evaluate
       the last action privately — this value is never stored in UICandidate.
* Browser lifecycle: a single Playwright browser/context/page is kept alive across
  episodes and reset between them. Call close() to tear down at the end.
* No pre-flight login: the frozen application does not require authentication for
  /products, /profile, or /products/:id routes.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    BrowserContext = Any
    Page = Any
    Playwright = Any

from rl.env.browser_adapter import BrowserAdapter
from rl.env.types import UICandidate, PrivateEvaluatorMetadata
from rl.candidates.extractor import RealDOMCandidateExtractor, CandidateExtractionConfig

logger = logging.getLogger(__name__)

# Base URL of the running Vite/React development server.
DEFAULT_BASE_URL: str = "http://localhost:3000"

# Time to wait (seconds) after navigation before extracting candidates.
# Vite dev server + React render is near-instant, but we allow a small grace period.
NAVIGATION_WAIT_SECONDS: float = 0.6

# Time to wait (seconds) after an action (fill/click) before validation.
ACTION_SETTLE_SECONDS: float = 0.4

# Selector used by validate_step() to find the element that was last acted on.
# Reads data-semantic-role from the DOM — EVALUATOR ONLY, never in UICandidate.
SEMANTIC_ROLE_ATTR: str = "data-semantic-role"


class PlaywrightBrowserAdapter(BrowserAdapter):
    """
    Real browser interaction via Playwright (synchronous API).

    Implements BrowserAdapter to drop in as a direct replacement for
    MockBrowserAdapter in UIRecoveryEnv.

    Parameters
    ----------
    base_url : str
        Base URL of the running application (e.g. ``http://localhost:3000``).
    max_candidates : int
        Maximum number of candidates returned per step.
    headless : bool
        Whether to run Chromium in headless mode.
    slow_mo : float
        Playwright slow_mo milliseconds (useful for visual debugging).

    Seed separation
    ---------------
    ``reset(start_path, level, seed)`` stores the mutation configuration and
    constructs the URL with query params ``?mutation_level=<level>&mutation_seed=<seed>``.
    The environment RNG (for candidate permutation) is injected via ``set_rng()``.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        max_candidates: int = 20,
        headless: bool = True,
        slow_mo: float = 0.0,
    ) -> None:
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            )

        self.base_url = base_url.rstrip("/")
        self.max_candidates = max_candidates
        self.headless = headless
        self.slow_mo = slow_mo

        # Playwright objects (initialised lazily in _ensure_browser)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        # Episode state
        self.current_path: str = "/"
        self.current_level: int = 0
        self.current_seed: int = 42
        self._last_executed_candidate: Optional[UICandidate] = None

        # Candidate extractor (Phase 5 pipeline)
        config = CandidateExtractionConfig(max_candidates=self.max_candidates)
        self._extractor = RealDOMCandidateExtractor(config=config)

        # Last extracted candidates (needed by validate_step)
        self._last_candidates: List[UICandidate] = []
        self._last_private_meta: List[PrivateEvaluatorMetadata] = []
        self._last_action_candidate_id: Optional[int] = None
        self.last_timing_diagnostics: Dict[str, float] = {
            "execute_action_sec": 0.0,
            "validate_step_sec": 0.0,
            "get_candidates_sec": 0.0,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _ensure_browser(self) -> None:
        """Lazily initialise Playwright + Chromium if not yet started."""
        if self._browser is not None:
            return
        logger.info("Launching Playwright Chromium (headless=%s)", self.headless)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
        )
        self._page = self._context.new_page()

    def close(self) -> None:
        """Tear down browser resources cleanly."""
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            logger.warning("Error during browser cleanup: %s", exc)
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None

    # ── BrowserAdapter Interface ───────────────────────────────────────────

    def set_rng(self, rng: np.random.Generator) -> None:
        """
        Inject the environment reset RNG into the extractor.

        This controls candidate PERMUTATION ONLY — distinct from the
        mutation_level/seed which controls DOM mutations.
        """
        self._extractor.set_rng(rng)

    def reset(self, start_path: str, level: int, seed: int) -> None:
        """
        Reset browser to the initial page for a new episode.

        Mutation activation follows the Phase 3 mechanism exactly:
            GET /api/set-mutation?level=X&seed=Y  (verify response)
            page.goto(<base_url><start_path>)      (clean URL — no query params)

        No pre-flight login is performed. The frozen application does not
        require authentication for /products, /profile, or /products/:id.
        """
        self._ensure_browser()
        self.current_path = start_path
        self.current_level = level
        self.current_seed = seed
        self._last_executed_candidate = None
        self._last_action_candidate_id = None
        self._last_candidates = []
        self._last_private_meta = []

        self._navigate_to(start_path, level, seed)

    # Action type → extraction mode mapping.
    # MUST be derived from action_type, never from expected_role.
    _ACTION_TO_MODE: Dict[str, str] = {
        "fill":   "interactive",
        "click":  "interactive",
        "verify": "verification",
    }

    def get_candidates(
        self,
        step_id: str,
        expected_role: Optional[str],
        action_type: str = "click",
    ) -> Tuple[List[UICandidate], List[PrivateEvaluatorMetadata]]:
        """
        Extract candidates from the current page using the Phase 5 pipeline.

        The extraction mode is derived SOLELY from ``action_type``:
            fill   -> interactive  (inputs, buttons, links, select)
            click  -> interactive
            verify -> verification (headings, alerts, text elements)

        Parameters
        ----------
        step_id : str
            Workflow step identifier (logging only; not used for mode selection).
        expected_role : Optional[str]
            Evaluator ground truth — used only for private metadata enrichment.
            NEVER used to choose extraction mode.
        action_type : str
            One of "fill", "click", "verify".

        Returns
        -------
        candidates : List[UICandidate]
            Agent-visible candidates — NO ground truth.
        private_meta : List[PrivateEvaluatorMetadata]
            Evaluator-private metadata enriched with semantic_role for the
            matching DOM element (never exposed to the agent).
        """
        t0 = time.time()
        mode = self._ACTION_TO_MODE.get(action_type, "interactive")
        logger.debug(
            "get_candidates: step_id=%s action_type=%s -> mode=%s expected_role=%s",
            step_id, action_type, mode, expected_role,
        )

        page = self._get_page()
        candidates, private_meta = self._extractor.extract(
            page, expected_role=expected_role, mode=mode
        )

        # Enrich private metadata: scan DOM for the element whose
        # data-semantic-role matches expected_role, then mark that candidate.
        # EVALUATOR ONLY — never exposed to the agent.
        if expected_role:
            enriched_meta = self._enrich_private_metadata(
                page, candidates, private_meta, expected_role
            )
            private_meta = enriched_meta

        self._last_candidates = candidates
        self._last_private_meta = private_meta
        self.last_timing_diagnostics["get_candidates_sec"] = time.time() - t0
        return candidates, private_meta

    def execute_action(
        self,
        candidate_id: int,
        action_type: str,
        value: Optional[str] = None,
    ) -> bool:
        """
        Execute a fill or click action on the candidate identified by ``candidate_id``.

        candidate_id is the agent-visible position index (0-based) in the
        last returned candidates list.

        Returns True if the action was dispatched without error.
        """
        t0 = time.time()
        page = self._get_page()

        # Look up the candidate from the last extraction
        candidate = self._find_candidate(candidate_id)
        if candidate is None:
            logger.warning("execute_action: candidate_id=%d not found in last candidates", candidate_id)
            self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
            return False

        self._last_action_candidate_id = candidate_id
        self._last_executed_candidate = candidate

        if action_type == "verify":
            # verify steps are evaluated by validate_step against private metadata
            self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
            return True

        # Locate the element in the live DOM using best-available identity
        locator = self._build_locator(page, candidate)
        if locator is None:
            logger.warning("execute_action: could not build locator for candidate_id=%d", candidate_id)
            self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
            return False

        try:
            if action_type == "fill":
                locator.first.fill(value or "", timeout=3000)
                time.sleep(ACTION_SETTLE_SECONDS)
            elif action_type == "click":
                locator.first.click(timeout=3000)
                time.sleep(ACTION_SETTLE_SECONDS)
            else:
                logger.warning("execute_action: unknown action_type='%s'", action_type)
                self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
                return False
        except Exception as exc:
            logger.error("execute_action failed (candidate_id=%d action=%s): %s", candidate_id, action_type, exc)
            self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
            return False

        self.last_timing_diagnostics["execute_action_sec"] = time.time() - t0
        return True

    def highlight_candidate(
        self,
        candidate_id: int,
        label: str = "DQN SELECTED",
        duration_sec: float = 1.0,
    ) -> bool:
        """
        Visually outline the selected element in the real browser.

        Style: 3px solid red, box-shadow, temporary label badge "DQN SELECTED".
        Keeps visible for duration_sec (default 1.0s) and restores original styles.
        Does NOT alter observation, candidate extraction semantics, reward, or
        evaluator validation.
        """
        if self._page is None:
            return False

        candidate = self._find_candidate(candidate_id)
        if candidate is None:
            return False

        locator = self._build_locator(self._page, candidate)
        if locator is None:
            return False

        try:
            el_handle = locator.first.element_handle(timeout=1000)
            if el_handle is None:
                return False

            self._page.evaluate(
                """({ el, labelText, durationMs }) => {
                    if (!el) return;
                    const origOutline = el.style.outline;
                    const origBoxShadow = el.style.boxShadow;

                    const badge = document.createElement('div');
                    badge.id = '__dqn_demo_highlight_badge__';
                    badge.innerText = labelText;
                    badge.style.position = 'absolute';
                    badge.style.backgroundColor = '#dc2626';
                    badge.style.color = '#ffffff';
                    badge.style.padding = '2px 8px';
                    badge.style.borderRadius = '4px';
                    badge.style.fontSize = '12px';
                    badge.style.fontWeight = 'bold';
                    badge.style.zIndex = '999999';
                    badge.style.pointerEvents = 'none';
                    badge.style.boxShadow = '0 2px 8px rgba(0,0,0,0.3)';

                    const rect = el.getBoundingClientRect();
                    badge.style.top = Math.max(0, rect.top + window.scrollY - 24) + 'px';
                    badge.style.left = (rect.left + window.scrollX) + 'px';
                    document.body.appendChild(badge);

                    el.style.outline = '3px solid red';
                    el.style.boxShadow = '0 0 12px rgba(220, 38, 38, 0.8)';

                    setTimeout(() => {
                        try {
                            el.style.outline = origOutline;
                            el.style.boxShadow = origBoxShadow;
                            if (badge.parentNode) {
                                badge.parentNode.removeChild(badge);
                            }
                        } catch (e) {}
                    }, durationMs);
                }""",
                {
                    "el": el_handle,
                    "labelText": label,
                    "durationMs": int(duration_sec * 1000),
                },
            )
            time.sleep(duration_sec)
            return True
        except Exception as exc:
            logger.debug("highlight_candidate failed: %s", exc)
            return False

    def validate_step(self, expected_role: Optional[str]) -> bool:
        """
        Privately validate whether the last action succeeded for this step.

        Uses evaluator-private metadata captured during extraction first (0ms latency).
        Only falls back to live DOM validation if metadata is genuinely unavailable.
        """
        t0 = time.time()
        if self._last_action_candidate_id is None:
            self.last_timing_diagnostics["validate_step_sec"] = time.time() - t0
            return False

        # 1. Fast path: check evaluator-private metadata first
        meta_result = self._validate_from_metadata(expected_role)
        if meta_result is not None:
            elapsed = time.time() - t0
            self.last_timing_diagnostics["validate_step_sec"] = elapsed
            logger.debug(
                "validate_step (metadata path): candidate_id=%s expected_role=%s -> %s (%.4fs)",
                self._last_action_candidate_id, expected_role, meta_result, elapsed
            )
            return meta_result

        # 2. Bounded live DOM fallback if metadata genuinely unavailable
        res = self._validate_from_dom_fallback(expected_role)
        self.last_timing_diagnostics["validate_step_sec"] = time.time() - t0
        return res

    def _validate_from_dom_fallback(self, expected_role: Optional[str]) -> bool:
        """
        Fallback live DOM validation with strict bounded timeout (EVALUATOR_TIMEOUT_MS).
        Only called if private metadata was genuinely unavailable.
        """
        EVALUATOR_TIMEOUT_MS = 500

        if self._last_action_candidate_id is None:
            return False

        page = self._get_page()
        candidate = self._last_executed_candidate
        if candidate is None:
            return False

        locator = self._build_locator(page, candidate)
        if locator is None:
            return False

        try:
            actual_role = locator.first.get_attribute(SEMANTIC_ROLE_ATTR, timeout=EVALUATOR_TIMEOUT_MS)
            matched = (actual_role == expected_role) if expected_role is not None else False
            return matched
        except Exception as exc:
            logger.debug("_validate_from_dom_fallback get_attribute failed: %s", exc)
            return False

    def validate_page_state(self, success_condition: str, expected_value: str) -> bool:
        """
        Evaluates a page-state assertion (e.g. url_equals) directly against current browser state.

        Does NOT use data-semantic-role.
        """
        page = self._get_page()
        if success_condition == "url_equals":
            from urllib.parse import urlparse
            current_url = page.url
            current_norm = (urlparse(current_url).path or "/").rstrip("/") or "/"
            expected_norm = (urlparse(expected_value).path if expected_value.startswith("http") else expected_value).rstrip("/") or "/"
            matched = (current_norm == expected_norm)
            logger.debug("validate_page_state: url_equals current='%s' expected='%s' matched=%s", current_norm, expected_norm, matched)
            return matched
        elif success_condition == "text_contains":
            try:
                content = page.content()
                return expected_value in content
            except Exception as exc:
                logger.debug("validate_page_state: text_contains check failed: %s", exc)
                return False
        return False

    # ── Internal Helpers ───────────────────────────────────────────────────

    def _get_page(self) -> Page:
        self._ensure_browser()
        assert self._page is not None
        return self._page

    def _navigate_to(self, path: str, level: int, seed: int) -> None:
        """
        Activate mutation via Phase 3 API, then navigate to the target path.

        Phase 3 mechanism (matches frozen evaluate_candidate_recall.py exactly):
            1. GET /api/set-mutation?level=X&seed=Y
            2. Verify response: {"ok": true, "level": X, "seed": Y}
            3. page.goto(<base_url><path>)   -- clean URL, no query params

        No pre-flight login is performed.
        """
        page = self._get_page()

        # Step 1 & 2: Activate mutation via Phase 3 API
        self._set_mutation(level, seed)

        # Step 3: Navigate to clean path (no query params)
        url = f"{self.base_url}{path}"
        logger.debug("Navigating to: %s (level=%d seed=%d)", url, level, seed)
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        time.sleep(NAVIGATION_WAIT_SECONDS)

    def _set_mutation(self, level: int, seed: int) -> None:
        """
        Call the Phase 3 /api/set-mutation endpoint and verify the response.
        Includes retries to handle transient Vite dev server delays gracefully.
        """
        api_url = f"{self.base_url}/api/set-mutation?level={level}&seed={seed}"
        logger.debug("Setting mutation: %s", api_url)
        last_exc: Optional[Exception] = None

        for attempt in range(3):
            try:
                with urllib.request.urlopen(api_url, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

                if not data.get("ok"):
                    raise RuntimeError(f"API returned error: {data}")
                if data.get("level") != level or data.get("seed") != seed:
                    raise RuntimeError(
                        f"confirmed level={data.get('level')} seed={data.get('seed')} "
                        f"but expected level={level} seed={seed}"
                    )
                logger.debug("Mutation confirmed: level=%d seed=%d", level, seed)
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(0.5)

        raise RuntimeError(f"_set_mutation: HTTP request failed after 3 attempts: {last_exc}") from last_exc

    def _build_locator(self, page: Page, candidate: UICandidate):
        """
        Build a Playwright locator for the given candidate using a priority chain:

        1. ``id`` attribute (most stable when present and not mutated)
        2. ``[name=...]`` attribute
        3. ``[aria-label=...]``
        4. ``[placeholder=...]``
        5. Position-based fallback: element at (x, y) coordinates

        Returns None if no usable identity can be derived.
        """
        el_id = candidate.attributes.get("id", "").strip()
        if el_id:
            return page.locator(f"#{el_id}")

        el_name = candidate.attributes.get("name", "").strip()
        if el_name and candidate.tag in ("input", "select", "textarea"):
            return page.locator(f"[name='{el_name}']")

        if candidate.aria_label:
            return page.get_by_label(candidate.aria_label, exact=True)

        if candidate.placeholder:
            return page.get_by_placeholder(candidate.placeholder, exact=True)

        if candidate.text:
            return page.get_by_text(candidate.text.strip(), exact=True)

        if candidate.tag and candidate.element_type:
            return page.locator(f"{candidate.tag}[type='{candidate.element_type}']")

        logger.debug("_build_locator: no stable locator for candidate_id=%d", candidate.candidate_id)
        return None

    def _find_candidate(self, candidate_id: int) -> Optional[UICandidate]:
        """Retrieve a UICandidate from the last extraction by its candidate_id."""
        for c in self._last_candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def _enrich_private_metadata(
        self,
        page: Page,
        candidates: List[UICandidate],
        private_meta: List[PrivateEvaluatorMetadata],
        expected_role: str,
    ) -> List[PrivateEvaluatorMetadata]:
        """
        Scan the DOM for the element whose ``data-semantic-role == expected_role``,
        determine which UICandidate it corresponds to, and update that candidate's
        PrivateEvaluatorMetadata.semantic_role.

        This enables validate_step() to fall back to metadata when get_attribute fails.
        Uses approximate bounding-box matching to link DOM ground truth to candidates.

        NOTE: This entire enrichment is EVALUATOR-ONLY. The enriched metadata is
        never placed in any agent-visible structure.
        """
        try:
            target_el = page.locator(f"[{SEMANTIC_ROLE_ATTR}='{expected_role}']").first
            target_rect_js = target_el.bounding_box(timeout=500)
        except Exception:
            return private_meta

        if target_rect_js is None:
            return private_meta

        tx = target_rect_js.get("x", -1)
        ty = target_rect_js.get("y", -1)
        tw = target_rect_js.get("width", 0)
        th = target_rect_js.get("height", 0)

        enriched = list(private_meta)
        best_cid = -1
        best_overlap = -1.0

        for cand in candidates:
            # Approximate overlap: check centre-point containment
            cx_centre = cand.x + cand.width / 2
            cy_centre = cand.y + cand.height / 2
            if tx <= cx_centre <= tx + tw and ty <= cy_centre <= ty + th:
                overlap = cand.width * cand.height
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cid = cand.candidate_id

        if best_cid >= 0:
            for idx, meta in enumerate(enriched):
                if meta.candidate_id == best_cid:
                    enriched[idx] = PrivateEvaluatorMetadata(
                        candidate_id=best_cid,
                        semantic_role=expected_role,
                    )
                    break

        return enriched

    def _validate_from_metadata(self, expected_role: Optional[str]) -> Optional[bool]:
        """
        Evaluates step correctness using evaluator-private metadata captured during get_candidates().

        Returns:
            True if metadata exists for the selected candidate AND its semantic_role == expected_role.
            False if metadata exists for the selected candidate BUT its semantic_role != expected_role.
            None ONLY if no private metadata exists for the selected candidate.
        """
        if self._last_action_candidate_id is None or not self._last_private_meta:
            return None

        for meta in self._last_private_meta:
            if meta.candidate_id == self._last_action_candidate_id:
                if expected_role is None:
                    return False
                matched = (meta.semantic_role == expected_role)
                return matched

        return None
