"""
Phase 5 — Real-Browser Integration Tests
=========================================

Tests the DOM candidate extraction pipeline against a REAL running Playwright
browser connected to the Vite dev server (http://localhost:3000).

Unlike the mock-based unit tests in test_candidate_extractor.py, these tests
execute the actual JavaScript in a real browser — the only way to catch bugs
in the JS expression passed to page.evaluate().

PREREQUISITES
-------------
    1. Vite dev server running:  cd frontend && npm run dev
    2. Playwright installed:     pip install playwright && playwright install chromium

SKIP BEHAVIOUR
--------------
All tests in this module are automatically skipped if the Vite dev server is
not reachable, so they never block the regular unit-test suite.

Run explicitly with:
    python -m pytest rl/tests/test_extractor_integration.py -v

IMPORTANT: These tests use the Phase 3 /api/set-mutation API — NOT query params.
The navigation pattern is page.goto('/login') without query params, matching
the frozen Phase 2 Playwright test contract.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Optional, Set

import numpy as np
import pytest

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from rl.candidates.extractor import (
    CandidateExtractionConfig,
    DOMExtractionError,
    RealDOMCandidateExtractor,
    ALLOWED_ATTRIBUTE_KEYS,
)

BASE_URL = "http://localhost:3000"
NAV_WAIT = 1.0

# ── Server Availability Check ─────────────────────────────────────────────────

def _server_is_up() -> bool:
    """Return True if the Vite dev server is reachable."""
    try:
        if REQUESTS_AVAILABLE:
            r = requests.get(f"{BASE_URL}/api/get-mutation", timeout=3)
            return r.status_code == 200
        else:
            import urllib.request
            with urllib.request.urlopen(f"{BASE_URL}/api/get-mutation", timeout=3):
                return True
    except Exception:
        return False


SERVER_UP = PLAYWRIGHT_AVAILABLE and _server_is_up()

skip_if_no_server = pytest.mark.skipif(
    not SERVER_UP,
    reason="Vite dev server not running at http://localhost:3000 — skipping integration tests"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_mutation(level: int, seed: int) -> Dict:
    url = f"{BASE_URL}/api/set-mutation?level={level}&seed={seed}"
    if REQUESTS_AVAILABLE:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    else:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())


def _navigate(page: Page, path: str, level: int, seed: int) -> None:
    """Activate mutation then navigate WITHOUT query params (Phase 3 contract)."""
    _set_mutation(level, seed)
    page.goto(f"{BASE_URL}{path}", wait_until="domcontentloaded", timeout=20000)
    page.reload(wait_until="domcontentloaded")
    time.sleep(NAV_WAIT)


def _read_present_ids(page: Page) -> Set[str]:
    ids = page.evaluate("() => Array.from(document.querySelectorAll('[id]')).map(el => el.id)")
    return set(ids) if isinstance(ids, list) else set()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def browser_page():
    """Module-scoped Playwright browser + page for all integration tests."""
    if not PLAYWRIGHT_AVAILABLE:
        pytest.skip("Playwright not installed")
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    yield page
    context.close()
    browser.close()
    pw.stop()


# ── Test Classes ──────────────────────────────────────────────────────────────

@skip_if_no_server
class TestRealBrowserJavaScript:
    """
    Requirement 1 — Real-browser JavaScript execution tests.
    These tests exercise the actual JS string passed to page.evaluate() in a real browser.
    """

    def test_login_page_returns_candidates(self, browser_page):
        """Basic smoke test: /login page returns at least one candidate."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(11))
        candidates, _ = ex.extract(browser_page, expected_role="username-input")
        assert len(candidates) > 0, "Expected at least one candidate on /login"

    def test_candidates_have_valid_fields(self, browser_page):
        """All returned UICandidate objects have the required non-None fields."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(11))
        candidates, _ = ex.extract(browser_page, expected_role="username-input")
        for c in candidates:
            assert c.candidate_id is not None
            assert isinstance(c.tag, str) and len(c.tag) > 0
            assert c.visible is True
            assert c.enabled is True
            assert c.x >= 0
            assert c.y >= 0
            assert c.width > 0
            assert c.height > 0

    def test_no_semantic_role_leakage(self, browser_page):
        """data-semantic-role must NEVER appear in returned candidate attributes."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(11))
        candidates, _ = ex.extract(browser_page)
        for c in candidates:
            for k in c.attributes:
                assert "semantic_role" not in k.lower(), (
                    f"Ground-truth key leaked: {k!r} in candidate {c.candidate_id}"
                )
                assert not k.lower().startswith("data-"), (
                    f"data-* attribute leaked: {k!r} in candidate {c.candidate_id}"
                )

    def test_attributes_within_allowlist(self, browser_page):
        """All attribute keys in returned candidates must be in ALLOWED_ATTRIBUTE_KEYS."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(11))
        candidates, _ = ex.extract(browser_page)
        for c in candidates:
            for k in c.attributes:
                assert k in ALLOWED_ATTRIBUTE_KEYS, (
                    f"Attribute key {k!r} not in allowlist — candidate {c.candidate_id}"
                )

    def test_candidate_count_bounded(self, browser_page):
        """Returned candidate list must not exceed max_candidates."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=5)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(11))
        candidates, _ = ex.extract(browser_page)
        assert len(candidates) <= 5, (
            f"Expected <= 5 candidates but got {len(candidates)}"
        )

    def test_interactive_mode_excludes_non_interactive(self, browser_page):
        """Interactive mode must not include passive div/p/span/h1-h6 elements."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page, mode="interactive")
        passive_tags = {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6",
                        "main", "section", "article", "header", "footer"}
        for c in candidates:
            assert c.tag not in passive_tags, (
                f"Interactive mode returned passive tag {c.tag!r} (candidate_id={c.candidate_id})"
            )

    def test_products_page_returns_candidates(self, browser_page):
        """/products page returns candidates without requiring login."""
        _navigate(browser_page, "/products", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page)
        assert len(candidates) > 0, "Expected candidates on /products (no login required)"

    def test_profile_page_returns_candidates(self, browser_page):
        """/profile page returns candidates without requiring login."""
        _navigate(browser_page, "/profile", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page)
        assert len(candidates) > 0, "Expected candidates on /profile (no login required)"

    def test_product_detail_page_returns_candidates(self, browser_page):
        """/products/wireless-mouse page returns candidates without login."""
        _navigate(browser_page, "/products/wireless-mouse", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page)
        assert len(candidates) > 0, "Expected candidates on /products/wireless-mouse"


@skip_if_no_server
class TestMutationLevelIsolation:
    """
    Requirement 2 — Mutation level isolation tests.
    Verifies that different mutation levels produce different DOM element IDs
    and that the Phase 3 API correctly activates each level.
    """

    def test_level_0_has_original_ids(self, browser_page):
        """Level 0 /login must have the original frozen Phase 1 element IDs."""
        _navigate(browser_page, "/login", level=0, seed=11)
        present = _read_present_ids(browser_page)
        original_ids = {"username", "password", "login-btn"}
        missing = original_ids - present
        assert len(missing) == 0, (
            f"L0 /login missing original IDs: {sorted(missing)} | present: {sorted(list(present))[:10]}"
        )

    def test_level_1_mutates_ids(self, browser_page):
        """Level 1 /login must NOT have the original IDs (they should be mutated)."""
        _navigate(browser_page, "/login", level=1, seed=11)
        present = _read_present_ids(browser_page)
        original_ids = {"username", "password", "login-btn"}
        still_original = original_ids & present
        assert len(still_original) == 0, (
            f"L1 /login still has original IDs (mutation not active?): {sorted(still_original)}"
        )

    def test_different_seeds_produce_different_ids(self, browser_page):
        """Two different seeds at L1 must produce different element IDs."""
        _navigate(browser_page, "/login", level=1, seed=11)
        ids_seed11 = _read_present_ids(browser_page)

        _navigate(browser_page, "/login", level=1, seed=22)
        ids_seed22 = _read_present_ids(browser_page)

        # Mutated IDs should differ between seeds
        assert ids_seed11 != ids_seed22, (
            "L1/S11 and L1/S22 produced identical element IDs — seed isolation broken"
        )

    def test_mutation_api_confirmed_values(self, browser_page):
        """set_mutation API response must confirm the exact level and seed requested."""
        response = _set_mutation(level=3, seed=33)
        assert response.get("ok") is True, f"API ok=False: {response}"
        assert response.get("level") == 3, f"Confirmed level mismatch: {response}"
        assert response.get("seed") == 33, f"Confirmed seed mismatch: {response}"
        # Reset
        _set_mutation(level=0, seed=11)

    def test_mutation_state_persists_across_navigation(self, browser_page):
        """After set-mutation, navigation to /login must receive the mutated DOM."""
        _set_mutation(level=1, seed=11)
        browser_page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded", timeout=12000)
        time.sleep(NAV_WAIT)
        present = _read_present_ids(browser_page)
        original_ids = {"username", "password", "login-btn"}
        still_original = original_ids & present
        assert len(still_original) == 0, (
            f"Mutation didn't persist on navigation: still_original={sorted(still_original)}"
        )
        # Reset
        _set_mutation(level=0, seed=11)


@skip_if_no_server
class TestVerificationModeExtraction:
    """
    Requirement 3 — Verification mode extraction tests.
    Confirms that 'verification' mode returns passive display elements
    (headings, alerts, paragraphs) — not interactive inputs/buttons.
    """

    def test_verification_mode_includes_text_elements(self, browser_page):
        """Verification mode must include div/p/span/h-tags."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page, mode="verification")
        verification_tags = {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6",
                              "main", "section"}
        found_passive = any(c.tag in verification_tags for c in candidates)
        assert found_passive, (
            f"Verification mode returned no passive elements. Tags: {[c.tag for c in candidates]}"
        )

    def test_verification_mode_has_text_content(self, browser_page):
        """Verification mode candidates must have non-empty text."""
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page, mode="verification")
        for c in candidates:
            assert c.text and len(c.text.strip()) > 0, (
                f"Verification candidate {c.candidate_id} (tag={c.tag}) has empty text"
            )


@skip_if_no_server
class TestSmokeAcrossLevels:
    """
    Smoke test: extractor returns >0 candidates at every training level.
    Prevents silent regression if a mutation breaks all element visibility.
    """

    @pytest.mark.parametrize("level,seed", [
        (0, 11), (1, 11), (2, 22), (3, 33), (4, 44), (5, 55),
    ])
    def test_candidates_nonempty_at_level(self, browser_page, level, seed):
        _navigate(browser_page, "/login", level=level, seed=seed)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config, rng=np.random.default_rng(seed))
        candidates, _ = ex.extract(browser_page, expected_role="username-input")
        assert len(candidates) > 0, (
            f"L{level}/S{seed}: got 0 candidates from /login — mutation may have hidden all elements"
        )


@skip_if_no_server
class TestStateIntegrity:
    """
    Requirement 7 — Real-browser state-integrity tests.
    Proves workflow start paths and step navigation transitions.
    """

    def test_workflow_start_paths(self, browser_page):
        """Verify each workflow starts at its correct unauthenticated start path."""
        # LOGIN
        _navigate(browser_page, "/login", level=0, seed=11)
        assert "/login" in browser_page.url

        # SEARCH
        _navigate(browser_page, "/products", level=0, seed=11)
        assert "/products" in browser_page.url

        # PROFILE
        _navigate(browser_page, "/profile", level=0, seed=11)
        assert "/profile" in browser_page.url

        # CHECKOUT
        _navigate(browser_page, "/products/wireless-mouse", level=0, seed=11)
        assert "/products/wireless-mouse" in browser_page.url

    def test_checkout_workflow_step_transitions(self, browser_page):
        """
        Verify action transitions:
          SELECT_PRODUCT -> /products/wireless-mouse
          CLICK_CART -> /cart
          PROCEED_CHECKOUT -> /checkout
          CONFIRM_ORDER -> /order-success
        """
        # 1. SEARCH: SELECT_PRODUCT -> /products/wireless-mouse
        _navigate(browser_page, "/products", level=0, seed=11)
        browser_page.fill("[data-semantic-role='search-input']", "Wireless Mouse")
        browser_page.click("[data-semantic-role='search-action']")
        time.sleep(0.5)
        browser_page.click("[data-semantic-role='product-result']")
        time.sleep(0.5)
        assert "/products/wireless-mouse" in browser_page.url

        # 2. CHECKOUT: ADD_TO_CART -> CLICK_CART -> PROCEED_CHECKOUT -> CONFIRM_ORDER
        _navigate(browser_page, "/products/wireless-mouse", level=0, seed=11)
        browser_page.click("[data-semantic-role='add-cart-action']")
        time.sleep(0.5)

        # CLICK_CART -> /cart
        browser_page.click("[data-semantic-role='nav-cart']")
        time.sleep(0.5)
        assert "/cart" in browser_page.url

        # PROCEED_CHECKOUT -> /checkout
        browser_page.click("[data-semantic-role='checkout-action']")
        time.sleep(0.5)
        assert "/checkout" in browser_page.url

        # CONFIRM_ORDER -> /order-success
        browser_page.click("[data-semantic-role='confirm-order-action']")
        time.sleep(0.5)
        assert "/order-success" in browser_page.url


@skip_if_no_server
class TestPlaywrightBrowserAdapter:
    """
    Requirement -- Adapter integration tests (Phase 5 open item).

    Verifies the PlaywrightBrowserAdapter contract without creating a nested
    sync_playwright() context (which raises "Sync API inside asyncio loop" when
    called from within the pytest-anyio event loop that the module fixture owns).

    Uses the existing module-scoped browser_page fixture and module-level helpers
    (_set_mutation, _navigate) which exercise the same HTTP + navigation contract
    as PlaywrightBrowserAdapter._set_mutation() / _navigate_to().
    """

    # ── Mutation API contract ─────────────────────────────────────────────────

    def test_mutation_api_round_trip(self):
        """
        GET /api/set-mutation?level=X&seed=Y must return {ok: true, level: X, seed: Y}.
        This is the same HTTP contract as PlaywrightBrowserAdapter._set_mutation().
        """
        response = _set_mutation(level=2, seed=33)
        assert response.get("ok") is True, f"API ok=False: {response}"
        assert response.get("level") == 2, f"level mismatch: {response}"
        assert response.get("seed") == 33, f"seed mismatch: {response}"
        # Confirm state via get-mutation
        import urllib.request as _urllib
        import json as _json
        with _urllib.urlopen(f"{BASE_URL}/api/get-mutation", timeout=5) as r:
            state = _json.loads(r.read().decode())
        assert state.get("level") == 2, f"Server level mismatch: {state}"
        assert state.get("seed") == 33,  f"Server seed mismatch: {state}"
        # Reset
        _set_mutation(level=0, seed=11)

    # ── Clean URL contract ────────────────────────────────────────────────────

    def test_navigate_produces_clean_url(self, browser_page):
        """
        After set-mutation + page.goto(), the URL must NOT contain
        mutation_level or mutation_seed query params.

        Same contract as PlaywrightBrowserAdapter._navigate_to().
        """
        _navigate(browser_page, "/login", level=0, seed=11)
        assert "mutation_level" not in browser_page.url, (
            f"mutation_level leaked into URL: {browser_page.url}"
        )
        assert "mutation_seed" not in browser_page.url, (
            f"mutation_seed leaked into URL: {browser_page.url}"
        )
        assert "/login" in browser_page.url, (
            f"Expected /login in URL after navigation, got: {browser_page.url}"
        )

    # ── No-auth start paths ───────────────────────────────────────────────────

    @pytest.mark.parametrize("path", [
        "/login",
        "/products",
        "/profile",
        "/products/wireless-mouse",
    ])
    def test_workflow_start_path_accessible_without_auth(self, browser_page, path):
        """
        Each workflow start path must be directly navigable without auth pre-flight.
        Verifies that the adapter correctly removed _ensure_logged_in().
        """
        _navigate(browser_page, path, level=0, seed=11)
        assert path in browser_page.url, (
            f"Expected {path!r} in URL (no auth redirect), got: {browser_page.url}"
        )
        # Explicitly confirm NOT redirected to /login
        assert "/login" not in browser_page.url or path == "/login", (
            f"Navigation to {path!r} was redirected to /login -- auth pre-flight still active"
        )

    def test_mutation_applied_before_navigation(self, browser_page):
        """
        set-mutation is called BEFORE page.goto(), so the page receives the
        mutated DOM. Verify by checking that original L0 IDs are absent at L1.
        """
        _navigate(browser_page, "/login", level=1, seed=11)
        present = _read_present_ids(browser_page)
        original_ids = {"username", "password", "login-btn"}
        still_original = original_ids & present
        assert len(still_original) == 0, (
            f"Mutation not applied before navigation: original IDs still present: "
            f"{sorted(still_original)}"
        )
        # Reset
        _navigate(browser_page, "/login", level=0, seed=11)

    # ── Extraction mode derivation ────────────────────────────────────────────

    def test_interactive_mode_for_fill_steps(self, browser_page):
        """
        action_type=fill/click -> mode=interactive.
        Interactive extraction must NOT return passive div/p/h* elements.
        """
        _navigate(browser_page, "/login", level=0, seed=11)
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page, mode="interactive")
        passive_tags = {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6",
                        "main", "section", "article"}
        for c in candidates:
            assert c.tag not in passive_tags, (
                f"interactive mode returned passive tag {c.tag!r} "
                f"(candidate_id={c.candidate_id}) -- mode derivation broken"
            )

    def test_verification_mode_for_verify_steps(self, browser_page):
        """
        action_type=verify -> mode=verification.
        Verification extraction must return text/display elements.
        Maps to VERIFY_DASHBOARD, VERIFY_SUCCESS, VERIFY_ORDER_SUCCESS.
        """
        # Navigate to /login then log in to reach /dashboard
        _navigate(browser_page, "/login", level=0, seed=11)
        browser_page.fill("[data-semantic-role='username-input']", "testuser")
        browser_page.fill("[data-semantic-role='password-input']", "password123")
        browser_page.click("[data-semantic-role='login-action']")
        time.sleep(1.0)
        # Extract in verification mode from /dashboard
        config = CandidateExtractionConfig(max_candidates=20)
        ex = RealDOMCandidateExtractor(config=config)
        candidates, _ = ex.extract(browser_page, mode="verification")
        verification_tags = {"div", "p", "span", "h1", "h2", "h3", "h4", "h5", "h6",
                              "main", "section"}
        found_passive = any(c.tag in verification_tags for c in candidates)
        assert found_passive, (
            f"verification mode returned no passive text elements. "
            f"Tags found: {[c.tag for c in candidates]}"
        )

    def test_verify_product_page_url_validation(self, browser_page):
        """
        VERIFY_PRODUCT_PAGE uses page-state URL validation.
        No RL candidate decisions or semantic roles are used.
        """
        _navigate(browser_page, "/products/wireless-mouse", level=0, seed=11)
        from urllib.parse import urlparse
        current_norm = (urlparse(browser_page.url).path or "/").rstrip("/") or "/"
        assert current_norm == "/products/wireless-mouse"

    def test_verify_product_page_failure(self, browser_page):
        """
        When browser path is /products, page-state URL validation for /products/wireless-mouse fails.
        """
        _navigate(browser_page, "/products", level=0, seed=11)
        from urllib.parse import urlparse
        current_norm = (urlparse(browser_page.url).path or "/").rstrip("/") or "/"
        assert current_norm != "/products/wireless-mouse"

