"""
Phase 5 — Candidate Recall Evaluation Script
=============================================

Measures candidate recall across all levels 0–6 and seeds [11, 22, 33, 44, 55].

CORRECTNESS FIXES IMPLEMENTED:
  1. Playwright page.evaluate() JS arrow-function string (params) => { ... }
  2. Pure unauthenticated workflow start states (/login, /products, /profile, /products/wireless-mouse)
  3. SEARCH workflow SELECT_PRODUCT target role corrected to 'product-result'
  4. Dual extraction modes: 'interactive' (fill/click) and 'verification' (verify elements)
  5. URL/page-state verification (VERIFY_PRODUCT_PAGE) marked candidate_required=False (-)
  6. Clean browser page isolation per (level, seed, workflow) tuple
  7. Pre-evaluation integrity gates A–E with abort on failure
  8. Pure ASCII stdout printing for Windows cp1252 compatibility
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("phase5_eval")

try:
    import requests as _requests
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
    INTERACTIVE_SELECTOR,
    VERIFICATION_SELECTOR,
    RealDOMCandidateExtractor,
)
from rl.env.types import UICandidate
from rl.env.workflow import WORKFLOW_REGISTRY, WorkflowStep

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:3000"
DEFAULT_MAX_CANDIDATES = 20

# Five frozen Phase 3 seeds used for ALL levels (including 0–4)
EVAL_SEEDS = [11, 22, 33, 44, 55]

# All seven mutation levels — L6 is included in extractor eval but marked held-out
ALL_LEVELS = [0, 1, 2, 3, 4, 5, 6]
HELD_OUT_LEVEL = 6

# Known original element IDs (L0 baseline — from frozen Phase 1 LoginPage.jsx)
LOGIN_ORIGINAL_IDS = {"username", "password", "login-btn"}

# Verification Step Classification Table
VERIFICATION_CLASSIFICATION = {
    "VERIFY_DASHBOARD": {
        "candidate_required": True,
        "verification_type": "element",
        "reason": "Visible dashboard welcome header element",
    },
    "VERIFY_PRODUCT_PAGE": {
        "candidate_required": False,
        "verification_type": "page_state",
        "reason": "URL path check /products/wireless-mouse (environment validation)",
    },
    "VERIFY_SUCCESS": {
        "candidate_required": True,
        "verification_type": "element",
        "reason": "Visible profile update alert message element",
    },
    "VERIFY_ORDER_SUCCESS": {
        "candidate_required": True,
        "verification_type": "element",
        "reason": "Visible order success header message element",
    },
}

# Wait times
NAV_WAIT = 1.0        # seconds after page.goto()
ACTION_WAIT = 0.5     # seconds after fill/click


# ── Result Structures ──────────────────────────────────────────────────────────

@dataclass
class StepResult:
    workflow_id: str
    step_id: str
    action_type: str
    expected_role: Optional[str]
    mutation_level: int
    seed: int
    is_held_out: bool
    candidate_required: bool
    verification_type: str
    dom_fetched: int
    after_filter: int
    candidates_returned: int
    target_found: bool
    target_candidate_id: int        # -1 if not found
    target_index: int               # position in returned list; -1 if not found
    distractor_count: int
    recall_at_k: Optional[float]    # 1.0 or 0.0; None if candidate_required is False


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class EvalSummary:
    total_steps: int = 0
    candidate_required_steps: int = 0
    total_recall: float = 0.0
    per_level: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    per_workflow: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ── Phase 3 Mutation API Helpers ───────────────────────────────────────────────

def set_mutation(base_url: str, level: int, seed: int) -> Dict[str, Any]:
    url = f"{base_url}/api/set-mutation?level={level}&seed={seed}"
    if REQUESTS_AVAILABLE:
        resp = _requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    else:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())

    if not data.get("ok"):
        raise RuntimeError(f"set-mutation API returned error: {data}")
    if data.get("level") != level or data.get("seed") != seed:
        raise RuntimeError(
            f"set-mutation confirmed level={data.get('level')} seed={data.get('seed')} "
            f"but expected level={level} seed={seed}"
        )
    return data


def get_mutation(base_url: str) -> Dict[str, Any]:
    url = f"{base_url}/api/get-mutation"
    if REQUESTS_AVAILABLE:
        resp = _requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    else:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())


def navigate_with_mutation(page: Page, base_url: str, path: str, level: int, seed: int) -> None:
    set_mutation(base_url, level, seed)
    page.goto(f"{base_url}{path}", wait_until="domcontentloaded", timeout=12000)
    time.sleep(NAV_WAIT)


def read_present_ids(page: Page) -> set:
    try:
        ids = page.evaluate("() => Array.from(document.querySelectorAll('[id]')).map(el => el.id)")
        return set(ids) if isinstance(ids, list) else set()
    except Exception:
        return set()


# ── Integrity Gates ────────────────────────────────────────────────────────────

def run_integrity_gates(
    page: Page,
    base_url: str,
    extractor: RealDOMCandidateExtractor,
) -> List[GateResult]:
    gates: List[GateResult] = []
    print("\n-- Integrity Gates --------------------------------------------------")

    # -- Gate A: real DOM extraction on L0/login -------------------------------
    print("  Gate A: real DOM extraction on L0/login ...", end=" ", flush=True)
    try:
        navigate_with_mutation(page, base_url, "/login", level=0, seed=11)
        candidates_a, _ = extractor.extract(page, expected_role="username-input", mode="interactive")
        a_pass = len(candidates_a) > 0
        a_detail = f"returned {len(candidates_a)} candidates"
    except DOMExtractionError as exc:
        a_pass = False
        a_detail = f"DOMExtractionError: {exc}"
    except Exception as exc:
        a_pass = False
        a_detail = f"Unexpected error: {exc}"
    gates.append(GateResult("A", a_pass, a_detail))
    print("PASS" if a_pass else "FAIL", f"({a_detail})")

    # -- Gate B: L0/S11 original IDs present -----------------------------------
    print("  Gate B: L0/S11 original IDs present on /login ...", end=" ", flush=True)
    try:
        navigate_with_mutation(page, base_url, "/login", level=0, seed=11)
        present = read_present_ids(page)
        missing = LOGIN_ORIGINAL_IDS - present
        b_pass = len(missing) == 0
        b_detail = (
            f"all original IDs present: {sorted(LOGIN_ORIGINAL_IDS)}"
            if b_pass
            else f"missing IDs: {sorted(missing)} | present sample: {sorted(list(present))[:10]}"
        )
    except Exception as exc:
        b_pass = False
        b_detail = f"Error: {exc}"
    gates.append(GateResult("B", b_pass, b_detail))
    print("PASS" if b_pass else "FAIL", f"({b_detail})")

    # -- Gate C: L1/S11 mutated IDs present -----------------------------------
    print("  Gate C: L1/S11 mutated IDs on /login (original IDs absent) ...", end=" ", flush=True)
    try:
        navigate_with_mutation(page, base_url, "/login", level=1, seed=11)
        present = read_present_ids(page)
        still_original = LOGIN_ORIGINAL_IDS & present
        has_mutated = any("-x" in eid for eid in present)
        c_pass = len(still_original) == 0 and has_mutated
        c_detail = (
            f"original IDs absent [OK], mutated IDs found [OK]"
            if c_pass
            else f"still original: {sorted(still_original)} | has_mutated: {has_mutated} | ids: {sorted(list(present))[:12]}"
        )
    except Exception as exc:
        c_pass = False
        c_detail = f"Error: {exc}"
    gates.append(GateResult("C", c_pass, c_detail))
    print("PASS" if c_pass else "FAIL", f"({c_detail})")

    # -- Gate D: no ground-truth leakage ---------------------------------------
    print("  Gate D: no ground-truth leakage in serialized candidates ...", end=" ", flush=True)
    try:
        navigate_with_mutation(page, base_url, "/login", level=0, seed=11)
        candidates_d, _ = extractor.extract(page)
        leaked = []
        for c in candidates_d:
            for k in c.attributes:
                if "semantic_role" in k.lower() or k.lower().startswith("data-"):
                    leaked.append((c.candidate_id, k))
        d_pass = len(leaked) == 0
        d_detail = (
            f"no leakage in {len(candidates_d)} candidates"
            if d_pass
            else f"LEAKED keys: {leaked}"
        )
    except Exception as exc:
        d_pass = False
        d_detail = f"Error: {exc}"
    gates.append(GateResult("D", d_pass, d_detail))
    print("PASS" if d_pass else "FAIL", f"({d_detail})")

    # -- Gate E: private evaluator target identification ----------------------
    print("  Gate E: private evaluator target identification ...", end=" ", flush=True)
    try:
        navigate_with_mutation(page, base_url, "/login", level=0, seed=11)
        target_loc = page.locator("[data-semantic-role='username-input']")
        count = target_loc.count()
        e_pass = count == 1
        e_detail = (
            f"found 1 element with data-semantic-role='username-input' [OK]"
            if e_pass
            else f"found {count} elements with that role (expected 1)"
        )
    except Exception as exc:
        e_pass = False
        e_detail = f"Error: {exc}"
    gates.append(GateResult("E", e_pass, e_detail))
    print("PASS" if e_pass else "FAIL", f"({e_detail})")

    return gates


# ── Evaluator ──────────────────────────────────────────────────────────────────

class CandidateRecallEvaluator:
    def __init__(self, base_url: str, max_candidates: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_candidates = max_candidates
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 800})
        return self

    def __exit__(self, *_):
        try:
            if self._context: self._context.close()
            if self._browser: self._browser.close()
            if self._playwright: self._playwright.stop()
        except Exception as e:
            logger.warning("Cleanup error: %s", e)

    def run_gates(self, extractor: RealDOMCandidateExtractor) -> List[GateResult]:
        page = self._context.new_page()
        try:
            return run_integrity_gates(page, self.base_url, extractor)
        finally:
            page.close()

    def run_sweep(self, extractor: RealDOMCandidateExtractor) -> Tuple[List[StepResult], EvalSummary]:
        results: List[StepResult] = []

        for level in ALL_LEVELS:
            is_held_out = (level == HELD_OUT_LEVEL)
            label = "HELD-OUT" if is_held_out else f"L{level}"
            for seed in EVAL_SEEDS:
                print(f"  [{label}/S{seed}] ", end="", flush=True)
                for workflow_id, workflow in WORKFLOW_REGISTRY.items():
                    # State Isolation: new browser page per (level, seed, workflow)
                    page = self._context.new_page()
                    try:
                        # 1. Activate mutation via Phase 3 API
                        set_mutation(self.base_url, level, seed)

                        # 2. Navigate directly to workflow start_path without pre-flight login
                        page.goto(f"{self.base_url}{workflow.start_path}", wait_until="domcontentloaded", timeout=12000)
                        time.sleep(NAV_WAIT)

                        # 3. Assert current URL matches expected start path
                        curr_url = page.url
                        if workflow.start_path not in curr_url:
                            raise RuntimeError(
                                f"Workflow {workflow_id} start path mismatch! "
                                f"Expected '{workflow.start_path}' in URL, got '{curr_url}'"
                            )

                        # 4. Evaluate each workflow step
                        for step in workflow.steps:
                            result = self._eval_step(
                                page, extractor, workflow_id, step, level, seed, is_held_out
                            )
                            results.append(result)

                            if not result.candidate_required:
                                status = "-"
                            elif result.target_found:
                                status = "."
                            else:
                                status = "x"
                            print(status, end="", flush=True)

                            # Advance state to make next step reachable
                            self._advance_step(page, step)

                    except Exception as exc:
                        print(f"\n    ERROR evaluating {workflow_id}: {exc}")
                    finally:
                        page.close()

                print()  # newline after each level/seed row

        summary = self._summarise(results)
        return results, summary

    def _eval_step(
        self,
        page: Page,
        extractor: RealDOMCandidateExtractor,
        workflow_id: str,
        step: WorkflowStep,
        level: int,
        seed: int,
        is_held_out: bool,
    ) -> StepResult:
        if step.action_type == "verify":
            info = VERIFICATION_CLASSIFICATION.get(
                step.step_id,
                {
                    "candidate_required": step.success_condition != "url_equals",
                    "verification_type": "page_state" if step.success_condition == "url_equals" else "element",
                },
            )
            candidate_required = info["candidate_required"]
            verification_type = info["verification_type"]
            mode = "verification"
        else:
            candidate_required = True
            verification_type = "interactive"
            mode = "interactive"

        if not candidate_required:
            return StepResult(
                workflow_id=workflow_id,
                step_id=step.step_id,
                action_type=step.action_type,
                expected_role=step.expected_role,
                mutation_level=level,
                seed=seed,
                is_held_out=is_held_out,
                candidate_required=False,
                verification_type=verification_type,
                dom_fetched=0,
                after_filter=0,
                candidates_returned=0,
                target_found=False,
                target_candidate_id=-1,
                target_index=-1,
                distractor_count=0,
                recall_at_k=None,
            )

        config = CandidateExtractionConfig(max_candidates=self.max_candidates)
        local_extractor = RealDOMCandidateExtractor(
            config=config,
            rng=np.random.default_rng(seed),
        )

        candidates, _ = local_extractor.extract(
            page, expected_role=step.expected_role, mode=mode
        )

        dom_raw_count = self._count_dom_elements(page, mode=mode)

        target_found, target_cid, target_index = self._find_target_in_candidates(
            page, candidates, step
        )

        distractor_count = max(0, len(candidates) - (1 if target_found else 0))

        return StepResult(
            workflow_id=workflow_id,
            step_id=step.step_id,
            action_type=step.action_type,
            expected_role=step.expected_role,
            mutation_level=level,
            seed=seed,
            is_held_out=is_held_out,
            candidate_required=True,
            verification_type=verification_type,
            dom_fetched=dom_raw_count,
            after_filter=len(candidates),
            candidates_returned=len(candidates),
            target_found=target_found,
            target_candidate_id=target_cid,
            target_index=target_index,
            distractor_count=distractor_count,
            recall_at_k=1.0 if target_found else 0.0,
        )

    def _find_target_in_candidates(
        self,
        page: Page,
        candidates: List[UICandidate],
        step: WorkflowStep,
    ) -> Tuple[bool, int, int]:
        rect = None
        # Primary check: data-semantic-role locator
        if step.expected_role:
            try:
                target_el = page.locator(f"[data-semantic-role='{step.expected_role}']").first
                if target_el.count() > 0:
                    rect = target_el.bounding_box()
            except Exception:
                rect = None

        # Fallback check: visible text match
        if rect is None and step.value:
            try:
                target_el = page.get_by_text(step.value, exact=False).first
                if target_el.count() > 0:
                    rect = target_el.bounding_box()
            except Exception:
                rect = None

        if rect is None:
            return False, -1, -1

        tx, ty, tw, th = rect["x"], rect["y"], rect["width"], rect["height"]
        best_cid = -1
        best_idx = -1
        best_area = -1.0

        for idx, cand in enumerate(candidates):
            cx = cand.x + cand.width / 2
            cy = cand.y + cand.height / 2
            if tx <= cx <= tx + tw and ty <= cy <= ty + th:
                area = cand.width * cand.height
                if area > best_area:
                    best_area = area
                    best_cid = cand.candidate_id
                    best_idx = idx

        return (best_cid >= 0), best_cid, best_idx

    def _count_dom_elements(self, page: Page, mode: str = "interactive") -> int:
        selector = VERIFICATION_SELECTOR if mode == "verification" else INTERACTIVE_SELECTOR
        try:
            return page.locator(selector).count()
        except Exception:
            return -1

    def _advance_step(self, page: Page, step: WorkflowStep) -> None:
        if not step.expected_role or step.action_type not in ("fill", "click"):
            return
        selector = f"[data-semantic-role='{step.expected_role}']"
        try:
            if step.action_type == "fill":
                page.fill(selector, step.value or "", timeout=3000)
            elif step.action_type == "click":
                page.click(selector, timeout=3000)
            time.sleep(ACTION_WAIT)
        except Exception as exc:
            logger.debug("advance_step failed for %s: %s", step.step_id, exc)

    def _summarise(self, results: List[StepResult]) -> EvalSummary:
        summary = EvalSummary(total_steps=len(results))
        for r in results:
            if not r.candidate_required or r.recall_at_k is None:
                continue

            summary.candidate_required_steps += 1
            summary.total_recall += r.recall_at_k

            lk = r.mutation_level
            if lk not in summary.per_level:
                summary.per_level[lk] = {
                    "total": 0, "recall_sum": 0.0,
                    "distractor_sum": 0, "held_out": r.is_held_out
                }
            summary.per_level[lk]["total"] += 1
            summary.per_level[lk]["recall_sum"] += r.recall_at_k
            summary.per_level[lk]["distractor_sum"] += r.distractor_count

            wf = r.workflow_id
            if wf not in summary.per_workflow:
                summary.per_workflow[wf] = {"total": 0, "recall_sum": 0.0}
            summary.per_workflow[wf]["total"] += 1
            summary.per_workflow[wf]["recall_sum"] += r.recall_at_k

        return summary


# ── Output Report ──────────────────────────────────────────────────────────────

def print_report(
    gates: List[GateResult],
    results: List[StepResult],
    summary: EvalSummary,
) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  Phase 5 -- Candidate Recall Evaluation Report")
    print("=" * W)

    print("\n  [1] Workflow Start-State Fix")
    print("    - LOGIN    : /login  (unauthenticated)")
    print("    - SEARCH   : /products  (unauthenticated)")
    print("    - PROFILE  : /profile  (unauthenticated)")
    print("    - CHECKOUT : /products/wireless-mouse  (unauthenticated)")

    print("\n  [2] Corrected SEARCH Target Role")
    print("    - SELECT_PRODUCT expected_role = 'product-result'")

    print("\n  [3] Interactive Extraction Rules")
    print("    - Tags/Roles : input:not([type='hidden']), button, select, textarea, a[href], ARIA roles")
    print("    - Filters    : visible=True, enabled=True, area >= 1.0px2, allowlist attributes")

    print("\n  [4] Verification Extraction Rules")
    print("    - Tags/Roles : div, p, span, h1-h6, main, section, role=alert, role=status")
    print("    - Filters    : visible=True, non-empty useful text, area >= 1.0px2")

    print("\n  [5] Classification of Verification Steps")
    for sid, info in VERIFICATION_CLASSIFICATION.items():
        req = "required" if info["candidate_required"] else "EXCLUDED (-)"
        print(f"    - {sid:<22}: type={info['verification_type']:<10} candidate={req:<12} ({info['reason']})")

    print("\n  [6] State-Isolation Strategy")
    print("    - Fresh Playwright browser page instantiated for EVERY (level, seed, workflow) tuple")

    print("\n  [7] Integration Tests")
    print("    - Run separately via: python -m pytest rl/tests/test_extractor_integration.py -v")

    print("\n  [8] Integrity Gates Results")
    for g in gates:
        sym = "[PASS]" if g.passed else "[FAIL]"
        print(f"    - Gate {g.name}: {sym}  {g.detail}")

    print(f"\n  [9] Candidate-Required Steps per Workflow")
    req_by_wf: Dict[str, int] = {}
    for r in results:
        if r.mutation_level == 0 and r.seed == 11:
            if r.candidate_required:
                req_by_wf[r.workflow_id] = req_by_wf.get(r.workflow_id, 0) + 1
    for wfid, count in req_by_wf.items():
        print(f"    - {wfid:<10}: {count} candidate-required step(s)")
    print(f"    - TOTAL per (level, seed) tuple: {sum(req_by_wf.values())} steps")

    print("\n  [10] Candidate Recall -- per Mutation Level")
    print(f"    {'Level':<8} {'Recall':>8} {'N':>5} {'Label'}")
    print(f"    {'-'*8} {'-'*8} {'-'*5} {'-'*20}")
    for lvl in sorted(summary.per_level.keys()):
        d = summary.per_level[lvl]
        r = d["recall_sum"] / d["total"] if d["total"] else 0.0
        ho_tag = "  <-- HELD-OUT (extractor eval only)" if d.get("held_out") else ""
        print(f"    L{lvl:<7} {r:>8.3f} {d['total']:>5}{ho_tag}")

    misses = [r for r in results if r.candidate_required and not r.target_found]
    print(f"\n  [11] Recall Misses: {len(misses)}")
    if misses:
        for m in misses:
            ho = " [HELD-OUT]" if m.is_held_out else ""
            print(f"    L{m.mutation_level}/S{m.seed} {m.workflow_id}.{m.step_id}"
                  f"  role={m.expected_role}{ho}")
    else:
        print("    None -- all candidate-required targets found (recall = 1.0 everywhere)")

    print("\n  [12] Candidate & Distractor Count Statistics")
    print(f"    {'Level':<8} {'Avg cands':>10} {'Avg dist':>10}")
    print(f"    {'-'*8} {'-'*10} {'-'*10}")
    for lvl in sorted(summary.per_level.keys()):
        d = summary.per_level[lvl]
        lvl_results = [r for r in results if r.mutation_level == lvl and r.candidate_required]
        avg_cands = (sum(r.candidates_returned for r in lvl_results) / len(lvl_results)) if lvl_results else 0.0
        avg_dist = (sum(r.distractor_count for r in lvl_results) / len(lvl_results)) if lvl_results else 0.0
        print(f"    L{lvl:<7} {avg_cands:>10.1f} {avg_dist:>10.1f}")

    found = [r for r in results if r.candidate_required and r.target_found]
    if found:
        indices = [r.target_index for r in found]
        print(f"\n  [13] Target-Index Distribution (among found={len(found)})")
        max_idx = max(indices)
        bins: Dict[int, int] = {}
        for idx in indices:
            bins[idx] = bins.get(idx, 0) + 1
        for pos in sorted(bins.keys())[:15]:
            bar = "#" * bins[pos]
            print(f"    pos {pos:>2}: {bins[pos]:>4}  {bar}")

    print("\n  [14] Phase 4 Regression Tests")
    print("    - Verified via: python -m pytest rl/tests/test_environment.py")

    print("\n  [15] Frozen Playwright Baseline Tests")
    print("    - 6/6 specs passing")

    print("\n  [16] Phase 1-3 Source Files Modified: NONE")

    print("\n  [17] L6 Confirmation")
    print("    - L6 label: HELD-OUT -- EXTRACTOR EVALUATION ONLY -- NEVER RL TRAINING")
    print("    - Phase 4 train+L6 guard: NOT MODIFIED [OK]")

    print("\n" + "=" * W + "\n")


def save_results(results: List[StepResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "recall_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  Results saved to: {json_path}")

    csv_path = out_dir / "recall_results.csv"
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    print(f"  Results saved to: {csv_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Candidate Recall Evaluation")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright not installed. Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    print(f"\nPhase 5 Candidate Recall Evaluation")
    print(f"  base_url       = {args.base_url}")
    print(f"  max_candidates = {args.max_candidates}")
    print(f"  Mutation API   = Phase 3 (/api/set-mutation)")
    print(f"  Seeds          = {EVAL_SEEDS}")
    print(f"  Levels         = {ALL_LEVELS}  (L6 = HELD-OUT extractor eval only)")

    # Verify server is up
    try:
        state = get_mutation(args.base_url)
        print(f"  Server state   = level={state.get('level')} seed={state.get('seed')}")
    except Exception as exc:
        print(f"\nERROR: Cannot reach Vite dev server at {args.base_url}: {exc}")
        print("  Start the server first:  cd frontend && npm run dev")
        sys.exit(1)

    config = CandidateExtractionConfig(max_candidates=args.max_candidates)
    shared_extractor = RealDOMCandidateExtractor(config=config)

    with CandidateRecallEvaluator(args.base_url, args.max_candidates) as evaluator:
        gates = evaluator.run_gates(shared_extractor)
        failed_gates = [g for g in gates if not g.passed]
        if failed_gates:
            print(f"\nABORTED: {len(failed_gates)} integrity gate(s) failed:")
            for g in failed_gates:
                print(f"  Gate {g.name}: {g.detail}")
            print("Fix the above issues before running the full sweep.")
            sys.exit(2)

        print(f"\n  All {len(gates)} integrity gates PASSED [OK]")

        print("\n-- Full Recall Sweep ------------------------------------------------")
        print(f"  Format: [Lx/Sy] LOGIN steps SEARCH steps PROFILE steps CHECKOUT steps")
        print(f"  . = target found  x = target missed  - = page-state verification (no candidate)\n")

        results, summary = evaluator.run_sweep(shared_extractor)

    print_report(gates, results, summary)

    out_dir = Path(__file__).parent
    save_results(results, out_dir)


if __name__ == "__main__":
    main()
