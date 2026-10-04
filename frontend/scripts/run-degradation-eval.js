/**
 * PHASE 3 — BASELINE DEGRADATION EVALUATION RUNNER (CORRECTED)
 *
 * FIXES IMPLEMENTED:
 *   1. Robust Playwright Result Collection reading raw JSON report file on disk.
 *   2. Strict expected test counts:
 *      - Activation: exactly 4 tests
 *      - Gate Check: exactly 2 tests
 *      - Normal Evaluation: exactly 6 tests per seed (210 executions total across L0-L6)
 *   3. Zero-Test Hard Failure: throws Error and ABORTS evaluation if tests.length === 0.
 *   4. Status Invariant: PASS only if totalTests > 0 AND failedCount === 0 AND passedCount === totalTests.
 *   5. Infrastructure failure separation: throws Error on process/server/file crash instead of treating as test failure.
 *   6. Failure Details Preservation & Classification: SELECTOR_NOT_FOUND, TIMEOUT, ASSERTION_FAILURE, NAVIGATION_FAILURE, APPLICATION_FAILURE.
 *   7. Completeness assertion: verifies exactly 210 baseline executions before producing final summary.json.
 */

import { execSync } from 'child_process';
import fs           from 'fs';
import path         from 'path';
import http         from 'http';

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const SEEDS  = [11, 22, 33, 44, 55];
const LEVELS = [0, 1, 2, 3, 4, 5, 6];

const RESULTS_DIR     = path.resolve('results/mutations');
const RAW_RESULTS_DIR = path.resolve('results/mutations/raw');
const OLD_RESULTS_DIR = path.resolve('results/mutations-INVALID-pre-fix');

const LEVEL_5_PRESETS = {
  11: { name: 'id+text',         activeMutations: ['id', 'text'] },
  22: { name: 'text+position',   activeMutations: ['text', 'position'] },
  33: { name: 'id+dom',          activeMutations: ['id', 'dom'] },
  44: { name: 'text+distractor', activeMutations: ['text', 'distractor'] },
  55: { name: 'id+position',     activeMutations: ['id', 'position'] }
};

const LEVEL_6_PRESETS = {
  // HELD OUT — NEVER USE FOR RL TRAINING
  11: { name: 'id+dom+distractor',       activeMutations: ['id', 'dom', 'distractor'] },
  22: { name: 'text+type+position',      activeMutations: ['text', 'type', 'position'] },
  33: { name: 'id+text+dom',             activeMutations: ['id', 'text', 'dom'] },
  44: { name: 'position+dom+distractor', activeMutations: ['position', 'dom', 'distractor'] },
  55: { name: 'id+text+dom+distractor',  activeMutations: ['id', 'text', 'dom', 'distractor'] }
};

const FIXED_LEVEL_MUTATIONS = {
  0: { name: 'Level 0 — Original',          activeMutations: [] },
  1: { name: 'Level 1 — ID+Class',          activeMutations: ['id', 'class'] },
  2: { name: 'Level 2 — Text',              activeMutations: ['text'] },
  3: { name: 'Level 3 — Structural',        activeMutations: ['position', 'dom'] },
  4: { name: 'Level 4 — Distractor',        activeMutations: ['distractor'] }
};

function resolvePreset(level, seed) {
  if (level === 5) return LEVEL_5_PRESETS[seed] || { name: 'id+text (fallback)', activeMutations: ['id', 'text'] };
  if (level === 6) return LEVEL_6_PRESETS[seed] || { name: 'id+dom+distractor (fallback)', activeMutations: ['id', 'dom', 'distractor'] };
  return FIXED_LEVEL_MUTATIONS[level] || FIXED_LEVEL_MUTATIONS[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function httpGet(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error(`JSON parse error from ${url}: ${e.message}\nBody: ${data}`)); }
      });
    }).on('error', reject);
  });
}

async function setMutation(level, seed, retries = 4) {
  const url = `http://localhost:3000/api/set-mutation?level=${level}&seed=${seed}`;
  let lastErr;
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const body = await httpGet(url);
      if (!body.ok || body.level !== level || body.seed !== seed) {
        throw new Error(`Unexpected response: ${JSON.stringify(body)}`);
      }
      return;
    } catch (e) {
      lastErr = e;
      const isTransient = e.code === 'ECONNRESET' || e.code === 'ECONNREFUSED' || e.code === 'ETIMEDOUT';
      if (isTransient && attempt < retries) {
        await sleep(800 * attempt);
        continue;
      }
      break;
    }
  }
  throw new Error(
    `Cannot reach /api/set-mutation after ${retries} attempts: ${lastErr?.message || lastErr}\n` +
    `Make sure the Vite dev server is running: npm run dev`
  );
}

function classifyFailure(errorMsg) {
  if (!errorMsg) return { failureType: 'UNKNOWN', selector: null };

  const locatorMatch = errorMsg.match(/locator\(['"]([^'"]+)['"]\)/) || errorMsg.match(/waiting for locator\(['"]([^'"]+)['"]\)/);
  const selector = locatorMatch ? locatorMatch[1] : null;

  const lower = errorMsg.toLowerCase();

  if (lower.includes('page.goto') || lower.includes('err_connection_refused') || lower.includes('net::') || lower.includes('navigation failed')) {
    return { failureType: 'NAVIGATION_FAILURE', selector };
  }
  if (locatorMatch || lower.includes('waiting for locator') || lower.includes('element handles')) {
    return { failureType: 'SELECTOR_NOT_FOUND', selector };
  }
  if (lower.includes('expect(') || lower.includes('tobe') || lower.includes('toequal') || lower.includes('tohavetext') || lower.includes('assert')) {
    return { failureType: 'ASSERTION_FAILURE', selector };
  }
  if (lower.includes('500') || lower.includes('internal server error') || lower.includes('uncaught exception')) {
    return { failureType: 'APPLICATION_FAILURE', selector };
  }
  if (lower.includes('timeout') || lower.includes('timed out')) {
    return { failureType: 'TIMEOUT', selector };
  }

  return { failureType: 'ASSERTION_FAILURE', selector };
}

function collectTestsFromSuite(suite, testDetails, parentSuiteTitle = '') {
  const currentSuiteTitle = parentSuiteTitle
    ? `${parentSuiteTitle} > ${suite.title}`
    : suite.title;

  if (suite.specs && Array.isArray(suite.specs)) {
    suite.specs.forEach((spec) => {
      spec.tests?.forEach((t) => {
        const lastResult = t.results && t.results.length > 0 ? t.results[t.results.length - 1] : null;
        const status = lastResult?.status === 'passed' ? 'PASS' : 'FAIL';

        let errorSnippet = null;
        if (lastResult?.errors && lastResult.errors.length > 0) {
          errorSnippet = lastResult.errors.map(e => e.message || String(e)).join('\n');
        } else if (lastResult?.error?.message) {
          errorSnippet = lastResult.error.message;
        }

        const duration = lastResult?.duration ?? null;

        let failureType = 'NONE';
        let selector = null;

        if (status === 'FAIL') {
          const classification = classifyFailure(errorSnippet);
          failureType = classification.failureType;
          selector = classification.selector;
        }

        testDetails.push({
          title: spec.title,
          workflow: currentSuiteTitle || spec.file,
          file: spec.file,
          status,
          failureType,
          selector,
          errorSnippet: errorSnippet ? errorSnippet.substring(0, 300) : null,
          duration
        });
      });
    });
  }

  if (suite.suites && Array.isArray(suite.suites)) {
    suite.suites.forEach((child) => collectTestsFromSuite(child, testDetails, currentSuiteTitle));
  }
}

/**
 * Run Playwright for a specific test file list.
 * Output is written to rawJsonFile, then read and parsed from disk.
 */
function runPlaywright(testFiles, expectedCount, rawJsonFile) {
  ensureDir(path.dirname(rawJsonFile));

  if (fs.existsSync(rawJsonFile)) {
    fs.rmSync(rawJsonFile, { force: true });
  }

  const env = {
    ...process.env,
    PLAYWRIGHT_JSON_OUTPUT_NAME: rawJsonFile,
    EVAL_RUNNER: '1'
  };

  let execError = null;
  try {
    execSync(
      `npx playwright test ${testFiles} --reporter=json`,
      { env, cwd: path.resolve('.'), encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    );
  } catch (err) {
    execError = err;
  }

  if (!fs.existsSync(rawJsonFile)) {
    throw new Error(
      `Infrastructure failure: Playwright did not create raw JSON report file at ${rawJsonFile}.\n` +
      `Exec error: ${execError?.message || 'Unknown'}\n` +
      `Stderr: ${execError?.stderr?.toString() || 'None'}`
    );
  }

  let reportData;
  try {
    const rawContent = fs.readFileSync(rawJsonFile, 'utf-8');
    reportData = JSON.parse(rawContent);
  } catch (parseErr) {
    throw new Error(
      `Infrastructure failure: Failed to parse raw Playwright JSON report file at ${rawJsonFile}: ${parseErr.message}`
    );
  }

  const testDetails = [];
  if (reportData.suites) {
    reportData.suites.forEach((s) => collectTestsFromSuite(s, testDetails));
  }

  // Requirement 3: Hard failure on 0 tests
  if (testDetails.length === 0) {
    throw new Error(
      `Evaluation integrity failure: Playwright returned 0 tests for ${testFiles}. Raw report: ${rawJsonFile}`
    );
  }

  // Requirement 4: Verify expected test count
  if (expectedCount !== undefined && testDetails.length !== expectedCount) {
    throw new Error(
      `Evaluation integrity failure: Expected ${expectedCount} tests but received ${testDetails.length} for ${testFiles}. Raw report: ${rawJsonFile}`
    );
  }

  const passedCount = testDetails.filter((t) => t.status === 'PASS').length;
  const failedCount = testDetails.filter((t) => t.status === 'FAIL').length;

  // Requirement 5: Status invariant — PASS only if totalTests > 0 AND failedCount === 0 AND passedCount === totalTests
  const status = (testDetails.length > 0 && failedCount === 0 && passedCount === testDetails.length)
    ? 'PASS'
    : 'DEGRADED';

  return {
    status,
    passedCount,
    failedCount,
    totalTests: testDetails.length,
    tests: testDetails
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 0 — Archive invalid old results
// ─────────────────────────────────────────────────────────────────────────────

console.log('===============================================================');
console.log('  PHASE 3 — BASELINE DEGRADATION EVALUATION RUNNER (CORRECTED)');
console.log('===============================================================');
console.log('');

if (fs.existsSync(RESULTS_DIR)) {
  if (fs.existsSync(OLD_RESULTS_DIR)) {
    fs.rmSync(OLD_RESULTS_DIR, { recursive: true, force: true });
  }
  try {
    fs.cpSync(RESULTS_DIR, OLD_RESULTS_DIR, { recursive: true });
    fs.rmSync(RESULTS_DIR, { recursive: true, force: true });
    console.log(`[ARCHIVE] Old results archived to: ${OLD_RESULTS_DIR}`);
  } catch (archiveErr) {
    console.warn(`[ARCHIVE] Could not archive old results: ${archiveErr.message}`);
    fs.rmSync(RESULTS_DIR, { recursive: true, force: true });
  }
}
ensureDir(RESULTS_DIR);
ensureDir(RAW_RESULTS_DIR);

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — Verify Vite server and /api/set-mutation endpoint
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n[STEP 1] Verifying Vite dev server and mutation injection endpoint...');
try {
  await setMutation(0, 42);
  console.log('  ✓ /api/set-mutation is reachable and functional');
} catch (e) {
  console.error(`\n  ✗ ABORT: ${e.message}\n`);
  process.exit(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 — Activation sanity: run mutation-activation.spec.js (Expected: 4 tests)
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n[STEP 2] Running activation sanity tests (Expected = 4 tests)...');
await setMutation(0, 42);

const actRawReport = path.join(RAW_RESULTS_DIR, 'activation-verification.json');
const actSummaryReport = path.join(RESULTS_DIR, 'activation-verification.json');

const actResult = runPlaywright('tests/playwright/mutation-activation.spec.js', 4, actRawReport);

fs.writeFileSync(actSummaryReport, JSON.stringify({
  level: 0,
  seed: 42,
  preset: 'Activation Verification',
  status: actResult.status,
  passedCount: actResult.passedCount,
  failedCount: actResult.failedCount,
  totalTests: actResult.totalTests,
  tests: actResult.tests
}, null, 2));

console.log(`  Activation tests: ${actResult.passedCount} passed, ${actResult.failedCount} failed (${actResult.totalTests} total)`);
actResult.tests.forEach((t) => console.log(`    [${t.status}] ${t.title}`));

if (actResult.failedCount > 0 || actResult.totalTests !== 4) {
  console.error('\n  ✗ ABORT: Activation verification failed or returned wrong test count.');
  process.exit(1);
}
console.log('  ✓ All 4 activation sanity tests passed');

await sleep(1500);

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 — Baseline failure gate: login.spec.js at Level 1 Seed 11 (Expected: 2 tests)
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n[STEP 3] Baseline failure gate: login.spec.js at Level 1 Seed 11 (Expected = 2 tests)...');
await setMutation(1, 11);

const gateRawReport = path.join(RAW_RESULTS_DIR, 'gate-check-L1-S11.json');
const gateSummaryReport = path.join(RESULTS_DIR, 'gate-check-L1-S11.json');

const gateResult = runPlaywright('tests/playwright/login.spec.js', 2, gateRawReport);

fs.writeFileSync(gateSummaryReport, JSON.stringify({
  level: 1,
  seed: 11,
  preset: 'Level 1 — ID+Class',
  activeMutations: ['id', 'class'],
  status: gateResult.status,
  passedCount: gateResult.passedCount,
  failedCount: gateResult.failedCount,
  totalTests: gateResult.totalTests,
  tests: gateResult.tests
}, null, 2));

console.log(`  Login tests (L1 S11): ${gateResult.passedCount} passed, ${gateResult.failedCount} failed (${gateResult.totalTests} total)`);
gateResult.tests.forEach((t) => {
  console.log(`    [${t.status}] ${t.title}`);
  if (t.status === 'FAIL') console.log(`      → failureType: ${t.failureType}, selector: ${t.selector || 'none'}`);
});

const loginSuccessTest = gateResult.tests.find((t) => t.title.toLowerCase().includes('successful'));
if (!loginSuccessTest) {
  console.error('\n  ✗ ABORT: Could not find the successful-login test in gate results');
  process.exit(1);
}
if (loginSuccessTest.status === 'PASS') {
  console.error('\n  ✗ ABORT: login.spec.js "Successful Login" PASSED at Level 1 Seed 11.');
  console.error('    Mutations are NOT degrading brittle selectors as required.');
  process.exit(1);
}
console.log(`  ✓ Successful-login test correctly FAILED with: ${loginSuccessTest.failureType}`);
console.log('  ✓ Baseline failure gate passed');

await sleep(1500);

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — Full L0–L6 × 5-seed evaluation (Expected: 6 baseline tests per run)
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n[STEP 4] Running full degradation evaluation: Levels 0–6 × Seeds 11,22,33,44,55');
console.log('  Expected total normal test executions = 210 (7 levels × 5 seeds × 6 tests)\n');

const BASELINE_TEST_FILES = 'tests/playwright/login.spec.js tests/playwright/search.spec.js tests/playwright/profile.spec.js tests/playwright/checkout.spec.js';

const levelSummaries = [];
let totalNormalExecutions = 0;
const failureClassificationCounts = {
  SELECTOR_NOT_FOUND: 0,
  TIMEOUT: 0,
  ASSERTION_FAILURE: 0,
  NAVIGATION_FAILURE: 0,
  APPLICATION_FAILURE: 0,
  UNKNOWN: 0
};

for (const level of LEVELS) {
  const levelDir = path.join(RESULTS_DIR, `level-${level}`);
  ensureDir(levelDir);

  const isHeldOut = level === 6;
  console.log(`>>> Evaluating Level ${level}${isHeldOut ? ' [HELD-OUT — evaluation only]' : ''} × ${SEEDS.length} seeds: ${SEEDS.join(', ')}...`);

  let levelPassed = 0, levelFailed = 0, levelTotal = 0;
  const levelRuns = [];

  for (const seed of SEEDS) {
    const preset = resolvePreset(level, seed);

    await setMutation(level, seed);

    const rawReportFile = path.join(RAW_RESULTS_DIR, `level-${level}-seed-${seed}.json`);
    const runReportFile = path.join(levelDir, `seed-${seed}.json`);

    const runRes = runPlaywright(BASELINE_TEST_FILES, 6, rawReportFile);

    const seedResult = {
      level,
      seed,
      preset: preset.name,
      activeMutations: preset.activeMutations,
      isHeldOut,
      status: runRes.status,
      passedCount: runRes.passedCount,
      failedCount: runRes.failedCount,
      totalTests: runRes.totalTests,
      tests: runRes.tests
    };

    ensureDir(path.dirname(runReportFile));
    fs.writeFileSync(runReportFile, JSON.stringify(seedResult, null, 2));

    levelPassed += runRes.passedCount;
    levelFailed += runRes.failedCount;
    levelTotal  += runRes.totalTests;
    totalNormalExecutions += runRes.totalTests;

    // Track failure classifications
    runRes.tests.filter((t) => t.status === 'FAIL').forEach((t) => {
      const type = t.failureType || 'UNKNOWN';
      failureClassificationCounts[type] = (failureClassificationCounts[type] || 0) + 1;
    });

    levelRuns.push(seedResult);

    const statusLabel = runRes.status;
    const degradedTests = runRes.tests.filter((t) => t.status === 'FAIL');
    console.log(`  - Level ${level} Seed ${seed} [${preset.name}]: ${runRes.passedCount}/${runRes.totalTests} passed (${statusLabel})`);
    if (degradedTests.length > 0) {
      degradedTests.forEach((t) =>
        console.log(`      ↳ FAIL [${t.failureType}${t.selector ? ` | selector: ${t.selector}` : ''}]: ${t.title}`)
      );
    }

    await sleep(800);
  }

  const passRate = levelTotal > 0 ? ((levelPassed / levelTotal) * 100).toFixed(1) : '0.0';
  levelSummaries.push({
    level,
    isHeldOut,
    testsRun: levelTotal,
    passed: levelPassed,
    failed: levelFailed,
    passRate: `${passRate}%`,
    seeds: levelRuns.map((r) => ({
      seed: r.seed,
      preset: r.preset,
      activeMutations: r.activeMutations,
      status: r.status,
      passedCount: r.passedCount,
      failedCount: r.failedCount,
      totalTests: r.totalTests
    }))
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 — Mandatory Completeness Check (Requirement 11)
// ─────────────────────────────────────────────────────────────────────────────

console.log('\n[STEP 5] Performing mandatory evaluation completeness check...');

let completenessFailed = false;
for (const level of LEVELS) {
  const levelDir = path.join(RESULTS_DIR, `level-${level}`);
  const seedFiles = fs.readdirSync(levelDir).filter((f) => f.startsWith('seed-') && f.endsWith('.json'));
  if (seedFiles.length !== 5) {
    console.error(`  ✗ COMPLETENESS ERROR: Level ${level} has ${seedFiles.length} seed files, expected 5`);
    completenessFailed = true;
  }
}

console.log(`  Total normal evaluation test executions: ${totalNormalExecutions} (Expected: 210)`);
if (totalNormalExecutions !== 210) {
  console.error(`  ✗ COMPLETENESS ERROR: Total executions = ${totalNormalExecutions}, expected 210.`);
  completenessFailed = true;
}

if (completenessFailed) {
  console.error('\n  ✗ ABORTING FINAL REPORT CREATION DUE TO COMPLETENESS FAILURE.\n');
  process.exit(1);
}

console.log('  ✓ Completeness check PASSED: Exactly 5 seed files per level (35 seed files total), 210 test executions total.');

// ─────────────────────────────────────────────────────────────────────────────
// Step 6 — Write Summary (Requirement 12)
// ─────────────────────────────────────────────────────────────────────────────

const summaryFile = path.join(RESULTS_DIR, 'summary.json');
fs.writeFileSync(summaryFile, JSON.stringify({
  phase: 'Phase 3 — Baseline Degradation Evaluation',
  generatedAt: new Date().toISOString(),
  injectionMechanism: 'window globals via /api/set-mutation + transformIndexHtml',
  expectedTestCount: 210,
  actualTestCount: totalNormalExecutions,
  evaluatedSeeds: SEEDS,
  evaluatedLevels: LEVELS,
  failureClassifications: failureClassificationCounts,
  levels: levelSummaries
}, null, 2));

console.log('\n===============================================================');
console.log('  DEGRADATION EVALUATION COMPLETE');
console.log('===============================================================');
console.log(`Summary written to: ${summaryFile}\n`);

const tableData = levelSummaries.map((l) => ({
  level:    l.level,
  heldOut:  l.isHeldOut ? 'YES' : 'no',
  testsRun: l.testsRun,
  passed:   l.passed,
  failed:   l.failed,
  passRate: l.passRate
}));
console.table(tableData);

await setMutation(0, 42);
console.log('\n[CLEANUP] Server reset to Level 0, Seed 42.\n');
