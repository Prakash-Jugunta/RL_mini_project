/**
 * MUTATION ACTIVATION VERIFICATION TESTS
 *
 * These tests verify that the mutation injection mechanism is working
 * correctly — i.e. that window.__MUTATION_LEVEL__ and
 * window.__MUTATION_SEED__ were received by the browser and that
 * MutationProvider applied the expected level/seed.
 *
 * PURPOSE:
 *   Evaluator/configuration integrity check only.
 *   These tests do NOT test application behaviour.
 *   They do NOT use data-semantic-role as an RL observation.
 *   They expose NO ground truth to any future RL agent.
 *
 * RUN BEFORE every full degradation evaluation sweep.
 *
 * NOTE:
 *   These tests read window.__MUTATION_LEVEL__ / window.__MUTATION_SEED__
 *   directly from the browser, which were injected by vite.config.js
 *   BEFORE React initialised. They do NOT query MutationContext internals.
 */

import { test, expect } from '@playwright/test';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reads the current mutation configuration from the Vite dev-server
 * status endpoint. Returns { level, seed }.
 */
async function getServerMutationConfig(request) {
  const response = await request.get('http://localhost:3000/api/get-mutation');
  return response.json();
}

/**
 * Reads the window globals that were injected into the browser
 * by vite.config.js transformIndexHtml.
 * Returns { level, seed }.
 */
async function getBrowserMutationConfig(page) {
  return page.evaluate(() => ({
    level: window.__MUTATION_LEVEL__,
    seed:  window.__MUTATION_SEED__
  }));
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('Mutation Activation Verification', () => {

  /**
   * Confirms the /api/set-mutation and /api/get-mutation endpoints are live
   * and reflect the server's current mutation configuration.
   */
  test('Server /api/set-mutation endpoint is reachable and updates state', async ({ request }) => {
    // Set to a known state
    const setResp = await request.get('http://localhost:3000/api/set-mutation?level=0&seed=42');
    expect(setResp.ok()).toBeTruthy();
    const setBody = await setResp.json();
    expect(setBody.ok).toBe(true);
    expect(setBody.level).toBe(0);
    expect(setBody.seed).toBe(42);

    // Confirm with GET
    const getResp = await request.get('http://localhost:3000/api/get-mutation');
    expect(getResp.ok()).toBeTruthy();
    const getBody = await getResp.json();
    expect(getBody.level).toBe(0);
    expect(getBody.seed).toBe(42);
  });

  /**
   * Level 0 activation:
   *   - window.__MUTATION_LEVEL__ === 0
   *   - window.__MUTATION_SEED__ is a number
   *   - Original IDs (#username, #password, #login-btn) are present
   *   - data-semantic-role attributes are present
   */
  test('Level 0 — browser receives level=0 and original IDs are intact', async ({ page, request }) => {
    // Configure server to Level 0 and allow a brief settle period
    await request.get('http://localhost:3000/api/set-mutation?level=0&seed=11');
    // Small pause to ensure any concurrent request handling is complete
    await page.waitForTimeout(300);

    await page.goto('/login', { timeout: 30_000 });

    // Verify browser received globals
    const config = await getBrowserMutationConfig(page);
    expect(config.level, 'window.__MUTATION_LEVEL__ must be 0').toBe(0);

    // Original IDs must be present
    await expect(page.locator('#username'),  '#username must exist at Level 0').toBeVisible();
    await expect(page.locator('#password'),  '#password must exist at Level 0').toBeVisible();
    await expect(page.locator('#login-btn'), '#login-btn must exist at Level 0').toBeVisible();

    // Ground-truth semantic roles must be present (evaluator check only)
    const usernameEl = page.locator('[data-semantic-role="username-input"]');
    const passwordEl = page.locator('[data-semantic-role="password-input"]');
    const loginBtn   = page.locator('[data-semantic-role="login-action"]');
    await expect(usernameEl, 'data-semantic-role="username-input" must exist').toBeAttached();
    await expect(passwordEl, 'data-semantic-role="password-input" must exist').toBeAttached();
    await expect(loginBtn,   'data-semantic-role="login-action" must exist').toBeAttached();

    // Collect and report actual IDs (evidence output)
    const actualUsernameId = await usernameEl.getAttribute('id');
    const actualPasswordId = await passwordEl.getAttribute('id');
    const actualLoginId    = await loginBtn.getAttribute('id');
    console.log('[ACTIVATION] Level 0 actual IDs:');
    console.log(`  username-input  id = "${actualUsernameId}" (expected: "username")`);
    console.log(`  password-input  id = "${actualPasswordId}" (expected: "password")`);
    console.log(`  login-action    id = "${actualLoginId}"    (expected: "login-btn")`);

    expect(actualUsernameId).toBe('username');
    expect(actualPasswordId).toBe('password');
    expect(actualLoginId).toBe('login-btn');
  });

  /**
   * Level 1, Seed 11 activation:
   *   - window.__MUTATION_LEVEL__ === 1
   *   - window.__MUTATION_SEED__ === 11
   *   - Original IDs (#username, #password, #login-btn) MUST NOT exist
   *   - data-semantic-role attributes MUST still exist (evaluator ground truth)
   *   - Actual mutated IDs are printed as evidence
   */
  test('Level 1 Seed 11 — browser receives level=1, original IDs gone, mutated IDs present', async ({ page, request }) => {
    // Configure server to L1 S11
    const setResp = await request.get('http://localhost:3000/api/set-mutation?level=1&seed=11');
    const setBody = await setResp.json();
    expect(setBody.level).toBe(1);
    expect(setBody.seed).toBe(11);

    await page.goto('/login');

    // Verify globals injected into browser
    const config = await getBrowserMutationConfig(page);
    console.log(`[ACTIVATION] Requested: level=1 seed=11 | Browser received: level=${config.level} seed=${config.seed}`);
    expect(config.level, 'window.__MUTATION_LEVEL__ must be 1').toBe(1);
    expect(config.seed,  'window.__MUTATION_SEED__ must be 11').toBe(11);

    // ── Original IDs MUST NOT exist ──────────────────────────────────
    const usernameById  = page.locator('#username');
    const passwordById  = page.locator('#password');
    const loginBtnById  = page.locator('#login-btn');

    await expect(usernameById,  '#username must NOT exist at Level 1').toHaveCount(0);
    await expect(passwordById,  '#password must NOT exist at Level 1').toHaveCount(0);
    await expect(loginBtnById,  '#login-btn must NOT exist at Level 1').toHaveCount(0);

    // ── Semantic roles MUST still exist (evaluator ground truth) ─────
    const usernameEl = page.locator('[data-semantic-role="username-input"]');
    const passwordEl = page.locator('[data-semantic-role="password-input"]');
    const loginBtn   = page.locator('[data-semantic-role="login-action"]');

    await expect(usernameEl, 'data-semantic-role="username-input" must still exist').toBeAttached();
    await expect(passwordEl, 'data-semantic-role="password-input" must still exist').toBeAttached();
    await expect(loginBtn,   'data-semantic-role="login-action" must still exist').toBeAttached();

    // ── Report actual mutated IDs (evidence) ─────────────────────────
    const actualUsernameId = await usernameEl.getAttribute('id');
    const actualPasswordId = await passwordEl.getAttribute('id');
    const actualLoginId    = await loginBtn.getAttribute('id');

    console.log('[ACTIVATION] Level 1 Seed 11 actual MUTATED IDs:');
    console.log(`  username-input  id = "${actualUsernameId}"  (was: "username")`);
    console.log(`  password-input  id = "${actualPasswordId}"  (was: "password")`);
    console.log(`  login-action    id = "${actualLoginId}"     (was: "login-btn")`);

    // Confirm they are actually different from the originals
    expect(actualUsernameId).not.toBe('username');
    expect(actualPasswordId).not.toBe('password');
    expect(actualLoginId).not.toBe('login-btn');

    // Confirm they follow the mutated hex pattern
    expect(actualUsernameId).toMatch(/^username-x[0-9a-f]+$/i);
    expect(actualPasswordId).toMatch(/^password-x[0-9a-f]+$/i);
    expect(actualLoginId).toMatch(/^login-x[0-9a-f]+$/i);
  });

  /**
   * Server-browser synchrony check:
   *   What the server says it set is what the browser received.
   *   Verifies there is no stale-state or caching issue.
   */
  test('Server config and browser globals are in sync after set-mutation', async ({ page, request }) => {
    const expected = {
      level: 3,
      seed: 22
    };

    // 1. Set configuration and verify the API response itself.
    const setResponse = await request.get(
      `http://localhost:3000/api/set-mutation?level=${expected.level}&seed=${expected.seed}`
    );

    expect(setResponse.ok()).toBeTruthy();

    const setBody = await setResponse.json();

    expect(setBody.level).toBe(expected.level);
    expect(setBody.seed).toBe(expected.seed);

    // 2. Independently verify server state BEFORE loading React.
    const serverBeforeNavigation =
      await getServerMutationConfig(request);

    expect(serverBeforeNavigation.level).toBe(expected.level);
    expect(serverBeforeNavigation.seed).toBe(expected.seed);

    // 3. Now request a fresh HTML document.
    // transformIndexHtml should inject the verified current state.
    await page.goto('/login', {
      waitUntil: 'domcontentloaded'
    });

    // 4. Read browser globals.
    const browserConfig =
      await getBrowserMutationConfig(page);

    // 5. Read server again for diagnostic evidence.
    const serverAfterNavigation =
      await getServerMutationConfig(request);

    console.log('[ACTIVATION SYNC]');
    console.log('Requested:', expected);
    console.log('Set API:', setBody);
    console.log('Server before navigation:', serverBeforeNavigation);
    console.log('Browser:', browserConfig);
    console.log('Server after navigation:', serverAfterNavigation);

    // 6. Assert against the REQUESTED values,
    // not merely browser === server.
    expect(browserConfig.level).toBe(expected.level);
    expect(browserConfig.seed).toBe(expected.seed);

    expect(serverAfterNavigation.level).toBe(expected.level);
    expect(serverAfterNavigation.seed).toBe(expected.seed);
  });

});
