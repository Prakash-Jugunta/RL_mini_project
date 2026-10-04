import { defineConfig, devices } from '@playwright/test';

/**
 * PLAYWRIGHT CONFIGURATION — PHASE 2/3
 *
 * MUTATION INJECTION MECHANISM:
 *   Mutations are NOT passed via baseURL query parameters (those are
 *   silently dropped when page.goto('/login') is called with an
 *   absolute-path route). Instead, the evaluation runner calls:
 *
 *     GET http://localhost:3000/api/set-mutation?level=X&seed=Y
 *
 *   before launching Playwright. The Vite dev-server plugin responds
 *   by injecting window.__MUTATION_LEVEL__ and window.__MUTATION_SEED__
 *   into every HTML response before React initialises.
 *
 *   The baseline test files do NOT change. They navigate with
 *   page.goto('/login') etc. and receive correctly mutated DOM.
 *
 * REPORTER:
 *   The reporter output file is controlled by the REPORT_FILE env var
 *   so the evaluation runner can direct each run's JSON to a unique
 *   location without modifying this file.
 */
export default defineConfig({
  testDir: './tests/playwright',

  // Deterministic sequential execution — no parallelism during evaluation
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,

  reporter: [
    ['list'],
    ['json', { outputFile: process.env.REPORT_FILE || 'results/baseline/playwright-baseline.json' }],
    ['html', { open: 'never' }]
  ],

  use: {
    // Clean base URL — NO query params.
    // Mutation level/seed are injected via window globals by the Vite plugin.
    baseURL: 'http://localhost:3000',

    trace: 'off',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',

    // Conservative timeouts for mutation evaluation stability
    actionTimeout: 10_000,
    navigationTimeout: 15_000
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    }
  ],

  webServer: process.env.EVAL_RUNNER ? undefined : {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    // Always reuse the existing server — the evaluation runner manages
    // mutation state via /api/set-mutation, not via server restarts.
    reuseExistingServer: true,
    timeout: 60 * 1000
  }
});
