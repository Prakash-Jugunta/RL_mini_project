import { join } from 'node:path';
import { defineConfig } from '@playwright/test';
import config from './playwright.config.js';

const output = process.env.RECOVERY_DEMO_DIR;
if (!output) throw new Error('Run the demo using npm run demo:recovery.');

const baseURL = 'http://127.0.0.1:3317';
const project = config.projects[0];

export default defineConfig({
  ...config,
  testMatch: 'storefront-recovery.spec.js',
  outputDir: join(output, 'evidence'),
  preserveOutput: 'always',
  reporter: [
    ['list'],
    ['json', { outputFile: join(output, 'results.json') }],
    ['html', { outputFolder: join(output, 'playwright-report'), open: 'never' }],
    ['./scripts/recovery-demo-reporter.js', { output }],
  ],
  projects: [{
    ...project,
    use: {
      ...project.use,
      baseURL,
      headless: process.env.RECOVERY_DEMO_HEADLESS === '1',
      trace: 'on',
      video: 'on',
    },
  }],
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 3317 --strictPort',
    url: baseURL,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
