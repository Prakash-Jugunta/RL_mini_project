import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const frontend = fileURLToPath(new URL('../', import.meta.url));
const run = new Date().toISOString().replace(/[:.]/g, '-');
const output = join(frontend, '..', 'artifacts', 'recovery-demo', run);
const report = join(output, 'index.html');
const options = process.argv.slice(2);
const cli = createRequire(import.meta.url).resolve('@playwright/test/cli');

mkdirSync(output, { recursive: true });
console.log(`Fielddesk recovery report: ${report}`);
const tests = spawn(process.execPath, [cli, 'test', '--config=playwright.demo.config.js'], {
  cwd: frontend,
  env: {
    ...process.env,
    RECOVERY_DEMO_DIR: output,
    RECOVERY_DEMO_HEADLESS: options.includes('--headless') ? '1' : '0',
  },
  stdio: 'inherit',
  windowsHide: true,
});
const [code] = await once(tests, 'close');
process.exitCode = code ?? 1;

if (existsSync(report)) {
  console.log(`\nOpen the workflow and full results at ${pathToFileURL(report).href}`);
  if (!options.includes('--no-open') && !options.includes('--headless') && !process.env.CI) {
    const opener = process.platform === 'win32' ? 'rundll32.exe' : process.platform === 'darwin' ? 'open' : 'xdg-open';
    const href = pathToFileURL(report).href;
    const args = process.platform === 'win32' ? ['url.dll,FileProtocolHandler', href] : [href];
    const browser = spawn(opener, args, { detached: true, stdio: 'ignore', windowsHide: true });
    browser.on('error', () => console.log('Open the report link above to view the results.'));
    browser.unref();
  }
} else {
  console.error('The run did not produce its HTML report. Check the Playwright output above.');
  process.exitCode ||= 1;
}
