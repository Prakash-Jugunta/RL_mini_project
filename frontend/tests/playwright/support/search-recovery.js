import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { setTimeout as pause } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const repository = fileURLToPath(new URL('../../../../', import.meta.url));
const submitControls = 'form[role="search"] button, form[role="search"] input[type="submit"]';
const recoveryTimeoutMs = 40_000;

export async function presentSearchStage(page, testInfo, message, element) {
  if (!process.env.RECOVERY_DEMO_DIR) return;
  console.log(`  ${message}`);
  await page.evaluate((label) => {
    let banner = document.getElementById('fielddesk-recovery-status');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'fielddesk-recovery-status';
      banner.setAttribute('aria-hidden', 'true');
      banner.style.cssText = 'position:fixed;bottom:18px;left:18px;right:18px;width:fit-content;max-width:calc(100% - 36px);padding:12px 16px;background:#172632;color:#fff;border-left:4px solid #8ce2b1;border-radius:6px;font:600 15px system-ui;z-index:2147483647;pointer-events:none;box-shadow:0 5px 20px #0003';
      document.body.append(banner);
    }
    banner.textContent = label;
  }, message);
  if (element) {
    await element.scrollIntoViewIfNeeded();
    await element.evaluate((node) => { node.style.outline = '4px solid #20a66b'; node.style.outlineOffset = '4px'; });
  }
  await testInfo.attach(message, { body: await page.screenshot(), contentType: 'image/png' });
  await pause(900);
}

async function snapshot(page) {
  const elements = await page.locator(submitControls).elementHandles();
  const candidates = await Promise.all(elements.map(async (element, index) => {
    const details = await element.evaluate((node) => {
      if (!node.isConnected || node.type !== 'submit' || !node.form) return null;
      const bounded = (text) => String(text ?? '').trim().slice(0, 120);
      const label = bounded(node.getAttribute('aria-label'));
      const text = bounded(node.innerText);
      const { x, y, width, height } = node.getBoundingClientRect();
      return {
        name: label || text,
        scope: bounded(node.form.getAttribute('aria-label')),
        tag: node.tagName.toLowerCase(),
        element_type: node.type,
        text,
        aria_label: label,
        css_class: bounded(node.className),
        x, y, width, height,
      };
    });
    if (!details || !await element.isVisible() || !await element.isEnabled()) return null;
    return { element, details: { candidate_id: index, ...details } };
  }));
  return { candidates: candidates.filter(Boolean), elements };
}

function resolve(request, env) {
  return new Promise((resolveDecision, reject) => {
    const args = ['-m', 'rl.recovery.resolve'];
    if (env.RECOVERY_CHECKPOINT) args.push('--checkpoint', env.RECOVERY_CHECKPOINT);
    const child = spawn(env.RECOVERY_PYTHON || 'python', args, {
      cwd: repository,
      env: { ...env, PYTHONDONTWRITEBYTECODE: '1', LANGSMITH_TRACING: 'false' },
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error(`Search recovery exceeded its ${recoveryTimeoutMs / 1000}-second budget`));
    }, recoveryTimeoutMs);
    let output = '';
    child.stdout.setEncoding('utf8').on('data', (chunk) => { output += chunk; });
    child.stdin.on('error', reject);
    child.on('error', (error) => { clearTimeout(timeout); reject(error); });
    child.on('close', (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        reject(new Error(`Search recovery process exited with code ${code}`));
        return;
      }
      try {
        resolveDecision(JSON.parse(output));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(JSON.stringify(request));
  });
}

export async function clickSearch(page, testInfo, env = process.env) {
  const originalSelector = '#search-btn';
  const original = page.locator(originalSelector);
  const originalMatches = await original.count();
  if (originalMatches === 1) {
    await presentSearchStage(page, testInfo, 'Playwright found the original Search control.');
    await original.click();
    return 'original';
  }

  const captured = await snapshot(page);
  try {
    const candidates = captured.candidates.slice(0, 20);
    const viewport = page.viewportSize();
    const request = {
      snapshot_id: randomUUID(),
      workflow_id: 'SEARCH',
      action_type: 'click',
      intent: 'Search the product catalog',
      reference: { name: 'Search', scope: 'Catalog' },
      candidates: candidates.map(({ details }) => details),
      viewport_width: viewport.width,
      viewport_height: viewport.height,
    };
    await presentSearchStage(page, testInfo, `Original locator has ${originalMatches} matches. DQN is ranking ${candidates.length} controls.`);
    const started = performance.now();
    const decision = await resolve(request, env);
    const selected = candidates.find(({ details }) => details.candidate_id === decision.candidate_id);
    await testInfo.attach('search-recovery', {
      body: JSON.stringify({
        ...decision,
        decision_ms: Math.round(performance.now() - started),
        original_locator: originalSelector,
        original_matches: originalMatches,
        model_mode: decision.source === 'llm' ? env.RECOVERY_MODEL_MODE || 'live' : null,
        model: decision.source === 'llm' ? env.RECOVERY_OPENAI_MODEL || 'gpt-6.2-sol' : null,
        ranked: decision.ranked.map((entry) => {
          const { name, scope } = candidates.find(({ details }) => details.candidate_id === entry.candidate_id).details;
          return { ...entry, name, scope };
        }),
      }),
      contentType: 'application/json',
    });
    if (decision.snapshot_id !== request.snapshot_id || !selected || !['dqn', 'llm'].includes(decision.source)) {
      await presentSearchStage(page, testInfo, 'Recovery stopped. There is no verified replacement.');
      throw new Error(`Search locator unresolved (${decision.reason})`);
    }

    const live = await snapshot(page);
    try {
      const matches = live.candidates.filter(({ details }) =>
        details.name === selected.details.name && details.scope === selected.details.scope);
      if (matches.length !== 1 || !await selected.element.evaluate((node, current) => node === current, matches[0].element)) {
        throw new Error('Search recovery candidate is stale or ambiguous');
      }
      const source = decision.source === 'dqn' ? 'DQN' : env.RECOVERY_MODEL_MODE === 'mock' ? 'LLM branch (local mock)' : 'Live LLM';
      await presentSearchStage(page, testInfo, `${source} selected ${selected.details.scope} / ${selected.details.name}.`, selected.element);
      await selected.element.click({ timeout: 3000 });
      if (process.env.RECOVERY_DEMO_DIR) {
        await testInfo.attach('search-action', { body: JSON.stringify({ clicked: true }), contentType: 'application/json' });
      }
    } finally {
      await Promise.all(live.elements.map((element) => element.dispose()));
    }
    return decision.source;
  } finally {
    await Promise.all(captured.elements.map((element) => element.dispose()));
  }
}
