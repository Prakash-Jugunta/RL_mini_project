import { createServer } from 'node:http';
import { once } from 'node:events';
import { expect, test } from '@playwright/test';
import { clickSearch, presentSearchStage } from './support/search-recovery.js';

const withoutModel = { ...process.env, OPENAI_API_KEY: '' };

test.setTimeout(55_000);

const recoveryTest = test.extend({
  modelFixture: async ({}, use) => {
    const requests = [];
    const server = createServer(async (request, response) => {
      if (request.method !== 'POST' || request.url !== '/v1/responses') {
        response.writeHead(404).end();
        return;
      }
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks));
      const input = JSON.parse(body.input.find(({ role }) => role === 'user').content);
      requests.push(input);
      const match = input.candidates.find(({ scope }) => scope === input.reference.scope);
      response.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
        id: 'resp_search_fixture',
        object: 'response',
        status: 'completed',
        output: [{
          id: 'msg_search_fixture', type: 'message', role: 'assistant', status: 'completed',
          content: [{ type: 'output_text', text: JSON.stringify({ candidate_id: match?.candidate_id ?? null }), annotations: [] }],
        }],
      }));
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    try {
      await use({ requests, env: {
        ...process.env, OPENAI_API_KEY: 'test-fixture-key',
        OPENAI_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`,
        RECOVERY_MODEL_MODE: 'mock',
      } });
    } finally {
      await new Promise((resolve) => server.close(resolve));
    }
  },
});

async function findWirelessMouse(page, testInfo, env) {
  await page.locator('#search-input').fill('Wireless Mouse');
  const outcome = await clickSearch(page, testInfo, env);
  const results = page.getByRole('region', { name: 'Shop all gear' });
  await expect(results.getByRole('status')).toHaveText('2 products');
  await results.getByRole('link', { name: 'Wireless Mouse', exact: true }).click();
  return outcome;
}

async function expectWirelessMouseSku(page, testInfo) {
  const sku = page.getByText('SKU', { exact: true }).locator('..').locator('dd');
  const expected = 'FD-M01';
  let passed = false;
  try {
    await expect(sku).toHaveText(expected, { timeout: 1000 });
    passed = true;
  } finally {
    if (process.env.RECOVERY_DEMO_DIR) {
      const actual = await sku.textContent({ timeout: 500 }).catch(() => null);
      await testInfo.attach('sku-assertion', {
        body: JSON.stringify({ expected, actual, passed }), contentType: 'application/json',
      });
      await presentSearchStage(page, testInfo,
        passed ? `Playwright verified SKU ${expected}.` : `Regression caught. Expected ${expected}, received ${actual ?? 'no SKU'}.`);
    }
  }
}

test('Release A completes the search with the original locator', async ({ page }, testInfo) => {
  await page.goto('/products');

  expect(await findWirelessMouse(page, testInfo, withoutModel)).toBe('original');
  await expectWirelessMouseSku(page, testInfo);
});

test('DQN recovers a supported Search control without model access', async ({ page }, testInfo) => {
  await page.goto('/products?storefront-fixture=search-release-b');
  const help = page.getByRole('search', { name: 'Help center' });
  await help.evaluate((form) => { form.style.display = 'none'; });
  await expect(help).toBeHidden();
  await expect(page.locator('#search-btn')).toHaveCount(0);

  expect(await findWirelessMouse(page, testInfo, withoutModel)).toBe('dqn');
  await expectWirelessMouseSku(page, testInfo);
});

recoveryTest('Release B sends two Search buttons to the model tiebreaker', async ({ page, modelFixture }, testInfo) => {
  await page.goto('/products?storefront-fixture=search-release-b');
  await expect(page.locator('#search-btn')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Search', exact: true })).toHaveCount(2);

  expect(await findWirelessMouse(page, testInfo, modelFixture.env)).toBe('llm');
  expect(modelFixture.requests).toHaveLength(1);
  expect(modelFixture.requests[0].candidates.map(({ scope }) => scope)).toEqual(
    expect.arrayContaining(['Catalog', 'Help center']),
  );
  expect(JSON.stringify(modelFixture.requests)).not.toMatch(/Wireless Mouse|semantic.role|expected_role/);
  await expectWirelessMouseSku(page, testInfo);
});

test('Ambiguous Search remains unresolved without a model', async ({ page }, testInfo) => {
  await page.goto('/products?storefront-fixture=search-release-b');
  await page.locator('#search-input').fill('Wireless Mouse');

  await expect(clickSearch(page, testInfo, withoutModel)).rejects.toThrow('model_unconfigured');
  await expect(page.getByRole('status')).toHaveText('6 products');
});

recoveryTest('A recovered search still fails the assertion when the store opens the wrong SKU', async ({ page, modelFixture }, testInfo) => {
  await page.goto('/products?storefront-fixture=search-wrong-product');

  expect(await findWirelessMouse(page, testInfo, modelFixture.env)).toBe('llm');
  await expect(page.getByRole('heading', { name: 'Wireless Mouse Pro' })).toBeVisible();
  await expect(page.getByText('FD-M02', { exact: true })).toBeVisible();
  test.fail(true, 'The wrong SKU must fail the original product assertion.');
  await expectWirelessMouseSku(page, testInfo);
});

test('Release B recovers with the live model', async ({ page }, testInfo) => {
  test.skip(process.env.RECOVERY_LIVE_MODEL !== '1', 'Enable to verify the live API integration');
  expect(process.env.OPENAI_API_KEY, 'The live model run requires OPENAI_API_KEY').toBeTruthy();
  await page.goto('/products?storefront-fixture=search-release-b');

  expect(await findWirelessMouse(page, testInfo, process.env)).toBe('llm');
  await expectWirelessMouseSku(page, testInfo);
});
