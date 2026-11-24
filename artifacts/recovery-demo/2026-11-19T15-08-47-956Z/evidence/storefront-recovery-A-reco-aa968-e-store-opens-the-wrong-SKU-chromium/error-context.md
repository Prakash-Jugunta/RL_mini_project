# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: storefront-recovery.spec.js >> A recovered search still fails the assertion when the store opens the wrong SKU
- Location: tests\playwright\storefront-recovery.spec.js:116:1

# Error details

```
Error: expect(locator).toHaveText(expected) failed

Locator:  getByText('SKU', { exact: true }).locator('..').locator('dd')
Expected: "FD-M01"
Received: "FD-M02"
Timeout:  1000ms

Call log:
  - Expect "toHaveText" getByText('SKU', { exact: true }).locator('..').locator('dd') with timeout 1000ms
  - waiting for getByText('SKU', { exact: true }).locator('..').locator('dd')
    11 × locator resolved to <dd>FD-M02</dd>
       - unexpected value "FD-M02"

```

```yaml
- definition: FD-M02
```

# Test source

```ts
  1   | import { createServer } from 'node:http';
  2   | import { once } from 'node:events';
  3   | import { expect, test } from '@playwright/test';
  4   | import { clickSearch, presentSearchStage } from './support/search-recovery.js';
  5   | 
  6   | const withoutModel = { ...process.env, OPENAI_API_KEY: '' };
  7   | 
  8   | test.setTimeout(55_000);
  9   | 
  10  | const recoveryTest = test.extend({
  11  |   modelFixture: async ({}, use) => {
  12  |     const requests = [];
  13  |     const server = createServer(async (request, response) => {
  14  |       if (request.method !== 'POST' || request.url !== '/v1/responses') {
  15  |         response.writeHead(404).end();
  16  |         return;
  17  |       }
  18  |       const chunks = [];
  19  |       for await (const chunk of request) chunks.push(chunk);
  20  |       const body = JSON.parse(Buffer.concat(chunks));
  21  |       const input = JSON.parse(body.input.find(({ role }) => role === 'user').content);
  22  |       requests.push(input);
  23  |       const match = input.candidates.find(({ scope }) => scope === input.reference.scope);
  24  |       response.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
  25  |         id: 'resp_search_fixture',
  26  |         object: 'response',
  27  |         status: 'completed',
  28  |         output: [{
  29  |           id: 'msg_search_fixture', type: 'message', role: 'assistant', status: 'completed',
  30  |           content: [{ type: 'output_text', text: JSON.stringify({ candidate_id: match?.candidate_id ?? null }), annotations: [] }],
  31  |         }],
  32  |       }));
  33  |     });
  34  |     server.listen(0, '127.0.0.1');
  35  |     await once(server, 'listening');
  36  |     try {
  37  |       await use({ requests, env: {
  38  |         ...process.env, OPENAI_API_KEY: 'test-fixture-key',
  39  |         OPENAI_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`,
  40  |         RECOVERY_MODEL_MODE: 'mock',
  41  |       } });
  42  |     } finally {
  43  |       await new Promise((resolve) => server.close(resolve));
  44  |     }
  45  |   },
  46  | });
  47  | 
  48  | async function findWirelessMouse(page, testInfo, env) {
  49  |   await page.locator('#search-input').fill('Wireless Mouse');
  50  |   const outcome = await clickSearch(page, testInfo, env);
  51  |   const results = page.getByRole('region', { name: 'Shop all gear' });
  52  |   await expect(results.getByRole('status')).toHaveText('2 products');
  53  |   await results.getByRole('link', { name: 'Wireless Mouse', exact: true }).click();
  54  |   return outcome;
  55  | }
  56  | 
  57  | async function expectWirelessMouseSku(page, testInfo) {
  58  |   const sku = page.getByText('SKU', { exact: true }).locator('..').locator('dd');
  59  |   const expected = 'FD-M01';
  60  |   let passed = false;
  61  |   try {
> 62  |     await expect(sku).toHaveText(expected, { timeout: 1000 });
      |                       ^ Error: expect(locator).toHaveText(expected) failed
  63  |     passed = true;
  64  |   } finally {
  65  |     if (process.env.RECOVERY_DEMO_DIR) {
  66  |       const actual = await sku.textContent({ timeout: 500 }).catch(() => null);
  67  |       await testInfo.attach('sku-assertion', {
  68  |         body: JSON.stringify({ expected, actual, passed }), contentType: 'application/json',
  69  |       });
  70  |       await presentSearchStage(page, testInfo,
  71  |         passed ? `Playwright verified SKU ${expected}.` : `Regression caught. Expected ${expected}, received ${actual ?? 'no SKU'}.`);
  72  |     }
  73  |   }
  74  | }
  75  | 
  76  | test('Release A completes the search with the original locator', async ({ page }, testInfo) => {
  77  |   await page.goto('/products');
  78  | 
  79  |   expect(await findWirelessMouse(page, testInfo, withoutModel)).toBe('original');
  80  |   await expectWirelessMouseSku(page, testInfo);
  81  | });
  82  | 
  83  | test('DQN recovers a supported Search control without model access', async ({ page }, testInfo) => {
  84  |   await page.goto('/products?storefront-fixture=search-release-b');
  85  |   const help = page.getByRole('search', { name: 'Help center' });
  86  |   await help.evaluate((form) => { form.style.display = 'none'; });
  87  |   await expect(help).toBeHidden();
  88  |   await expect(page.locator('#search-btn')).toHaveCount(0);
  89  | 
  90  |   expect(await findWirelessMouse(page, testInfo, withoutModel)).toBe('dqn');
  91  |   await expectWirelessMouseSku(page, testInfo);
  92  | });
  93  | 
  94  | recoveryTest('Release B sends two Search buttons to the model tiebreaker', async ({ page, modelFixture }, testInfo) => {
  95  |   await page.goto('/products?storefront-fixture=search-release-b');
  96  |   await expect(page.locator('#search-btn')).toHaveCount(0);
  97  |   await expect(page.getByRole('button', { name: 'Search', exact: true })).toHaveCount(2);
  98  | 
  99  |   expect(await findWirelessMouse(page, testInfo, modelFixture.env)).toBe('llm');
  100 |   expect(modelFixture.requests).toHaveLength(1);
  101 |   expect(modelFixture.requests[0].candidates.map(({ scope }) => scope)).toEqual(
  102 |     expect.arrayContaining(['Catalog', 'Help center']),
  103 |   );
  104 |   expect(JSON.stringify(modelFixture.requests)).not.toMatch(/Wireless Mouse|semantic.role|expected_role/);
  105 |   await expectWirelessMouseSku(page, testInfo);
  106 | });
  107 | 
  108 | test('Ambiguous Search remains unresolved without a model', async ({ page }, testInfo) => {
  109 |   await page.goto('/products?storefront-fixture=search-release-b');
  110 |   await page.locator('#search-input').fill('Wireless Mouse');
  111 | 
  112 |   await expect(clickSearch(page, testInfo, withoutModel)).rejects.toThrow('model_unconfigured');
  113 |   await expect(page.getByRole('status')).toHaveText('6 products');
  114 | });
  115 | 
  116 | recoveryTest('A recovered search still fails the assertion when the store opens the wrong SKU', async ({ page, modelFixture }, testInfo) => {
  117 |   await page.goto('/products?storefront-fixture=search-wrong-product');
  118 | 
  119 |   expect(await findWirelessMouse(page, testInfo, modelFixture.env)).toBe('llm');
  120 |   await expect(page.getByRole('heading', { name: 'Wireless Mouse Pro' })).toBeVisible();
  121 |   await expect(page.getByText('FD-M02', { exact: true })).toBeVisible();
  122 |   test.fail(true, 'The wrong SKU must fail the original product assertion.');
  123 |   await expectWirelessMouseSku(page, testInfo);
  124 | });
  125 | 
  126 | test('Release B recovers with the live model', async ({ page }, testInfo) => {
  127 |   test.skip(process.env.RECOVERY_LIVE_MODEL !== '1', 'Enable to verify the live API integration');
  128 |   expect(process.env.OPENAI_API_KEY, 'The live model run requires OPENAI_API_KEY').toBeTruthy();
  129 |   await page.goto('/products?storefront-fixture=search-release-b');
  130 | 
  131 |   expect(await findWirelessMouse(page, testInfo, process.env)).toBe('llm');
  132 |   await expectWirelessMouseSku(page, testInfo);
  133 | });
  134 | 
```