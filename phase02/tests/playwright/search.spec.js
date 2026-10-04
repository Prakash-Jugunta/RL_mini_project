import { test, expect } from '@playwright/test';

test.describe('Workflow 2 — Product Search Baseline Tests', () => {
  test('Search and Select Product Workflow', async ({ page }) => {
    // 1. Navigate to catalog page
    await page.goto('/products');

    // 2. Use exact brittle Phase 1 selectors
    await page.locator('#search-input').fill('Wireless Mouse');
    await page.locator('#search-btn').click();

    // 3. Click the specific product result link
    await page.locator('#product-result-wireless-mouse').click();

    // 4. Validate success conditions
    await expect(page).toHaveURL(/\/products\/wireless-mouse/);
    await expect(page.getByText('Wireless Mouse')).toBeVisible();
  });
});
