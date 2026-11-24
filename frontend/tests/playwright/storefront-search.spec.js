import { test, expect } from '@playwright/test';

test.describe('Storefront Search', () => {
  test('Distinguishes close results and opens the requested SKU', async ({ page }) => {
    await page.goto('/products');
    const search = page.getByRole('search', { name: 'Catalog' });
    await search.getByRole('textbox', { name: 'Search products' }).fill(' wireless mouse ');
    await search.getByRole('button', { name: 'Search', exact: true }).click();

    const results = page.getByRole('region', { name: 'Shop all gear' });
    await expect(results.getByRole('status')).toHaveText('2 products');
    await expect(results.getByRole('article')).toHaveCount(2);
    await expect(results.getByText('SKU FD-M02', { exact: true })).toBeVisible();

    const target = results.getByRole('article').filter({ hasText: 'SKU FD-M01' });
    await target.getByRole('link', { name: 'Wireless Mouse', exact: true }).click();

    await expect(page).toHaveURL(/\/products\/wireless-mouse$/);
    await expect(page.getByRole('heading', { name: 'Wireless Mouse', exact: true })).toBeVisible();
    await expect(page.getByText('FD-M01', { exact: true })).toBeVisible();
  });

  test('Clears an unmatched search and restores the catalog', async ({ page }) => {
    await page.goto('/products');
    const catalog = page.getByRole('region', { name: 'Shop all gear' });
    const input = catalog.getByRole('textbox', { name: 'Search products' });
    await expect(catalog.getByRole('article')).toHaveCount(6);
    await input.fill('teleportation dock');
    await input.press('Enter');

    await expect(catalog.getByRole('status')).toHaveText('0 products');
    await expect(catalog.getByRole('heading', { name: 'No products found' })).toBeVisible();
    await expect(catalog.getByRole('article')).toHaveCount(0);

    await catalog.getByRole('button', { name: 'Clear search and filters' }).click();

    await expect(input).toHaveValue('');
    await expect(catalog.getByRole('article')).toHaveCount(6);
    await expect(catalog.getByRole('heading', { name: 'No products found' })).toHaveCount(0);
  });
});
