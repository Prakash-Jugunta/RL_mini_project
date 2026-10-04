import { test, expect } from '@playwright/test';

test.describe('Workflow 4 — Checkout & Cart Baseline Tests', () => {
  test('Complete End-to-End Checkout Workflow', async ({ page }) => {
    // 1. Open product page
    await page.goto('/products/wireless-mouse');

    // 2. Add product to cart
    await page.locator('#add-to-cart-btn').click();

    // 3. Navigate to cart using navbar cart button
    await page.locator('#cart-btn').click();

    // 4. Click checkout on cart page
    await page.locator('#checkout-btn').click();

    // 5. Confirm order on checkout page
    await page.locator('#confirm-order-btn').click();

    // 6. Validate order success page and success element
    await expect(page).toHaveURL(/\/order-success/);
    const orderSuccessBanner = page.locator('#order-success');
    await expect(orderSuccessBanner).toBeVisible();
    await expect(orderSuccessBanner).toContainText('Order placed successfully');
  });

  test('Empty Cart Protection — Direct Access Redirects to Cart', async ({ page }) => {
    // 1. Fresh page with empty cart - navigate directly to /checkout
    await page.goto('/checkout');

    // 2. Assert immediate redirect to /cart
    await expect(page).toHaveURL(/\/cart/);
    await expect(page).not.toHaveURL(/\/order-success/);
  });
});
