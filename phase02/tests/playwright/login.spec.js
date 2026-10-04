import { test, expect } from '@playwright/test';

test.describe('Workflow 1 — Login Baseline Tests', () => {
  test('Successful Login Workflow with Valid Credentials', async ({ page }) => {
    // 1. Navigate to login route
    await page.goto('/login');

    // 2. Use exact brittle Phase 1 selectors
    await page.locator('#username').fill('testuser');
    await page.locator('#password').fill('password123');
    await page.locator('#login-btn').click();

    // 3. Validate success conditions
    await expect(page).toHaveURL(/\/dashboard/);
    await expect(page.getByText('Welcome, testuser')).toBeVisible();
  });

  test('Negative Test — Invalid Credentials', async ({ page }) => {
    await page.goto('/login');

    await page.locator('#username').fill('invaliduser');
    await page.locator('#password').fill('wrongpassword');
    await page.locator('#login-btn').click();

    // Assert remaining on login page with error message
    await expect(page).toHaveURL(/\/login/);
    await expect(page.getByText('Invalid username or password')).toBeVisible();
  });
});
