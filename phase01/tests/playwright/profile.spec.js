import { test, expect } from '@playwright/test';

test.describe('Workflow 3 — Profile Update Baseline Tests', () => {
  test('Update Profile Details Workflow', async ({ page }) => {
    // 1. Navigate to profile page
    await page.goto('/profile');

    // 2. Fill inputs using exact Phase 1 selectors
    await page.locator('#profile-name').fill('Test User');
    await page.locator('#profile-email').fill('test@example.com');
    await page.locator('#save-profile-btn').click();

    // 3. Validate success element and text
    const successBanner = page.locator('#profile-success');
    await expect(successBanner).toBeVisible();
    await expect(successBanner).toContainText('Profile updated successfully');
  });
});
