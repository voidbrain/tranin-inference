import { test, expect } from '@playwright/test';

test.describe('Vision Inference Tests', () => {
  test.setTimeout(60000); // 60 second timeout for inference tests

  test('vision_test_inference - upload image and detect objects', async ({ page }) => {
    // Navigate to the see page
    await page.goto('http://localhost:4223/see');

    // Wait for the page to load and backend to be connected
    await page.waitForSelector('.see-container', { timeout: 30000 });

    // Wait for backend connection (should show "Connected" status) - reduced timeout
    try {
      await page.waitForFunction(() => {
        const statusElement = document.querySelector('.connection-indicator');
        return statusElement && statusElement.textContent?.includes('Connected');
      }, { timeout: 10000 });
    } catch (error) {
      console.log('Backend connection check timed out, proceeding anyway...');
      // Continue with test even if connection status doesn't show as connected
    }

    // Select "Digits LoRA (base + digit)" model
    const digitsModelButton = page.locator('button.model-btn').filter({ hasText: 'Digits LoRA (base + digit)' });
    await expect(digitsModelButton).toBeVisible();
    await digitsModelButton.click();

    // Wait for model to be selected and loaded
    await page.waitForTimeout(2000);

    // Set detection mode to digits
    const digitsRadio = page.locator('input[name="detectionMode"][value="digits"]');
    await digitsRadio.check();

    const imagePath = '/Users/daniele.bordignon/Projects/custom-train/frontend/test-files/IMG_2451.png';
    const fileInput = page.locator('input[type="file"]#imageUpload');
    await fileInput.setInputFiles(imagePath);

    // Wait for image to be loaded and displayed
    await page.waitForSelector('img.captured-image', { timeout: 10000 });

    // Click "Detect Objects" button
    const detectButton = page.locator('button').filter({ hasText: 'Detect Objects' });
    await expect(detectButton).toBeEnabled();
    await detectButton.click();

    // Wait for detection to complete by waiting for the button to become enabled again
    await page.waitForFunction(() => {
      const button = document.querySelector('button') as HTMLButtonElement;
      return button && button.textContent?.includes('Detect Objects') && !button.disabled;
    }, { timeout: 30000 });

    // Check that we have at least one detection result
    const detectionsList = page.locator('.detections-list');
    const detectionItems = detectionsList.locator('.detection-item');

    // Expect at least one detection
    await expect(detectionItems.first()).toBeVisible();

    // Get the number of detections
    const detectionCount = await detectionItems.count();
    console.log(`Found ${detectionCount} detections`);

    // Ensure we have at least one detection (the test requirement)
    expect(detectionCount).toBeGreaterThan(0);

    // Verify the detections contain expected elements
    for (let i = 0; i < Math.min(detectionCount, 3); i++) {
      const detectionItem = detectionItems.nth(i);
      await expect(detectionItem).toBeVisible();

      // Check that each detection has a label and confidence
      const label = detectionItem.locator('.detection-label');
      const confidence = detectionItem.locator('.detection-confidence');

      await expect(label).toBeVisible();
      await expect(confidence).toBeVisible();
      await expect(label).not.toBeEmpty();
      await expect(confidence).not.toBeEmpty();
    }
  });
});
