import { test, expect } from '@playwright/test';

test.describe('Speech Inference Tests', () => {
  test.setTimeout(120000); // 2 minute timeout for speech tests (transcription can be slow)

  test('speech_test_inference - upload audio and transcribe', async ({ page }) => {
    // Navigate to the listen page
    await page.goto('http://localhost:4223/listen');

    // Wait for the page to load and backend to be connected
    await page.waitForSelector('.listen-container', { timeout: 30000 });

    // Wait for backend connection (should show "Connected" status)
    await page.waitForFunction(() => {
      const statusElement = document.querySelector('.connection-indicator');
      return statusElement && statusElement.textContent?.includes('Connected');
    }, { timeout: 30000 });

    // Select Italian language
    const italianRadio = page.locator('input[name="language"][value="it"]');
    await expect(italianRadio).toBeVisible();
    await italianRadio.check();

    // Wait for language model to load
    await page.waitForTimeout(3000);

    // Upload the test audio file
    const audioPath = '/Users/daniele.bordignon/Projects/custom-train/frontend/test-files/recorded_audio_1765916065374.webm';
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(audioPath);

    // Wait for audio file to be loaded
    await page.waitForTimeout(2000);

    // Wait for audio to be loaded in memory
    const loadAudioButton = page.locator('button').filter({ hasText: 'Load Audio' });
    await expect(loadAudioButton).toBeEnabled();
    await loadAudioButton.click();

    // Click the "Transcribe" button
    const transcribeButton = page.locator('button').filter({ hasText: 'Transcribe' });
    await expect(transcribeButton).toBeEnabled();
    await transcribeButton.click();

    // Wait for transcription to complete by checking that the textarea has content
    const transcriptionTextarea = page.locator('.transcript-input textarea');
    await expect(transcriptionTextarea).toBeVisible();

    // Wait for the transcription text to appear (not empty and not placeholder)
    await page.waitForFunction(() => {
      const textarea = document.querySelector('.transcript-input textarea') as HTMLTextAreaElement;
      const text = textarea?.value || '';
      return text.length > 0 && text !== 'Transcription result will appear here...';
    }, { timeout: 90000 }); // 90 seconds for transcription

    // Get the transcription text
    const transcriptionText = await transcriptionTextarea.inputValue();

    // Verify that we have some transcription result (not empty or placeholder)
    expect(transcriptionText.length).toBeGreaterThan(0);
    expect(transcriptionText).not.toBe('Transcription result will appear here...');

    console.log(`Transcription result: "${transcriptionText}"`);
    console.log(`Transcription length: ${transcriptionText.length} characters`);
  });
});
