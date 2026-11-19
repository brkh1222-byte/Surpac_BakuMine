import { defineConfig, devices } from '@playwright/test'

// A port of its own, so the tests never talk to a dev server someone left running.
const PORT = 5199
// The API the tests talk to: its own uvicorn, serving the demo project.
const API_PORT = Number(process.env.BAKUMINE_E2E_API_PORT ?? 8199)
const API_URL = `http://127.0.0.1:${API_PORT}`

export default defineConfig({
  testDir: './e2e',
  outputDir: './test-results',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  reporter: [['list']],
  timeout: 60_000,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 800 },
        // Ask headless Chromium for the real GPU where there is one (the performance
        // probe is meaningless on the software rasteriser); it falls back by itself.
        launchOptions: { args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=default'] },
      },
    },
  ],
  webServer: [
    {
      command: `uv run uvicorn bakumine.api:app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: '..',
      url: `${API_URL}/api/health`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${PORT} --strictPort`,
      url: `http://127.0.0.1:${PORT}`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { BAKUMINE_API_URL: API_URL },
    },
  ],
})
