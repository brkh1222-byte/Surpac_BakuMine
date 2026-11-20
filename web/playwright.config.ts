import os from 'node:os'
import path from 'node:path'
import { defineConfig, devices } from '@playwright/test'

// A port of its own, so the tests never talk to a dev server someone left running.
const PORT = 5199
// The API the tests talk to: its own uvicorn, serving the demo project.
const API_PORT = Number(process.env.BAKUMINE_E2E_API_PORT ?? 8199)
const API_URL = `http://127.0.0.1:${API_PORT}`
// A second API on a fresh, empty project (e2e/workflow.spec.ts builds a model in it), with
// its own dev server.
const WORK_API_PORT = Number(process.env.BAKUMINE_E2E_WORK_API_PORT ?? 8198)
const WORK_WEB_PORT = 5198
const WORK_API_URL = `http://127.0.0.1:${WORK_API_PORT}`
const WORK_PROJECT = path.join(os.tmpdir(), 'bakumine-e2e-workflow-project')

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
    {
      command: `sh -c 'rm -rf "$WORK_PROJECT" && uv run bakumine project init "$WORK_PROJECT" --name Workflow && uv run uvicorn bakumine.api:app --host 127.0.0.1 --port ${WORK_API_PORT}'`,
      cwd: '..',
      url: `${WORK_API_URL}/api/health`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { WORK_PROJECT, BAKUMINE_PROJECT: WORK_PROJECT },
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${WORK_WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WORK_WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { BAKUMINE_API_URL: WORK_API_URL },
    },
  ],
})
