import os from 'node:os'
import path from 'node:path'
import { defineConfig, devices } from '@playwright/test'

/**
 * README screenshots (`make screenshots`): e2e/readme-screenshots.ts builds the synthetic
 * model through the sidebar panels on a fresh project and saves PNGs to docs/screenshots/.
 * Ports of its own (overridable), so it runs beside `make dev` and `make e2e`.
 */
const API_PORT = Number(process.env.BAKUMINE_SHOTS_API_PORT ?? 8197)
const WEB_PORT = Number(process.env.BAKUMINE_SHOTS_WEB_PORT ?? 5197)
const API_URL = `http://127.0.0.1:${API_PORT}`
const SHOTS_PROJECT = path.join(os.tmpdir(), 'bakumine-readme-project')

export default defineConfig({
  testDir: './e2e',
  testMatch: 'readme-screenshots.ts',
  outputDir: './test-results',
  workers: 1,
  reporter: [['list']],
  timeout: 10 * 60_000,
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 1000 },
        launchOptions: { args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=default'] },
      },
    },
  ],
  webServer: [
    {
      // A fresh project with the synthetic ore solid and topography (no dialog imports meshes).
      command:
        `sh -c 'rm -rf "$P" && uv run bakumine project init "$P" --name "Synthetic walkthrough"` +
        ` && uv run bakumine project add "$P" solid data/synthetic/default/ore_solid.obj --name "Ore solid" --id ore-solid` +
        ` && uv run bakumine project add "$P" dtm data/synthetic/default/topo.obj --name Topography --id topo` +
        ` && uv run uvicorn bakumine.api:app --host 127.0.0.1 --port ${API_PORT}'`,
      cwd: '..',
      url: `${API_URL}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: { P: SHOTS_PROJECT, BAKUMINE_PROJECT: SHOTS_PROJECT },
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { BAKUMINE_API_URL: API_URL },
    },
  ],
})
