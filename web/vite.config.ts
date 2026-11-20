import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The API (uvicorn) runs on port 8000; the dev server proxies /api to it.
// BAKUMINE_API_URL points the proxy elsewhere (the Playwright run uses its own port).
const API_URL = process.env.BAKUMINE_API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  // Binary test fixtures (src/api/fixtures) are imported with `?inline`.
  assetsInclude: ['**/*.bkma'],
  server: {
    proxy: {
      // Keep the browser's Host header: the API refuses state-changing requests whose
      // Origin is not its Host (bakumine/api/guard.py), and the page's origin is this server.
      '/api': { target: API_URL, changeOrigin: false },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
