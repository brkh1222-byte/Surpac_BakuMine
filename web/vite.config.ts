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
      '/api': API_URL,
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
