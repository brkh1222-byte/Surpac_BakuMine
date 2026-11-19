import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The API (uvicorn) runs on port 8000; the dev server proxies /api to it.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
