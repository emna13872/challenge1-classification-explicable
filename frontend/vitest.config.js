import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: { environment: 'jsdom', include: ['tests/**/*.test.jsx'], env: { VITE_API_URL: 'http://localhost:8000' } },
})
