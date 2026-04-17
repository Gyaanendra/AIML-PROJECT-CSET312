import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/analyze': 'http://localhost:8000',
      '/analyze_stream': 'http://localhost:8000'
    }
  }
})
