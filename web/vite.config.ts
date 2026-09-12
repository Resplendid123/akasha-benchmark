import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 本地开发两进程：Vite dev server (:5173) + FastAPI (:8848)。
// /api 代理到后端，浏览器始终只跟 5173 打交道。
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8848',
        changeOrigin: true,
      },
    },
  },
  build: {
    // build 产物放 web/dist，用于本地预览。生产部署由 nginx/CDN 托管。
    outDir: 'dist',
    sourcemap: true,
  },
})
