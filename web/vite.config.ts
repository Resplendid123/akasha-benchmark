import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发时两个进程：这个 dev server 加 FastAPI 后端。/api 代理过去，
// 这样前端代码里的请求路径在开发和生产下完全一样（生产是同源）。
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
    // 后端按 web/dist 挂 StaticFiles，见 akasha_platform/main.py。
    outDir: 'dist',
    sourcemap: true,
  },
})
