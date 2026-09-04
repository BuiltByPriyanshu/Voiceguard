import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/analyze": "http://127.0.0.1:8000",
      "/config": "http://127.0.0.1:8000",
      "/stream": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
