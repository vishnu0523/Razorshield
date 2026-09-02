import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Lets the dashboard call /api/* in dev without CORS juggling.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
