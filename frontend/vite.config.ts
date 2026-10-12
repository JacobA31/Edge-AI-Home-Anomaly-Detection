import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // During `npm run dev`, /api/* goes to the backend running with `func start`
    proxy: { "/api": "http://localhost:7071" },
  },
});
