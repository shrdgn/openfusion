import path from "path";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/playground/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: { proxy: { "/v1": { target: "http://localhost:8000", changeOrigin: true } } },
  build: {
    outDir: path.resolve(__dirname, "../openfusion/static/playground"),
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/lib/__tests__/setup.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      // main.tsx is the React bootstrap entrypoint (render() call); nothing to unit-test.
      exclude: ["src/main.tsx"],
      thresholds: {
        statements: 90,
        branches: 70,
        functions: 85,
        lines: 90,
      },
    },
  },
});
