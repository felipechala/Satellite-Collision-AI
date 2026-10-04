import { defineConfig } from "vite";

// The demo talks to the API at VITE_API_URL (default http://127.0.0.1:8000), which allows this origin.
export default defineConfig({
  server: { port: 5173, strictPort: true },
});
