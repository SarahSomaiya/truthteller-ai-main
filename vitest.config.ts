import { defineConfig } from "vite"; // ✅ MUST be vite
import react from "@vitejs/plugin-react-swc";
import path from "path";

export default defineConfig({
  base: "/truthteller-ai-main/", // ✅ THIS FIXES PATHS
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
});
