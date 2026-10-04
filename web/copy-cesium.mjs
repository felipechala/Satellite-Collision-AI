// Copies Cesium's runtime assets (workers, widgets CSS, imagery) to public/cesium so Vite serves
// them at /cesium/. Runs before `npm run dev` and `npm run build`.
import { cpSync, existsSync } from "node:fs";

const src = "node_modules/cesium/Build/Cesium";
for (const dir of ["Assets", "ThirdParty", "Widgets", "Workers"]) {
  if (!existsSync(`public/cesium/${dir}`)) cpSync(`${src}/${dir}`, `public/cesium/${dir}`, { recursive: true });
}
