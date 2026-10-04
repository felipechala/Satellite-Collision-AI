// One-user demo: submit a breakup to the API, wait for the worker, animate the stored cloud.
// All data comes through the API (which stores and reads it in SpacetimeDB):
//   POST /v1/simulations, GET /v1/simulations/{id}, .../particles (orbits), .../frames (density voxels).
<<<<<<< HEAD
=======
// The top-right card shows a Grok Imagine illustration of the target satellite (POST /v1/satellite-image),
// generated from the same scenario parameters; it is decoration, not model output.
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
// The cloud is drawn as a probability density, not as fragments: each stored representative particle
// (a weighted sample standing for many real fragments) is a soft Gaussian blob, and overlapping blobs
// add up to a continuous cloud (a kernel density estimate drawn live). Every render frame evaluates the
// samples' orbits at the current time (same math as cloud_predictor engine.propagator.orbit_positions),
// so the cloud moves as smoothly as the display. The ensemble's hourly voxel frames (mean/p10/p90
// across Monte Carlo runs) are an optional uncertainty overlay.
import "./style.css";
import {
  BillboardCollection,
  BlendOption,
  Billboard,
  Cartesian3,
  Color,
  ImageryLayer,
  JulianDate,
  Matrix3,
  Matrix4,
  PointPrimitiveCollection,
  TileMapServiceImageryProvider,
  Transforms,
  Viewer,
  buildModuleUrl,
} from "cesium";
<<<<<<< HEAD
=======
import { densityAtSamples } from "./density";
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
import { type Orbits, aliveAt, positionAt } from "./orbit";

const API: string = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
const POLL_MS = 1000;
<<<<<<< HEAD
// Probability cloud: one warm hue; denser = more opaque. Ensemble overlay: sequential orange ramp.
const CLOUD_COLOR = Color.fromCssColorString("#f4a582");
const DENSITY_RAMP = ["#5c2410", "#9a3d1c", "#d95926", "#eb6834", "#f4a582"];
// Blob diameter: samples sit ~60-150 km apart (nearest to 8th-nearest neighbour, 0.5-48 h), so
// 300 km blobs merge into a continuous field at every time.
const BLOB_M = 300_000;
// Blob opacity ~ sqrt(weight): weights span ~1-500 real fragments per sample; linear opacity would
// let the smallest (heaviest-weighted) size bins hide the rest. Still monotonic in weight.
const ALPHA_MIN = 0.03, ALPHA_MAX = 0.22;
=======
// Expected-fragment density: one warm hue, dark -> near-white, interpolated (sequential, log scale).
// Ensemble overlay: a blue sequential ramp, so the two layers are never confused.
const CLOUD_RAMP = ["#3d1607", "#9a3d1c", "#eb6834", "#f4a582", "#fde4d6"];
const ENSEMBLE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"];
// Blob diameter: samples sit ~60-150 km apart (nearest to 8th-nearest neighbour, 0.5-48 h), so
// 400 km blobs blend into one continuous field at every time.
const BLOB_M = 400_000;
// Density cells for colouring [km] (density.ts; same method as the API's voxel product).
const CELL_KM = 100;
// Recompute densities every few render frames: samples move with their neighbours, so a sample's
// density changes slowly. Scrubbing or loading recomputes immediately.
const DENSITY_EVERY_FRAMES = 6;
// Colour scale: percentiles of log10 density over these times, fixed per simulation.
const SCALE_TIMES_H = [1, 6, 24, 48];
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb

type Voxel = [x: number, y: number, z: number, mean: number, p10: number, p50: number, p90: number];
interface Frame { frame: number; t_s: number; fragments_in_orbit: number; voxels: Voxel[] }
interface Simulation {
  id: number; status: string; progress: number; error: string | null;
<<<<<<< HEAD
  request?: { event?: { epoch?: string } };
=======
  request?: { event?: { epoch?: string }; orbit?: unknown };
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
}
type OrbitColumns = Record<keyof Orbits, number[]>;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("scenario");
const runButton = $<HTMLButtonElement>("run");
const statusEl = $("status");
const progressEl = $("progress");
const progressBar = $("progress-bar");
const playbackEl = $("playback");
const slider = $<HTMLInputElement>("slider");
const playButton = $<HTMLButtonElement>("play");
const speedSelect = $<HTMLSelectElement>("speed");
const showParticles = $<HTMLInputElement>("show-particles");
const showVoxels = $<HTMLInputElement>("show-voxels");

const viewer = new Viewer("globe", {
  baseLayer: ImageryLayer.fromProviderAsync(
    TileMapServiceImageryProvider.fromUrl(buildModuleUrl("Assets/Textures/NaturalEarthII")),
  ),
  animation: false, timeline: false, baseLayerPicker: false, geocoder: false, homeButton: false,
  sceneModePicker: false, navigationHelpButton: false, fullscreenButton: false, infoBox: false,
  selectionIndicator: false,
});
viewer.camera.setView({ destination: Cartesian3.fromDegrees(10, 20, 30_000_000) });
viewer.scene.debugShowFramesPerSecond = new URLSearchParams(location.search).has("fps");

// Positions stay in ECI (meters); one model matrix per collection rotates them to Earth-fixed.
const cloudLayer = viewer.scene.primitives.add(
  new BillboardCollection({ blendOption: BlendOption.TRANSLUCENT }),
) as BillboardCollection;

/** A white radial Gaussian (to 2.5 sigma at the edge), tinted per blob through Billboard.color. */
function gaussianTexture(px = 64): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = px;
  const ctx = c.getContext("2d")!;
  const g = ctx.createRadialGradient(px / 2, px / 2, 0, px / 2, px / 2, px / 2);
  for (let s = 0; s <= 1.0001; s += 0.1) g.addColorStop(Math.min(s, 1), `rgba(255,255,255,${Math.exp(-0.5 * (2.5 * s) ** 2)})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, px, px);
  return c;
}
const BLOB_IMAGE = gaussianTexture();
let voxelLayers: PointPrimitiveCollection[] = [];

// ---------- simulation state ----------

let orbits: Orbits | null = null;
let blobs: Billboard[] = [];
<<<<<<< HEAD
=======
let posKm = new Float64Array(0); // ECI positions of the samples at the current time [km], 3 per sample
let aliveNow = new Uint8Array(0);
let density = new Float64Array(0); // expected fragments / km^3 around each sample
let cloudScale = { lo: 0, hi: 1 };   // log10 density range of the colour scale
let refreshDensity = true;
let framesSinceDensity = 0;
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
let frames: Frame[] = [];
let epoch = JulianDate.now();
let tEnd = 0;
let tSim = 0;
let playing = false;
let metric = 3; // voxel column: 3 mean, 4 p10, 6 p90
let densityScale = { lo: 0, hi: 1 };
let dirty = true;

function setStatus(text: string, error = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", error);
}

function setProgress(fraction: number | null) {
  progressEl.hidden = fraction === null;
  progressBar.style.width = `${Math.round((fraction ?? 0) * 100)}%`;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, init);
  const body = r.status === 204 ? null : await r.json();
  if (!r.ok) {
    const detail = body?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => `${d.loc}: ${d.msg}`).join("; ") : String(detail));
  }
  return body as T;
}

<<<<<<< HEAD
=======
// ---------- satellite illustration (top right) ----------

const satCard = $("sat-card");
const satImage = $<HTMLImageElement>("sat-image");
const satPlaceholder = $("sat-placeholder");
const satStatus = $("sat-status");
let satRequest = 0; // only the newest request may update the card

async function showSatellite(scenario: { event?: unknown; orbit?: unknown } | undefined) {
  if (!scenario?.event) return;
  const mine = ++satRequest;
  satCard.hidden = false;
  satImage.hidden = true;
  satPlaceholder.hidden = false;
  satPlaceholder.classList.add("loading");
  satStatus.classList.remove("error");
  satStatus.textContent = "Generating with Grok Imagine…";
  try {
    const r = await api<{ image: string; prompt: string }>("/v1/satellite-image", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ event: scenario.event, orbit: scenario.orbit }),
    });
    if (mine !== satRequest) return;
    satImage.src = r.image;
    satImage.alt = r.prompt;
    satImage.title = r.prompt;
    satImage.hidden = false;
    satPlaceholder.hidden = true;
    satStatus.textContent = "";
  } catch (err) {
    if (mine !== satRequest) return;
    satPlaceholder.classList.remove("loading");
    satStatus.classList.add("error");
    satStatus.textContent = `No illustration: ${(err as Error).message}`;
  }
}

>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
// ---------- scenario form ----------

const eventType = form.querySelector<HTMLSelectElement>('select[name="event_type"]')!;
function syncEventType() {
  const collision = eventType.value === "collision";
  form.querySelectorAll<HTMLElement>(".collision-only").forEach((el) => (el.hidden = !collision));
}
eventType.addEventListener("change", syncEventType);
syncEventType();

function requestBody() {
  const f = new FormData(form);
  const num = (k: string) => Number(f.get(k));
  const collision = f.get("event_type") === "collision";
  const event: Record<string, unknown> = {
    event_type: f.get("event_type"),
    epoch: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
    target: { object_class: "payload", dry_mass_kg: num("dry_mass_kg") },
    ...(collision
      ? { impactor: { mass_kg: num("impactor_mass_kg"), v_rel_km_s: num("v_rel_km_s") } }
      : { explosion_cause: "unknown" }),
  };
  return { sat_name: "Demo Sat", event, orbit: { alt_km: num("alt_km"), inc_deg: num("inc_deg") }, n_runs: num("n_runs") };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  runButton.disabled = true;
  setPlaying(false);
<<<<<<< HEAD
  try {
    const sim = await api<{ id: number }>("/v1/simulations", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(requestBody()),
=======
  const body = requestBody();
  void showSatellite(body); // in parallel with the simulation
  try {
    const sim = await api<{ id: number }>("/v1/simulations", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    });
    await follow(sim.id);
  } catch (err) {
    setStatus(`Request failed: ${(err as Error).message}`, true);
    setProgress(null);
  } finally {
    runButton.disabled = false;
  }
});

// ---------- waiting for the worker, loading ----------

async function follow(id: number) {
  for (;;) {
    const sim = await api<Simulation>(`/v1/simulations/${id}`);
    if (sim.status === "done") return load(sim);
    if (sim.status === "failed") throw new Error(sim.error ?? "simulation failed");
    setStatus(sim.status === "queued" ? `Simulation ${id}: queued` : `Simulation ${id}: running Monte Carlo…`);
    setProgress(sim.progress);
    await new Promise((r) => setTimeout(r, POLL_MS));
  }
}

async function load(sim: Simulation) {
  setStatus(`Simulation ${sim.id}: loading…`);
  const [p, f] = await Promise.all([
    api<{ particles: OrbitColumns; t_end_s: number }>(`/v1/simulations/${sim.id}/particles`),
    api<{ frames: Frame[] }>(`/v1/simulations/${sim.id}/frames`),
  ]);
  epoch = JulianDate.fromIso8601(sim.request?.event?.epoch ?? new Date().toISOString());
  frames = f.frames;
  tEnd = p.t_end_s;
  orbits = Object.fromEntries(Object.entries(p.particles).map(([k, v]) => [k, Float64Array.from(v)])) as unknown as Orbits;
  buildCloud();
  buildVoxelLayers();
  setProgress(null);
  setStatus(`Simulation ${sim.id}: ${blobs.length.toLocaleString()} density samples, ${frames.length} ensemble frames`);
  slider.max = String(tEnd);
  tSim = 0;
  slider.value = "0";
  playbackEl.hidden = false;
  dirty = true;
  setPlaying(true);
}

// ---------- probability cloud: weighted samples moving along their orbits ----------

function rampColor(ramp: string[], u: number, alpha: number): Color {
  const i = Math.min(ramp.length - 1, Math.max(0, Math.floor(u * ramp.length)));
  return Color.fromCssColorString(ramp[i]).withAlpha(alpha);
}

<<<<<<< HEAD
function buildCloud() {
  cloudLayer.removeAll();
  const w = orbits!.weight;
  const wMax = w.reduce((m, x) => Math.max(m, x), 0) || 1;
  blobs = Array.from(w, (weight) =>
    cloudLayer.add({
      position: Cartesian3.ZERO, image: BLOB_IMAGE, sizeInMeters: true, width: BLOB_M, height: BLOB_M,
      color: CLOUD_COLOR.withAlpha(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * Math.sqrt(weight / wMax)),
    }),
  );
  $("cloud-ramp").style.background =
    `linear-gradient(to right, ${CLOUD_COLOR.withAlpha(0.05).toCssColorString()}, ${CLOUD_COLOR.toCssColorString()})`;
}

const scratch = new Cartesian3();
/** Move every sample to its position at t (ECI meters); hide re-entered ones. */
function updateCloud(t: number) {
  const o = orbits!;
  for (let i = 0; i < blobs.length; i++) {
    const b = blobs[i];
    b.show = aliveAt(o, i, t);
    if (!b.show) continue;
    positionAt(o, i, t, scratch);
    b.position = scratch; // copied by Cesium
=======
/** 256-step continuous interpolation of a ramp (no visible banding). */
function lut(ramp: string[], steps = 256): Color[] {
  const stops = ramp.map((c) => Color.fromCssColorString(c));
  return Array.from({ length: steps }, (_, k) => {
    const x = (k / (steps - 1)) * (stops.length - 1);
    const j = Math.min(stops.length - 2, Math.floor(x));
    return Color.lerp(stops[j], stops[j + 1], x - j, new Color());
  });
}
const CLOUD_LUT = lut(CLOUD_RAMP);

const fmtDensity = (x: number) => (x < 0.01 || x >= 1000 ? x.toExponential(1) : x.toPrecision(2));

function buildCloud() {
  cloudLayer.removeAll();
  const n = orbits!.weight.length;
  posKm = new Float64Array(3 * n);
  aliveNow = new Uint8Array(n);
  density = new Float64Array(n);
  blobs = Array.from({ length: n }, () =>
    cloudLayer.add({ position: Cartesian3.ZERO, image: BLOB_IMAGE, sizeInMeters: true, width: BLOB_M, height: BLOB_M }),
  );
  // Fix the colour scale for this simulation, so a colour means the same density at every moment.
  const logs: number[] = [];
  for (const h of SCALE_TIMES_H) {
    if (h * 3600 > tEnd) continue;
    placeSamples(h * 3600);
    densityAtSamples(posKm, orbits!.weight, aliveNow, CELL_KM, density);
    for (let i = 0; i < n; i++) if (aliveNow[i] && density[i] > 0) logs.push(Math.log10(density[i]));
  }
  logs.sort((a, b) => a - b);
  const q = (p: number) => logs[Math.min(logs.length - 1, Math.floor(p * logs.length))];
  cloudScale = logs.length ? { lo: q(0.05), hi: Math.max(q(0.995), q(0.05) + 0.5) } : { lo: -6, hi: 0 };
  $("cloud-ramp").style.background = `linear-gradient(to right, ${CLOUD_RAMP.join(", ")})`;
  $("cloud-min").textContent = fmtDensity(10 ** cloudScale.lo);
  $("cloud-mid").textContent = fmtDensity(10 ** ((cloudScale.lo + cloudScale.hi) / 2));
  $("cloud-max").textContent = fmtDensity(10 ** cloudScale.hi);
  refreshDensity = true;
}

const scratch = new Cartesian3();
/** Evaluate every sample's orbit at t into posKm / aliveNow; with move, also update the blobs. */
function placeSamples(t: number, move = false) {
  const o = orbits!;
  for (let i = 0; i < aliveNow.length; i++) {
    const alive = aliveAt(o, i, t);
    aliveNow[i] = alive ? 1 : 0;
    if (move) blobs[i].show = alive;
    if (!alive) continue;
    positionAt(o, i, t, scratch); // meters
    posKm[3 * i] = scratch.x / 1000;
    posKm[3 * i + 1] = scratch.y / 1000;
    posKm[3 * i + 2] = scratch.z / 1000;
    if (move) blobs[i].position = scratch; // copied by Cesium
  }
}

const tint = new Color();
/** Move the samples to t; every few frames, recolour them by the expected fragment density around them. */
function updateCloud(t: number) {
  placeSamples(t, true);
  if (!refreshDensity && ++framesSinceDensity < DENSITY_EVERY_FRAMES) return;
  refreshDensity = false;
  framesSinceDensity = 0;
  densityAtSamples(posKm, orbits!.weight, aliveNow, CELL_KM, density);
  const { lo, hi } = cloudScale;
  for (let i = 0; i < blobs.length; i++) {
    if (!aliveNow[i]) continue;
    const u = density[i] > 0 ? Math.min(1, Math.max(0, (Math.log10(density[i]) - lo) / (hi - lo))) : 0;
    Color.clone(CLOUD_LUT[Math.round(u * 255)], tint);
    tint.alpha = 0.06 + 0.5 * u ** 1.3; // sparse fringe fades out, dense core glows
    blobs[i].color = tint; // copied by Cesium
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
  }
}

// ---------- density hot spots: stored voxel frames ----------

function rescaleDensity() {
  let lo = Infinity, hi = -Infinity;
  for (const f of frames) for (const v of f.voxels) if (v[metric] > 0) {
    const l = Math.log10(v[metric]);
    lo = Math.min(lo, l);
    hi = Math.max(hi, l);
  }
  densityScale = Number.isFinite(lo) ? { lo, hi: hi > lo ? hi : lo + 1 } : { lo: 0, hi: 1 };
<<<<<<< HEAD
  $("density-ramp").style.background = `linear-gradient(to right, ${DENSITY_RAMP.join(", ")})`;
=======
  $("density-ramp").style.background = `linear-gradient(to right, ${ENSEMBLE_RAMP.join(", ")})`;
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
  $("legend-min").textContent = (10 ** densityScale.lo).toPrecision(2);
  $("legend-max").textContent = (10 ** densityScale.hi).toPrecision(2);
}

/** One collection per stored frame, built once; playback only toggles which one is shown. */
function buildVoxelLayers() {
  for (const layer of voxelLayers) viewer.scene.primitives.remove(layer);
  rescaleDensity();
  const { lo, hi } = densityScale;
  voxelLayers = frames.map((f) => {
    const layer = viewer.scene.primitives.add(new PointPrimitiveCollection()) as PointPrimitiveCollection;
    layer.show = false;
    for (const v of f.voxels) {
      if (!(v[metric] > 0)) continue;
      const u = Math.min(1, Math.max(0, (Math.log10(v[metric]) - lo) / (hi - lo)));
      layer.add({ position: new Cartesian3(v[0] * 1000, v[1] * 1000, v[2] * 1000), pixelSize: 4 + 6 * u,
<<<<<<< HEAD
                  color: rampColor(DENSITY_RAMP, u, 0.25 + 0.6 * u) });
=======
                  color: rampColor(ENSEMBLE_RAMP, u, 0.25 + 0.6 * u) });
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    }
    return layer;
  });
}

/** Latest stored frame at or before t (frames are sorted by time). */
function frameIndexAt(t: number): number {
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t_s <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Fragments in orbit at t, interpolated between stored frames (ensemble mean). */
function inOrbitAt(t: number): number {
  const i = frameIndexAt(t);
  const a = frames[i], b = frames[Math.min(i + 1, frames.length - 1)];
  if (b.t_s <= a.t_s) return a.fragments_in_orbit;
  const w = (t - a.t_s) / (b.t_s - a.t_s);
  return a.fragments_in_orbit + w * (b.fragments_in_orbit - a.fragments_in_orbit);
}

// ---------- render loop ----------

const temeToFixed = new Matrix3();
const modelMatrix = new Matrix4();
const frameMatrix = new Matrix4();
const jd = new JulianDate();

/** Rotation from ECI (TEME) to Earth-fixed at t seconds after the breakup, written into out. */
function earthFixedAt(t: number, out: Matrix4): Matrix4 {
  JulianDate.addSeconds(epoch, t, jd);
  Transforms.computeTemeToPseudoFixedMatrix(jd, temeToFixed);
  return Matrix4.fromRotationTranslation(temeToFixed, Cartesian3.ZERO, out);
}

function draw() {
  if (!orbits) return;
  earthFixedAt(tSim, modelMatrix);

  cloudLayer.show = showParticles.checked;
  cloudLayer.modelMatrix = modelMatrix;
  if (showParticles.checked) updateCloud(tSim);

  // A stored frame is the density at its own snapshot time, so it is rotated to Earth-fixed at that
  // time (not the playback time) and stays put until the next snapshot.
  const current = frameIndexAt(tSim);
  voxelLayers.forEach((layer, i) => {
    layer.show = showVoxels.checked && i === current;
    if (layer.show) layer.modelMatrix = earthFixedAt(frames[i].t_s, frameMatrix);
  });

  const h = Math.floor(tSim / 3600), m = Math.floor((tSim % 3600) / 60);
  $("time-label").textContent = `T+${h}:${String(m).padStart(2, "0")}`;
  $("orbit-label").textContent = `${Math.round(inOrbitAt(tSim)).toLocaleString()} fragments in orbit`;
}

let last = performance.now();
function tick(now: number) {
  const dt = Math.min(0.1, (now - last) / 1000); // clamp after a background-tab pause
  last = now;
  if (playing && tEnd > 0) {
    tSim = (tSim + dt * Number(speedSelect.value)) % tEnd;
    slider.value = String(tSim);
    dirty = true;
  }
  if (dirty) {
    draw();
    dirty = false;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

function setPlaying(on: boolean) {
  playing = on;
  playButton.textContent = on ? "❚❚" : "▶";
  playButton.setAttribute("aria-label", on ? "Pause" : "Play");
}

playButton.addEventListener("click", () => setPlaying(!playing));
slider.addEventListener("input", () => {
  tSim = Number(slider.value);
<<<<<<< HEAD
  dirty = true;
});
showParticles.addEventListener("change", () => (dirty = true));
=======
  refreshDensity = true;
  dirty = true;
});
showParticles.addEventListener("change", () => {
  refreshDensity = true;
  dirty = true;
});
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
showVoxels.addEventListener("change", () => (dirty = true));
document.querySelectorAll<HTMLInputElement>('input[name="metric"]').forEach((el) =>
  el.addEventListener("change", () => {
    metric = Number(el.value);
    buildVoxelLayers();
    dirty = true;
  }),
);

// ---------- on load: show the newest stored simulation, or follow one still running ----------

(async () => {
  try {
    const sims = await api<Simulation[]>("/v1/simulations");
    const active = sims.find((s) => s.status === "queued" || s.status === "running");
    const done = sims.find((s) => s.status === "done");
<<<<<<< HEAD
    if (active) await follow(active.id);
    else if (done) await load(await api<Simulation>(`/v1/simulations/${done.id}`));
=======
    const shown = active ?? done;
    const sim = shown ? await api<Simulation>(`/v1/simulations/${shown.id}`) : null;
    void showSatellite(sim?.request);
    if (active) await follow(active.id);
    else if (sim) await load(sim);
>>>>>>> 235d3119d07aa72c2f143baeebde775011dc23eb
    else setStatus("Set a scenario and run a simulation.");
  } catch (err) {
    setStatus(`Could not load from the API at ${API}: ${(err as Error).message}`, true);
  }
})();
