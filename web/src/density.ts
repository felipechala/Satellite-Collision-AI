// Expected fragment density at each sample, in fragments/km^3: bin the samples' weights into cubic
// cells, then evaluate the Gaussian-smoothed cell density (3x3x3 kernel, sigma = 1 cell) at each
// sample's cell. Same method as the API's voxel product (cloud_predictor engine.density_grid:
// voxelize + smoothed_at(sigma_voxels=1, truncate=1)), on coarser cells. O(N * 27), so it stays cheap
// even when every sample sits in one cell (the t = 0 burst).

const OFF = 1024; // cell indices must stay within +-1024 (100 km cells: +-102,400 km)
const SPAN = 2048;

/** Neighbour key offsets and normalized Gaussian weights for the 27 cells around a cell. */
const KERNEL = (() => {
  const dk: number[] = [];
  const w: number[] = [];
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
    dk.push((dx * SPAN + dy) * SPAN + dz);
    w.push(Math.exp(-(dx * dx + dy * dy + dz * dz) / 2));
  }
  const total = w.reduce((a, b) => a + b, 0);
  return { dk: Float64Array.from(dk), w: Float64Array.from(w.map((x) => x / total)) };
})();

let keys = new Float64Array(0);
const mass = new Map<number, number>();
const cellDensity = new Map<number, number>();

/**
 * posKm: ECI positions, 3 per sample [km]; weight: real fragments per sample; alive: 1 if the sample
 * is in orbit. Writes each alive sample's density [fragments/km^3] to out (0 for re-entered samples).
 */
export function densityAtSamples(posKm: Float64Array, weight: Float64Array, alive: Uint8Array,
                                 cellKm: number, out: Float64Array): void {
  const n = weight.length;
  if (keys.length < n) keys = new Float64Array(n);
  mass.clear();
  for (let i = 0; i < n; i++) {
    if (!alive[i]) continue;
    const ix = Math.floor(posKm[3 * i] / cellKm) + OFF;
    const iy = Math.floor(posKm[3 * i + 1] / cellKm) + OFF;
    const iz = Math.floor(posKm[3 * i + 2] / cellKm) + OFF;
    const key = (ix * SPAN + iy) * SPAN + iz;
    keys[i] = key;
    mass.set(key, (mass.get(key) ?? 0) + weight[i]);
  }
  const volume = cellKm ** 3;
  cellDensity.clear(); // samples sharing a cell share its density: smooth each occupied cell once
  for (let i = 0; i < n; i++) {
    if (!alive[i]) {
      out[i] = 0;
      continue;
    }
    let d = cellDensity.get(keys[i]);
    if (d === undefined) {
      let s = 0;
      for (let k = 0; k < 27; k++) s += KERNEL.w[k] * (mass.get(keys[i] - KERNEL.dk[k]) ?? 0);
      d = s / volume;
      cellDensity.set(keys[i], d);
    }
    out[i] = d;
  }
}
