"""
Step 3: 3D voxel density field (particles/km^3) from propagated particle positions.

Voxels are an ECI-aligned cubic grid. The grid is stored sparsely (only occupied voxels),
because a dense grid over all of LEO would be ~1e9 cells while a 10k-particle cloud only
touches a few thousand. Densities sum particle *weights* (real fragments represented),
never particle counts. Gaussian smoothing is a sparse 3D convolution, standing in for
the gameplan's KDE step; it conserves total weight.

Output rows (top_voxels) are shaped for the SpacetimeDB DebrisVoxel table / Deck.gl:
voxel center x, y, z [km, ECI] + density [particles/km^3], capped to the 200-1000 voxel
budget the gameplan recommends for WebGL streaming.

Units: km in, particles/km^3 out. Demo (from cloud_predictor/): python -m engine.density_grid
"""
from __future__ import annotations

import numpy as np


def _accumulate(indices, values):
    """Sum values over duplicate integer index rows. Returns (unique rows, sums)."""
    lo = indices.min(axis=0)
    shifted = indices - lo
    dims = tuple(shifted.max(axis=0) + 1)
    key = np.ravel_multi_index(tuple(shifted.T), dims)
    unique_key, inverse = np.unique(key, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    rows = np.stack(np.unravel_index(unique_key, dims), axis=1) + lo
    return rows, sums


def voxelize(r_km, weight, voxel_km=20.0):
    """Bin particle positions into a sparse voxel grid.

    r_km: (K, 3) ECI positions; NaN rows (decayed particles) are dropped.
    weight: (K,) real fragments represented by each particle.

    Returns {"indices": (n, 3) int voxel coordinates, "density": (n,) particles/km^3,
    "voxel_km": float}.
    """
    r = np.asarray(r_km, dtype=float)
    w = np.asarray(weight, dtype=float)
    keep = np.isfinite(r).all(axis=1) & (w > 0)
    if not keep.any():
        return {"indices": np.empty((0, 3), dtype=np.int64),
                "density": np.empty(0), "voxel_km": float(voxel_km)}
    idx = np.floor(r[keep] / voxel_km).astype(np.int64)
    rows, mass = _accumulate(idx, w[keep])
    return {"indices": rows, "density": mass / voxel_km**3, "voxel_km": float(voxel_km)}


def _kernel(sigma_voxels, truncate):
    """Voxel offsets (m, 3) and normalized Gaussian weights (m,) out to truncate*sigma."""
    if sigma_voxels <= 0:
        return np.zeros((1, 3), dtype=np.int64), np.ones(1)
    reach = max(1, int(np.ceil(truncate * sigma_voxels)))
    axis = np.arange(-reach, reach + 1)
    ox, oy, oz = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([ox, oy, oz], axis=-1).reshape(-1, 3)
    kernel = np.exp(-np.sum(offsets**2, axis=1) / (2.0 * sigma_voxels**2))
    return offsets, kernel / kernel.sum()


_KEY_OFFSET = 1 << 20  # voxel indices must stay within +-2^20 (20 km voxels: +-21 million km)


def _keys(indices):
    """One sortable int64 per voxel index row."""
    i = np.asarray(indices, dtype=np.int64) + _KEY_OFFSET
    return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]


def gaussian_smooth(grid, sigma_voxels=1.0, truncate=2.0):
    """Smooth a sparse voxel grid with a 3D Gaussian kernel (KDE step).

    Each occupied voxel's mass is spread over its neighbors out to truncate*sigma.
    Total weight is conserved; the result has more, lower-density voxels.
    """
    if grid["indices"].shape[0] == 0 or sigma_voxels <= 0:
        return grid
    offsets, kernel = _kernel(sigma_voxels, truncate)

    idx = (grid["indices"][:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    rho = (grid["density"][:, None] * kernel[None, :]).ravel()
    rows, rho_sum = _accumulate(idx, rho)
    return {"indices": rows, "density": rho_sum, "voxel_km": grid["voxel_km"]}


def smoothed_at(grid, query_indices, sigma_voxels=1.0, truncate=2.0):
    """gaussian_smooth(grid) evaluated only at the given voxel indices (q, 3); 0 where empty.

    Equal to looking the query voxels up in the fully smoothed grid, without building it:
    a query voxel receives kernel[o] of the mass of the raw voxel at query - o.
    """
    query = np.asarray(query_indices, dtype=np.int64).reshape(-1, 3)
    if grid["indices"].shape[0] == 0 or query.shape[0] == 0:
        return np.zeros(query.shape[0])
    offsets, kernel = _kernel(sigma_voxels, truncate)
    keys = _keys(grid["indices"])
    order = np.argsort(keys)
    sorted_keys, sorted_density = keys[order], np.asarray(grid["density"], dtype=float)[order]

    sources = _keys((query[:, None, :] - offsets[None, :, :]).reshape(-1, 3))
    pos = np.minimum(np.searchsorted(sorted_keys, sources), sorted_keys.size - 1)
    raw = np.where(sorted_keys[pos] == sources, sorted_density[pos], 0.0)
    return raw.reshape(query.shape[0], offsets.shape[0]) @ kernel


def top_voxels(grid, max_voxels=1000):
    """Densest voxels as frontend-ready rows: centers [km, ECI] + density.

    Returns {"xyz_km": (n, 3), "density": (n,)}, sorted densest-first and capped at
    max_voxels (the gameplan's WebGL streaming budget).
    """
    order = np.argsort(grid["density"])[::-1][:max_voxels]
    centers = (grid["indices"][order] + 0.5) * grid["voxel_km"]
    return {"xyz_km": centers, "density": grid["density"][order]}


def density_timeline(r_km, alive, weight, voxel_km=20.0, sigma_voxels=1.0, max_voxels=1000):
    """Voxelize + smooth every snapshot of a propagate_cloud result.

    r_km: (T, K, 3), alive: (T, K), weight: (K,). Returns a list of top_voxels dicts.
    """
    frames = []
    for j in range(r_km.shape[0]):
        grid = voxelize(r_km[j], np.where(alive[j], weight, 0.0), voxel_km)
        frames.append(top_voxels(gaussian_smooth(grid, sigma_voxels), max_voxels))
    return frames


if __name__ == "__main__":
    from engine.fragmentation import generate_cloud
    from engine.propagator import circular_state, propagate_cloud
    from ml.schema import BreakupEvent, Impactor, Spacecraft

    event = BreakupEvent(
        event_type="collision",
        epoch="2026-10-03T00:00:00Z",
        target=Spacecraft(object_class="payload", dry_mass_kg=950.0),
        impactor=Impactor(mass_kg=50.0, v_rel_km_s=10.0),
    )
    cloud = generate_cloud(event)
    r0, v0 = circular_state(alt_km=780.0, inc_deg=86.4)
    out = propagate_cloud(cloud, r0, v0, times_s=[0.0, 6 * 3600.0, 48 * 3600.0])

    voxel_km = 20.0
    for j, t in enumerate(out["times_s"]):
        w = np.where(out["alive"][j], out["weight"], 0.0)
        grid = voxelize(out["r_km"][j], w, voxel_km)
        smoothed = gaussian_smooth(grid, sigma_voxels=1.0)
        rows = top_voxels(smoothed, max_voxels=1000)
        kept = w.sum() and smoothed["density"].sum() * voxel_km**3 / w.sum()
        print(f"t={t / 3600.0:5.1f} h: {grid['indices'].shape[0]:6d} occupied voxels "
              f"({smoothed['indices'].shape[0]} after smoothing, mass kept {kept:.1%}), "
              f"peak density {rows['density'][0]:.3g} fragments/km^3")
