"""
Step 2: Monte Carlo propagation of the fragment cloud.

Each representative particle gets its ejection velocity added to the parent's orbital
state, is converted to osculating Keplerian elements, and is advanced with secular J2
drift (nodal regression, perigee rotation, mean-motion correction) plus an
exponential-atmosphere drag decay scaled by each particle's sampled A/M. That captures
the two effects that shape the cloud over the demo horizon — the debris band stretching
into a torus, and high-A/M fragments de-orbiting fast — while staying fully vectorized.

(The gameplan names SGP4, but SGP4 needs TLE mean elements and there is no robust
state-vector -> TLE conversion; for a 0-48 h horizon this is the standard substitute.)

Units: km, km/s and seconds throughout. The fragmentation cloud's dv is in m/s and is
converted here. Run the demo from cloud_predictor/: python -m engine.propagator
"""
from __future__ import annotations

import numpy as np

MU_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
J2 = 1.08262668e-3
CD = 2.2                # drag coefficient for all fragments
DECAY_ALT_KM = 150.0    # perigee below this counts as decayed

# Exponential atmosphere (Vallado, Table 8-4): base altitude [km], density [kg/m^3],
# scale height [km]. Below 150 km a particle is already flagged as decayed.
_ATM = np.array([
    (150.0, 2.070e-09, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
])


def atmosphere_density_kg_m3(h_km):
    h = np.asarray(h_km, dtype=float)
    i = np.clip(np.searchsorted(_ATM[:, 0], h, side="right") - 1, 0, len(_ATM) - 1)
    h0, rho0, scale = _ATM[i, 0], _ATM[i, 1], _ATM[i, 2]
    return rho0 * np.exp(-(h - h0) / scale)


def _pqw_basis(raan, inc, argp):
    """First two columns of the perifocal->ECI rotation, each shape (..., 3)."""
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    p = np.stack([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si], axis=-1)
    q = np.stack([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si], axis=-1)
    return p, q


def circular_state(alt_km, inc_deg, raan_deg=0.0, arg_lat_deg=0.0):
    """ECI position [km] and velocity [km/s] of a circular parent orbit (demo helper)."""
    a = R_EARTH_KM + alt_km
    v = np.sqrt(MU_KM3_S2 / a)
    raan, inc, u = np.radians([raan_deg, inc_deg, arg_lat_deg])
    p, q = _pqw_basis(raan, inc, 0.0)
    return a * (np.cos(u) * p + np.sin(u) * q), v * (-np.sin(u) * p + np.cos(u) * q)


def rv_to_elements(r_km, v_km_s):
    """Osculating elements for each row of r, v. Returns a dict of arrays."""
    r_km = np.atleast_2d(np.asarray(r_km, dtype=float))
    v_km_s = np.atleast_2d(np.asarray(v_km_s, dtype=float))
    r = np.linalg.norm(r_km, axis=1)
    v2 = np.sum(v_km_s**2, axis=1)
    rv = np.sum(r_km * v_km_s, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        a = 1.0 / (2.0 / r - v2 / MU_KM3_S2)
    h_vec = np.cross(r_km, v_km_s)
    h = np.linalg.norm(h_vec, axis=1)
    e_vec = np.cross(v_km_s, h_vec) / MU_KM3_S2 - r_km / r[:, None]
    e = np.linalg.norm(e_vec, axis=1)

    inc = np.arccos(np.clip(h_vec[:, 2] / h, -1.0, 1.0))
    raan = np.arctan2(h_vec[:, 0], -h_vec[:, 1])

    n_vec = np.stack([-h_vec[:, 1], h_vec[:, 0], np.zeros_like(h)], axis=1)
    n = np.linalg.norm(n_vec, axis=1)
    eps = 1e-12
    argp = np.arccos(np.clip(np.sum(n_vec * e_vec, axis=1) / (n * e + eps), -1.0, 1.0))
    argp = np.where(e_vec[:, 2] < 0.0, 2.0 * np.pi - argp, argp)

    nu = np.arccos(np.clip(np.sum(e_vec * r_km, axis=1) / (e * r + eps), -1.0, 1.0))
    nu = np.where(rv < 0.0, 2.0 * np.pi - nu, nu)

    E = 2.0 * np.arctan2(np.sqrt(np.maximum(1.0 - e, 0.0)) * np.sin(nu / 2.0),
                         np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M = E - e * np.sin(E)

    bound = (a > 0.0) & (e < 0.99)
    return {"a": a, "e": e, "inc": inc, "raan": raan, "argp": argp, "M": M, "bound": bound}


def _kepler_E(M, e, iters=12):
    E = np.where(e < 0.8, M, np.pi)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def elements_to_positions(el, mask=None):
    """ECI positions [km], shape (K, 3); rows where mask is False are NaN."""
    a, e = el["a"], el["e"]
    E = _kepler_E(np.mod(el["M"], 2.0 * np.pi), e)
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                          np.sqrt(np.maximum(1.0 - e, 0.0)) * np.cos(E / 2.0))
    rmag = a * (1.0 - e * np.cos(E))
    p, q = _pqw_basis(el["raan"], el["inc"], el["argp"])
    r = rmag[:, None] * (np.cos(nu)[:, None] * p + np.sin(nu)[:, None] * q)
    if mask is not None:
        r[~mask] = np.nan
    return r


def _step(el, alive, bstar_m2_kg, dt_s):
    """Advance elements of alive particles by dt: J2 secular rates + drag decay of a."""
    a, e, inc = el["a"], el["e"], el["inc"]
    n = np.sqrt(MU_KM3_S2 / np.where(alive, a, 1.0) ** 3)
    p_slr = a * (1.0 - e**2)
    fac = 1.5 * J2 * (R_EARTH_KM / np.where(alive, p_slr, 1.0)) ** 2 * n
    sin2i = np.sin(inc) ** 2

    root = np.sqrt(np.maximum(1.0 - e**2, 0.0))
    el["raan"] += np.where(alive, -fac * np.cos(inc) * dt_s, 0.0)
    el["argp"] += np.where(alive, fac * (2.0 - 2.5 * sin2i) * dt_s, 0.0)
    el["M"] += np.where(alive, (n + fac * root * (1.0 - 1.5 * sin2i)) * dt_s, 0.0)

    # Drag: da/dt = -rho * B * sqrt(mu a), evaluated at perigee altitude. The 1e3 factor
    # converts rho[kg/m^3] * B[m^2/kg] * km^2/s to km/s.
    h_perigee = a * (1.0 - e) - R_EARTH_KM
    rho = atmosphere_density_kg_m3(np.maximum(h_perigee, _ATM[0, 0]))
    el["a"] -= np.where(alive, 1e3 * rho * bstar_m2_kg * np.sqrt(MU_KM3_S2 * np.maximum(a, 1.0)) * dt_s, 0.0)
    return alive & (el["a"] * (1.0 - e) - R_EARTH_KM > DECAY_ALT_KM)


def propagate_cloud(cloud, r0_km, v0_km_s, times_s, cd=CD, substep_s=60.0):
    """Propagate a fragmentation cloud from the parent state at the moment of breakup.

    cloud: dict from engine.fragmentation.generate_cloud (dv in m/s).
    times_s: snapshot times in seconds after breakup (>= 0).

    Returns {"times_s": (T,), "r_km": (T, K, 3) with NaN rows for decayed/unbound
    particles, "alive": (T, K) bool, "weight": (K,)}.
    """
    times = np.sort(np.unique(np.asarray(times_s, dtype=float)))
    if times.size == 0 or times[0] < 0.0:
        raise ValueError("times_s must be non-empty and non-negative")

    k = cloud["dv"].shape[0]
    r_part = np.broadcast_to(np.asarray(r0_km, dtype=float), (k, 3))
    v_part = np.asarray(v0_km_s, dtype=float) + cloud["dv"] / 1000.0
    el = rv_to_elements(r_part, v_part)
    alive = el.pop("bound")
    bstar = cd * cloud["am"]  # Cd * A/M [m^2/kg]

    r_out = np.empty((times.size, k, 3))
    alive_out = np.empty((times.size, k), dtype=bool)
    t = 0.0
    for j, t_snap in enumerate(times):
        while t < t_snap:
            dt = min(substep_s, t_snap - t)
            alive = _step(el, alive, bstar, dt)
            t += dt
        r_out[j] = elements_to_positions(el, alive)
        alive_out[j] = alive
    return {"times_s": times, "r_km": r_out, "alive": alive_out, "weight": cloud["weight"]}


if __name__ == "__main__":
    from engine.fragmentation import generate_cloud
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

    # Self-check: at t=0 the cloud should still sit on the parent position.
    err = np.nanmax(np.linalg.norm(out["r_km"][0] - r0, axis=1))
    print(f"t=0 position error:   {err:.3e} km")
    for j, t in enumerate(out["times_s"]):
        r = out["r_km"][j]
        alive = out["alive"][j]
        alt = np.linalg.norm(r[alive], axis=1) - R_EARTH_KM
        # Orbit coverage: share of 5-degree phase bins (around the parent plane) occupied.
        phase = np.arctan2(r[alive] @ (v0 / np.linalg.norm(v0)), r[alive] @ (r0 / np.linalg.norm(r0)))
        coverage = np.unique((np.degrees(phase) // 5).astype(int)).size / 72.0
        frac = cloud["weight"][alive].sum() / cloud["weight"].sum()
        print(f"t={t / 3600.0:5.1f} h: in orbit {frac:6.1%} of fragments, "
              f"altitude p10/p50/p90 = {np.percentile(alt, 10):.0f}/{np.percentile(alt, 50):.0f}/"
              f"{np.percentile(alt, 90):.0f} km, orbit coverage {coverage:.0%}")
