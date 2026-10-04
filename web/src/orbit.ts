// Fragment position from its stored orbit; the JS twin of cloud_predictor engine.propagator.orbit_positions.
// a = a_km + a_dot t, raan/argp drift linearly, M = mean_anomaly + m_dot t + m_ddot t^2 / 2 (e, inc fixed).

export interface Orbits {
  a_km: Float64Array; e: Float64Array; inc: Float64Array; raan: Float64Array; argp: Float64Array;
  mean_anomaly: Float64Array; raan_dot: Float64Array; argp_dot: Float64Array; m_dot: Float64Array;
  a_dot: Float64Array; m_ddot: Float64Array; t_decay_s: Float64Array; lc_m: Float64Array; weight: Float64Array;
}

const TWO_PI = 2 * Math.PI;

/** True while fragment i is in orbit at t (t_decay_s -1 = never re-enters within the horizon). */
export function aliveAt(o: Orbits, i: number, t: number): boolean {
  const decay = o.t_decay_s[i];
  return decay < 0 || t < decay;
}

/** ECI position of fragment i at t seconds after breakup, in meters, written into out. */
export function positionAt(o: Orbits, i: number, t: number, out: { x: number; y: number; z: number }): void {
  const e = o.e[i];
  const a = o.a_km[i] + o.a_dot[i] * t;
  const raan = o.raan[i] + o.raan_dot[i] * t;
  const argp = o.argp[i] + o.argp_dot[i] * t;
  let M = (o.mean_anomaly[i] + o.m_dot[i] * t + 0.5 * o.m_ddot[i] * t * t) % TWO_PI;
  if (M < 0) M += TWO_PI;
  let E = e < 0.8 ? M : Math.PI; // Kepler's equation by Newton, as engine.propagator._kepler_E
  for (let k = 0; k < 12; k++) {
    const dE = (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    E -= dE;
    if (Math.abs(dE) < 1e-12) break;
  }
  const xp = a * (Math.cos(E) - e) * 1000; // perifocal position [m]
  const yp = a * Math.sqrt(1 - e * e) * Math.sin(E) * 1000;
  const co = Math.cos(raan), so = Math.sin(raan), ci = Math.cos(o.inc[i]), si = Math.sin(o.inc[i]);
  const cw = Math.cos(argp), sw = Math.sin(argp);
  out.x = xp * (co * cw - so * sw * ci) + yp * (-co * sw - so * cw * ci);
  out.y = xp * (so * cw + co * sw * ci) + yp * (-so * sw + co * cw * ci);
  out.z = xp * (sw * si) + yp * (cw * si);
}
