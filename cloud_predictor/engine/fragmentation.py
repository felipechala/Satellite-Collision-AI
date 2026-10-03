"""
Step 1: fragmentation engine (NASA Standard Breakup Model, small-fragment regime).

Given a breakup, produce K weighted "representative particles" between 2 mm and 10 cm,
each with a size, area-to-mass ratio, mass, and ejection velocity vector.

Units: metres, kilograms, m/s (convert dv to km/s before adding to an orbital state).
"""
import numpy as np

L_MIN, L_MAX = 0.002, 0.10  # characteristic length range [m]


# ---------- 1. How many fragments? (cumulative count larger than lc) ----------

def n_collision(lc, m_eff):
    """m_eff: target + impactor mass [kg] if catastrophic (EMR >= 40 J/g),
    else impactor mass [kg] * (v_rel [km/s])**2."""
    return 0.1 * m_eff**0.75 * lc**-1.71


def n_explosion(lc, s=1.0):
    """s: unitless scaling factor by explosion type (~1 for a typical rocket body)."""
    return 6.0 * s * lc**-1.6


# ---------- 2. Area-to-mass for small fragments (lc < ~8 cm) ----------

def sample_am(lc, rng):
    """chi = log10(A/M) is normal with size-dependent mean and spread."""
    lam = np.log10(lc)
    mu = np.clip(-0.3 - 1.4 * (lam + 1.75), -1.0, -0.3)
    sigma = np.where(lam <= -3.5, 0.2, 0.2 + 0.1333 * (lam + 3.5))
    return 10 ** rng.normal(mu, sigma)  # m^2/kg


def area(lc):
    return np.where(lc < 0.00167, 0.540424 * lc**2, 0.556945 * lc**2.5003943)  # m^2


# ---------- 3. Ejection velocity ----------

def sample_dv(am, rng, alpha=0.9, beta=2.9, sigma=0.4):
    """log10(dv) ~ N(alpha*log10(A/M) + beta, sigma).
    Defaults are the collision values; explosions use alpha=0.2, beta=1.85.
    These are the knobs the ML model would later adjust per satellite."""
    speed = 10 ** rng.normal(alpha * np.log10(am) + beta, sigma)  # m/s
    # isotropic direction
    v = rng.normal(size=(am.size, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * speed[:, None]


# ---------- 4. Put it together ----------

def generate_cloud(m_eff, k=10_000, n_bins=50, collision=True, seed=0, **dv_kwargs):
    rng = np.random.default_rng(seed)
    n_of = (lambda l: n_collision(l, m_eff)) if collision else n_explosion

    edges = np.logspace(np.log10(L_MIN), np.log10(L_MAX), n_bins + 1)
    per_bin = k // n_bins
    real_in_bin = n_of(edges[:-1]) - n_of(edges[1:])  # real fragments per size bin

    # per_bin particles in each bin, log-uniform within the bin
    lo = np.repeat(np.log10(edges[:-1]), per_bin)
    hi = np.repeat(np.log10(edges[1:]), per_bin)
    lc = 10 ** rng.uniform(lo, hi)
    weight = np.repeat(real_in_bin / per_bin, per_bin)  # real fragments per particle

    am = sample_am(lc, rng)
    return {
        "lc": lc,                   # m
        "weight": weight,           # real fragments represented
        "am": am,                   # m^2/kg
        "mass": area(lc) / am,      # kg
        "dv": sample_dv(am, rng, **dv_kwargs),  # m/s, shape (K, 3)
    }


if __name__ == "__main__":
    cloud = generate_cloud(m_eff=1000.0)  # e.g. 950 kg satellite + 50 kg impactor
    speed = np.linalg.norm(cloud["dv"], axis=1)
    print(f"particles:            {cloud['lc'].size}")
    print(f"real fragments > 2mm: {cloud['weight'].sum():,.0f}")
    print(f"expected from model:  {n_collision(L_MIN, 1000.0) - n_collision(L_MAX, 1000.0):,.0f}")
    print(f"median A/M:           {np.median(cloud['am']):.3f} m^2/kg")
    print(f"median dv:            {np.median(speed):.0f} m/s")
    print(f"fragment mass total:  {(cloud['mass'] * cloud['weight']).sum():.0f} kg")