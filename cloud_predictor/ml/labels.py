"""Per-event training labels, in the space each head is trained in.

n_multiplier and the two scales are learned as natural logs; the shifts are learned linearly.
Every label is measured against the same SBM curves the engine applies (contract.py).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import sbm
from .data import TrainingRow
from .schema import parse_epoch

LEARNED_HEADS = ("n_multiplier", "am_mu_shift", "am_sigma_scale", "dv_mu_shift", "dv_sigma_scale")
CATALOG_LAG = timedelta(days=365)


def _log_scale_label(std_ratio: float, n: int) -> float:
    if std_ratio <= 0:
        return math.nan  # identical fragments: no usable spread
    # E[ln s] is about ln(sigma) - 1/(2(n-1)); remove that bias so noise centres on neutral.
    return math.log(std_ratio) + 1.0 / (2.0 * (n - 1))


def count_label(row: TrainingRow, as_of: datetime) -> float:
    if row.n_cataloged is None or row.n_cataloged <= 0:
        return math.nan
    if parse_epoch(row.event.epoch) > as_of - CATALOG_LAG:
        return math.nan
    return math.log(row.n_cataloged / float(sbm.n_cum_sbm(sbm.LC_ANCHOR_M, row.event)))


def am_labels(frags: pd.DataFrame, min_fragments: int, min_tle_span_days: float) -> tuple[float, float]:
    span = frags["tle_span_days"]
    ok = (frags["lc_m"] > 0) & (frags["am_m2_kg"] > 0) & ~(span < min_tle_span_days)
    f = frags[ok]
    n = len(f)
    if n < max(min_fragments, 2):
        return math.nan, math.nan
    lam = np.log10(f["lc_m"].to_numpy(float))
    r = np.log10(f["am_m2_kg"].to_numpy(float)) - sbm.am_mu(lam)
    shift = float(r.mean())
    z = (r - shift) / sbm.am_sigma(lam)
    scale = math.sqrt(float((z ** 2).sum()) / (n - 1))
    return shift, _log_scale_label(scale, n)


def dv_labels(frags: pd.DataFrame, event_type: str, min_fragments: int) -> tuple[float, float]:
    ok = (frags["lc_m"] > 0) & (frags["am_m2_kg"] > 0) & (frags["dv_m_s"] > 0)
    f = frags[ok]
    n = len(f)
    if n < max(min_fragments, 2):
        return math.nan, math.nan
    chi = np.log10(f["am_m2_kg"].to_numpy(float))
    u = np.log10(f["dv_m_s"].to_numpy(float)) - sbm.dv_mu(chi, event_type)
    return float(u.mean()), _log_scale_label(float(u.std(ddof=1)) / sbm.DV_SIGMA, n)


def compute_labels(rows: Sequence[TrainingRow], fragments: Optional[pd.DataFrame], as_of: datetime,
                   min_fragments: int = 5, min_tle_span_days: float = 180.0) -> pd.DataFrame:
    """One row per event (same order as rows), one column per learned head; NaN = no label."""
    by_event = dict(tuple(fragments.groupby("event_id"))) if fragments is not None else {}
    out = []
    for row in rows:
        labels = dict.fromkeys(LEARNED_HEADS, math.nan)
        labels["n_multiplier"] = count_label(row, as_of)
        frags = by_event.get(row.event_id)
        if frags is not None:
            labels["am_mu_shift"], labels["am_sigma_scale"] = am_labels(frags, min_fragments, min_tle_span_days)
            labels["dv_mu_shift"], labels["dv_sigma_scale"] = dv_labels(frags, row.event.event_type, min_fragments)
        out.append(labels)
    return pd.DataFrame(out, columns=list(LEARNED_HEADS), index=[r.event_id for r in rows])
