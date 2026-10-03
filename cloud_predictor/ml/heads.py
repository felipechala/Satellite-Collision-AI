from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from .features import CATEGORICAL_INDEX, FEATURE_COLUMNS

QUANTILES = (0.1, 0.5, 0.9)
BAND_COVERAGE = QUANTILES[2] - QUANTILES[0]
NUM_BOOST_ROUND = 150

# Construction effects that DebriSat says can only push a correction one way (+1 up, -1 down).
MONOTONE: dict[str, dict[str, int]] = {"am_mu_shift": {"mli_fraction": 1}}


def _params(alpha: float, seed: int) -> dict:
    # Shallow, heavily regularized trees: the training set is a few hundred events.
    return {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "lambda_l2": 1.0,
        "min_data_per_group": 10,
        "cat_smooth": 10.0,
        "cat_l2": 10.0,
        "feature_pre_filter": False,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
        "seed": seed,
        "verbose": -1,
    }


class QuantileHead:
    """Three LightGBM regressors predicting the p10, p50 and p90 of one label.

    LightGBM refuses monotone constraints with the quantile objective, so monotonicity is
    enforced at prediction time instead: each tree ensemble is piecewise constant in a feature
    between its split thresholds, so a running max (or min) over one point per threshold
    interval yields the tightest monotone function that matches the model wherever the model
    is already monotone.
    """

    def __init__(self, boosters: list[lgb.Booster], monotone: Optional[dict[str, int]] = None):
        self.boosters = boosters
        self.monotone = {f: s for f, s in (monotone or {}).items() if s}
        self._thresholds = {f: self._split_thresholds(f) for f in self.monotone}

    def _split_thresholds(self, feature: str) -> np.ndarray:
        found: set[float] = set()
        for b in self.boosters:
            df = b.trees_to_dataframe()
            found.update(df.loc[df["split_feature"] == feature, "threshold"].dropna().astype(float))
        return np.array(sorted(found))

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, monotone: Optional[dict[str, int]] = None,
            seed: int = 0) -> "QuantileHead":
        boosters = []
        for q in QUANTILES:
            ds = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS,
                             categorical_feature=CATEGORICAL_INDEX, free_raw_data=False)
            boosters.append(lgb.train(_params(q, seed), ds, num_boost_round=NUM_BOOST_ROUND))
        return cls(boosters, monotone)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([b.predict(X, num_threads=1) for b in self.boosters])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Quantile predictions in training space, shape (n, 3)."""
        out = self._raw(X)
        for feature, sign in self.monotone.items():
            ts = self._thresholds[feature]
            if len(ts) == 0:
                continue
            j = FEATURE_COLUMNS.index(feature)
            x = X[:, j]
            known = np.flatnonzero(~np.isnan(x))
            if len(known) == 0:
                continue
            # One representative per interval (-inf, t0], (t0, t1], ..., (t_last, inf).
            reps = np.append(ts, ts[-1] + 1.0)
            interval = np.searchsorted(ts, x[known], side="left")
            grid = np.repeat(X[known], len(reps), axis=0)
            grid[:, j] = np.tile(reps, len(known))
            preds = self._raw(grid).reshape(len(known), len(reps), 3)
            # Only intervals at or below each row's own interval may raise (or lower) it.
            later = np.arange(len(reps))[None, :] > interval[:, None]
            if sign > 0:
                preds[later] = -np.inf
                out[known] = preds.max(axis=1)
            else:
                preds[later] = np.inf
                out[known] = preds.min(axis=1)
        return out

    @staticmethod
    def paths(directory: Path, name: str) -> list[Path]:
        return [directory / f"{name}_p{round(q * 100)}.txt" for q in QUANTILES]

    def save(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for b, p in zip(self.boosters, self.paths(directory, name)):
            b.save_model(str(p))

    @classmethod
    def load(cls, directory: Path, name: str, monotone: Optional[dict[str, int]] = None) -> "QuantileHead":
        return cls([lgb.Booster(model_file=str(p)) for p in cls.paths(directory, name)], monotone)


def pinball_loss(y: np.ndarray, pred: np.ndarray) -> float:
    losses = []
    for j, q in enumerate(QUANTILES):
        diff = y - pred[:, j]
        losses.append(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))
    return float(np.mean(losses))


def conformal_offset(scores: np.ndarray, coverage: float = BAND_COVERAGE) -> float:
    """Split-conformal (CQR) widening so [p10 - off, p90 + off] covers the target share."""
    n = len(scores)
    if n == 0:
        return 0.0
    k = min(n, math.ceil((n + 1) * coverage))
    return float(np.sort(scores)[k - 1])


def label_spread(y: np.ndarray) -> tuple[float, float]:
    """p10 and p90 of the labels relative to their median, for fallback bands."""
    if len(y) == 0:
        return 0.0, 0.0
    q10, q50, q90 = np.quantile(y, QUANTILES)
    return float(q10 - q50), float(q90 - q50)


def apply_offset(pred: np.ndarray, offset) -> np.ndarray:
    out = pred.copy()
    out[:, 0] -= offset
    out[:, 2] += offset
    return np.sort(out, axis=1)


@dataclass
class CVResult:
    n_folds: int
    oof: np.ndarray  # conformally adjusted out-of-fold predictions, shape (n, 3)
    cv_loss: float
    baseline_loss: float
    coverage: float
    cqr_offset: float  # fitted on all out-of-fold scores, for the final model


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int,
               monotone: Optional[dict[str, int]] = None, seed: int = 0) -> Optional[CVResult]:
    """Grouped K-fold CV against the neutral baseline; None if there are fewer than 2 groups."""
    k = min(n_folds, len(np.unique(groups)))
    if k < 2:
        return None
    folds = np.empty(len(y), dtype=int)
    raw = np.empty((len(y), 3))
    baseline = np.empty((len(y), 3))
    for f, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, y, groups)):
        folds[te] = f
        raw[te] = QuantileHead.fit(X[tr], y[tr], monotone, seed).predict(X[te])
        lo, hi = label_spread(y[tr])
        baseline[te] = (lo, 0.0, hi)

    raw_sorted = np.sort(raw, axis=1)
    scores = np.maximum(raw_sorted[:, 0] - y, y - raw_sorted[:, 2])
    # Each fold is calibrated only on the other folds' scores, so coverage stays held-out.
    fold_offsets = np.array([conformal_offset(scores[folds != f]) for f in range(k)])
    oof = apply_offset(raw_sorted, fold_offsets[folds])
    covered = (oof[:, 0] <= y) & (y <= oof[:, 2])
    return CVResult(
        n_folds=k,
        oof=oof,
        cv_loss=pinball_loss(y, oof),
        baseline_loss=pinball_loss(y, baseline),
        coverage=float(covered.mean()),
        cqr_offset=conformal_offset(scores),
    )
