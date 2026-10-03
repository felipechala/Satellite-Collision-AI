"""Quantile regression heads: small PyTorch neural networks trained with pinball loss.

These heads were originally LightGBM gradient-boosted trees; they are now MLPs with the
same fit/predict/save/load contract. One network per head predicts the p10/p50/p90
jointly. Preprocessing the trees got for free lives inside the head: numeric features
are standardized with NaN -> 0 plus a missing indicator per column, and categorical
codes are one-hot encoded with an explicit missing/unseen bucket.

Monotone constraints (MONOTONE) are enforced by construction rather than post hoc: the
constrained feature bypasses the trunk through an additive non-decreasing path (a sum
of positive-weight ReLU ramps), so every prediction is monotone in that feature. The
trade-off versus the trees' threshold envelope is that the constrained feature cannot
interact with other features, which matches how the DebriSat rule is stated: the
correction may only move one way.

Everything runs in float64 on CPU, full-batch, with seeded initialization, so training
and prediction are deterministic. Weights persist as .npz (plain arrays, no pickle).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from .features import CATEGORICAL_INDEX, FEATURE_COLUMNS

QUANTILES = (0.1, 0.5, 0.9)
BAND_COVERAGE = QUANTILES[2] - QUANTILES[0]

# Construction effects that DebriSat says can only push a correction one way (+1 up, -1 down).
MONOTONE: dict[str, dict[str, int]] = {"am_mu_shift": {"mli_fraction": 1}}

# Small and heavily regularized: the training set is a few hundred events.
HIDDEN = (16, 8)
N_KNOTS = 8
EPOCHS = 800
LEARNING_RATE = 0.02
WEIGHT_DECAY = 8e-3  # L2 on trunk weights only (plain Adam, not AdamW)

_WEIGHT_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3", "Wm")


class _MonotoneQuantileNet(torch.nn.Module):
    """Trunk MLP plus the additive monotone ReLU-ramp path for one constrained feature."""

    def __init__(self, d: int, mono_sign: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        h1, h2 = HIDDEN

        def he(fan_in: int, fan_out: int) -> torch.nn.Parameter:
            w = torch.randn((fan_in, fan_out), generator=g, dtype=torch.float64)
            return torch.nn.Parameter(w * math.sqrt(2.0 / fan_in))

        self.W1, self.b1 = he(d, h1), torch.nn.Parameter(torch.zeros(h1, dtype=torch.float64))
        self.W2, self.b2 = he(h1, h2), torch.nn.Parameter(torch.zeros(h2, dtype=torch.float64))
        self.W3, self.b3 = he(h2, 3), torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        # softplus(-2) ~ 0.13 per knot: gentle initial slope on the monotone path
        self.Wm = torch.nn.Parameter(torch.full((3, N_KNOTS), -2.0, dtype=torch.float64))
        self.register_buffer("knots", torch.arange(N_KNOTS, dtype=torch.float64) / N_KNOTS)
        self.mono_sign = float(mono_sign)

    def forward(self, Z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(Z @ self.W1 + self.b1)
        h2 = torch.relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        # A missing monotone feature (NaN) contributes nothing to the ramp path.
        ramps = torch.nan_to_num(torch.relu(t.unsqueeze(1) - self.knots), nan=0.0)
        return out + self.mono_sign * ramps @ torch.nn.functional.softplus(self.Wm).T


def _transform(p: dict, X) -> tuple[np.ndarray, np.ndarray]:
    """X (n, F) with NaNs -> trunk input Z (n, d) and scaled monotone feature t (n,)."""
    X = np.asarray(X, dtype=float)
    xn = X[:, p["trunk_num_idx"]]
    missing = np.isnan(xn)
    z = (xn - p["num_mean"]) / p["num_std"]
    z[missing] = 0.0
    parts = [z, missing.astype(float)]
    for j, card in zip(p["cat_idx"], p["cat_card"]):
        c = X[:, j]
        code = np.where(np.isfinite(c) & (c >= 0) & (c < card), c, card).astype(int)
        parts.append(np.eye(int(card) + 1)[code])
    mono_idx = int(p["mono_idx"])
    if mono_idx >= 0:
        t_raw = X[:, mono_idx]
        parts.append(np.isnan(t_raw).astype(float)[:, None])
        t = (t_raw - p["mono_lo"]) / (p["mono_hi"] - p["mono_lo"])
    else:
        t = np.full(X.shape[0], np.nan)
    return np.concatenate(parts, axis=1), t


class QuantileHead:
    """One PyTorch MLP predicting the p10, p50 and p90 of a label.

    Inputs may contain NaN anywhere; the stored preprocessing handles it. Training is
    full-batch Adam on mean pinball loss, deterministic for a given seed.
    """

    def __init__(self, params: dict, monotone: Optional[dict[str, int]] = None):
        # monotone is kept for signature compatibility; the constraint is baked into
        # the parameters at fit time and reloaded from them.
        self.p = {k: np.asarray(v) for k, v in params.items() if k not in _WEIGHT_KEYS}
        self.net = _MonotoneQuantileNet(int(np.asarray(params["W1"]).shape[0]),
                                        float(self.p["mono_sign"]))
        with torch.no_grad():
            for k in _WEIGHT_KEYS:
                getattr(self.net, k).copy_(torch.as_tensor(np.asarray(params[k]),
                                                           dtype=torch.float64))
        self.net.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Quantile predictions in training space, shape (n, 3)."""
        Z, t = _transform(self.p, X)
        with torch.no_grad():
            return self.net(torch.from_numpy(Z), torch.from_numpy(t)).numpy()

    def _raw(self, X: np.ndarray) -> np.ndarray:
        """Identical to predict: the network is monotone by construction, so there is
        no separate unconstrained model (kept for the previous API's callers/tests)."""
        return self.predict(X)

    # ---------- training ----------

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, monotone: Optional[dict[str, int]] = None,
            seed: int = 0) -> "QuantileHead":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        mono = {f: s for f, s in (monotone or {}).items() if s}
        if len(mono) > 1:
            raise ValueError("at most one monotone feature is supported")
        mono_name, mono_sign = next(iter(mono.items())) if mono else (None, 0)
        mono_idx = FEATURE_COLUMNS.index(mono_name) if mono_name else -1

        num_idx = [j for j in range(len(FEATURE_COLUMNS))
                   if j not in CATEGORICAL_INDEX and j != mono_idx]
        p: dict = {
            "trunk_num_idx": np.array(num_idx, dtype=int),
            "cat_idx": np.array(CATEGORICAL_INDEX, dtype=int),
            "mono_idx": mono_idx,
            "mono_sign": float(mono_sign),
        }

        xn = X[:, num_idx]
        finite = np.isfinite(xn)
        cnt = np.maximum(finite.sum(axis=0), 1)
        mean = np.where(finite, xn, 0.0).sum(axis=0) / cnt
        std = np.sqrt((np.where(finite, xn - mean, 0.0) ** 2).sum(axis=0) / cnt)
        p["num_mean"] = mean
        p["num_std"] = np.where(std > 1e-6, std, 1.0)

        cards = []
        for j in CATEGORICAL_INDEX:
            c = X[:, j]
            vals = c[np.isfinite(c)]
            cards.append(int(vals.max()) + 1 if vals.size else 0)
        p["cat_card"] = np.array(cards, dtype=int)

        if mono_idx >= 0:
            tv = X[:, mono_idx]
            vals = tv[np.isfinite(tv)]
            lo = float(vals.min()) if vals.size else 0.0
            hi = float(vals.max()) if vals.size else 1.0
            p["mono_lo"], p["mono_hi"] = lo, (hi if hi > lo else lo + 1.0)
        else:
            p["mono_lo"], p["mono_hi"] = 0.0, 1.0

        Z, t = _transform(p, X)
        net = _MonotoneQuantileNet(Z.shape[1], mono_sign, seed)
        with torch.no_grad():
            net.b3.fill_(float(np.median(y)))
        cls._train(net, Z, t, y)
        weights = {k: getattr(net, k).detach().numpy() for k in _WEIGHT_KEYS}
        return cls({**p, **weights})

    @staticmethod
    def _train(net: _MonotoneQuantileNet, Z: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
        Zt = torch.from_numpy(Z)
        tt = torch.from_numpy(t)
        yt = torch.from_numpy(y).unsqueeze(1)
        q = torch.tensor(QUANTILES, dtype=torch.float64)
        optimizer = torch.optim.Adam(
            [{"params": [net.W1, net.W2, net.W3], "weight_decay": WEIGHT_DECAY},
             {"params": [net.b1, net.b2, net.b3, net.Wm], "weight_decay": 0.0}],
            lr=LEARNING_RATE,
        )
        net.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            diff = yt - net(Zt, tt)
            loss = torch.maximum(q * diff, (q - 1.0) * diff).mean()
            loss.backward()
            optimizer.step()
        net.eval()

    # ---------- persistence ----------

    @staticmethod
    def paths(directory: Path, name: str) -> list[Path]:
        return [directory / f"{name}.npz"]

    def save(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        weights = {k: getattr(self.net, k).detach().numpy() for k in _WEIGHT_KEYS}
        np.savez(self.paths(directory, name)[0], **self.p, **weights)

    @classmethod
    def load(cls, directory: Path, name: str, monotone: Optional[dict[str, int]] = None) -> "QuantileHead":
        with np.load(cls.paths(directory, name)[0]) as data:
            return cls({k: data[k] for k in data.files}, monotone)


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
