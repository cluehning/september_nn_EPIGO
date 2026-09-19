from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)


def shannon_entropy(prob: np.ndarray, axis: int | None = None) -> np.ndarray:
    """
    Shannon entropy H(p) = -sum(p * log2(p)).

    Accepts probabilities or raw counts; raw counts are normalized.
    """
    p = np.asarray(prob, dtype=float)

    if np.any(p < 0):
        raise ValueError("Probabilities/counts must be non-negative")

    if axis is None:
        total = float(np.sum(p))
        if total <= 0.0:
            raise ValueError("Sum must be > 0 to compute entropy")
        p = p / total
    else:
        total = np.sum(p, axis=axis, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError("All slices must sum to > 0 to compute entropy")
        p = p / total

    eps = np.finfo(float).tiny
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log2(p), axis=axis)


def spectrum_1d(x: np.ndarray, detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple 1D power spectrum using rFFT.

    Returns (freqs, power). No SciPy required.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("spectrum_1d expects a 1D array")

    if detrend:
        x = x - float(np.mean(x))

    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 samples")

    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs, power


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """
    Container for a feature matrix and labels.
    """
    X: np.ndarray
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]


def build_feature_matrix(
    rows: Iterable[dict[str, float]],
    sample_ids: Iterable[str] | None = None,
) -> FeatureMatrix:
    """
    Build a dense feature matrix from an iterable of dicts.

    This is a robust replacement for ad-hoc pandas merges:
    - stable feature ordering (sorted)
    - fills missing features with 0.0
    """
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No rows provided")

    all_feats: set[str] = set()
    for r in rows_list:
        all_feats.update(r.keys())

    feature_names = tuple(sorted(all_feats))
    X = np.zeros((len(rows_list), len(feature_names)), dtype=float)

    idx = {name: j for j, name in enumerate(feature_names)}
    for i, r in enumerate(rows_list):
        for k, v in r.items():
            X[i, idx[k]] = float(v)

    if sample_ids is None:
        sample_ids_out = tuple(str(i) for i in range(len(rows_list)))
    else:
        sample_ids_out = tuple(sample_ids)
        if len(sample_ids_out) != len(rows_list):
            raise ValueError("sample_ids length must match number of rows")

    return FeatureMatrix(X=X, feature_names=feature_names, sample_ids=sample_ids_out)


def normalize_matrix(
    X: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> np.ndarray:
    """
    Normalize a matrix safely.

    method:
      - "zscore": (x - mean) / std
      - "minmax": (x - min) / (max - min)
      - "l2": x / ||x||
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("Cannot normalize an empty array")

    m = method.lower()
    if m == "zscore":
        mean = np.mean(X, axis=axis, keepdims=True)
        std = np.std(X, axis=axis, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)
        return (X - mean) / std

    if m == "minmax":
        mn = np.min(X, axis=axis, keepdims=True)
        mx = np.max(X, axis=axis, keepdims=True)
        denom = np.where((mx - mn) == 0.0, 1.0, (mx - mn))
        return (X - mn) / denom

    if m == "l2":
        norm = np.linalg.norm(X, axis=axis, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return X / norm

    raise ValueError(f"Unknown normalization method: {method}")
