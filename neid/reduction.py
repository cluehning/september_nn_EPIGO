from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PCAResult:
    """A fitted singular-value decomposition and its reconstruction tools."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = _validate_matrix(X, self.mean.size)
        return (values - self.mean) @ self.components.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        values = np.asarray(Z, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.components.shape[0]:
            raise ValueError(
                "Latent matrix has the wrong number of components"
            )
        return values @ self.components + self.mean

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.inverse_transform(self.transform(X))

    def relative_error(self, X: np.ndarray) -> float:
        values = _validate_matrix(X, self.mean.size)
        difference = self.reconstruct(values) - values
        scale = max(float(np.linalg.norm(values)), np.finfo(float).eps)
        return float(np.linalg.norm(difference) / scale)


def fit_pca(X: np.ndarray, n_components: int) -> PCAResult:
    """Fit PCA with SVD, returning a linear reconstruction baseline."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "X must have at least two rows and be two-dimensional"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain finite values")
    max_components = min(values.shape)
    if not 0 < n_components <= max_components:
        raise ValueError(
            f"n_components must be between 1 and {max_components}"
        )

    mean = np.mean(values, axis=0)
    centered = values - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    variance = (singular_values**2) / max(values.shape[0] - 1, 1)
    total = max(float(np.sum(variance)), np.finfo(float).eps)
    return PCAResult(
        mean=mean,
        components=vh[:n_components],
        explained_variance=variance[:n_components],
        explained_variance_ratio=variance[:n_components] / total,
    )


@dataclass(frozen=True, slots=True)
class DictionaryResult:
    """A small learned dictionary and sparse window codes."""

    dictionary: np.ndarray
    codes: np.ndarray

    @property
    def reconstruction(self) -> np.ndarray:
        return self.codes @ self.dictionary

    def relative_error(self, X: np.ndarray) -> float:
        values = _validate_matrix(X, self.dictionary.shape[1])
        difference = self.reconstruction - values
        scale = max(float(np.linalg.norm(values)), np.finfo(float).eps)
        return float(np.linalg.norm(difference) / scale)


def learn_dictionary(
    X: np.ndarray,
    n_atoms: int = 8,
    sparsity: int = 2,
    iterations: int = 20,
    seed: int = 0,
) -> DictionaryResult:
    """Learn simple sparse Lego-like blocks with alternating updates.

    Each row is approximated by a small number of dictionary atoms. This is a
    compact, dependency-free baseline rather than a replacement for a full
    sparse-coding library.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "X must have at least two rows and be two-dimensional"
        )
    if not 0 < sparsity <= n_atoms or n_atoms > values.shape[0]:
        raise ValueError("Require 0 < sparsity <= n_atoms <= number of rows")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    choices = rng.choice(values.shape[0], size=n_atoms, replace=False)
    dictionary = values[choices].copy()
    dictionary /= np.maximum(
        np.linalg.norm(dictionary, axis=1, keepdims=True), 1e-12
    )

    for _ in range(iterations):
        correlations = values @ dictionary.T
        codes = np.zeros((values.shape[0], n_atoms), dtype=float)
        active = np.argpartition(np.abs(correlations), -sparsity, axis=1)[
            :, -sparsity:
        ]
        rows = np.arange(values.shape[0])[:, None]
        codes[rows, active] = correlations[rows, active]
        for atom in range(n_atoms):
            weights = codes[:, atom]
            denominator = float(np.dot(weights, weights))
            if denominator > 1e-12:
                dictionary[atom] = (weights @ values) / denominator
                norm = np.linalg.norm(dictionary[atom])
                if norm > 1e-12:
                    dictionary[atom] /= norm

    correlations = values @ dictionary.T
    codes = np.zeros((values.shape[0], n_atoms), dtype=float)
    active = np.argpartition(np.abs(correlations), -sparsity, axis=1)[
        :, -sparsity:
    ]
    rows = np.arange(values.shape[0])[:, None]
    codes[rows, active] = correlations[rows, active]
    return DictionaryResult(dictionary=dictionary, codes=codes)


def _validate_matrix(X: np.ndarray, expected_features: int) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] != expected_features:
        raise ValueError("X has the wrong matrix shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain finite values")
    return values
