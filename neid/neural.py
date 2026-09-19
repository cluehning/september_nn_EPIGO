from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import shannon_entropy, spectrum_1d


@dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """Measurements showing how closely the decoder rebuilt the input."""

    mean_squared_error: float
    mean_absolute_error: float
    relative_error: float

    @property
    def passed(self) -> bool:
        """Return whether the report contains valid finite measurements."""
        return bool(
            np.isfinite(self.mean_squared_error)
            and np.isfinite(self.mean_absolute_error)
            and np.isfinite(self.relative_error)
        )

    def within_tolerance(self, max_relative_error: float = 0.5) -> bool:
        """Check whether reconstruction error is below a chosen tolerance."""
        if max_relative_error < 0.0:
            raise ValueError("max_relative_error must be non-negative")
        return self.passed and self.relative_error <= max_relative_error


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """Training and validation reconstruction losses."""

    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]


def build_signal_windows(
    signals: np.ndarray,
    window_size: int,
    step: int | None = None,
) -> np.ndarray:
    """Turn aligned tracks into rows suitable for the autoencoder.

    ``signals`` has shape ``(channels, bins)``. Each output row contains all
    channels for one genomic window, in channel-major order.
    """
    values = np.asarray(signals, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must be a non-empty (channels, bins) matrix")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    stride = window_size if step is None else int(step)
    if stride <= 0:
        raise ValueError("step must be positive")
    if values.shape[1] < window_size:
        raise ValueError("window_size cannot exceed the number of bins")

    windows = [
        values[:, start : start + window_size].reshape(-1)
        for start in range(0, values.shape[1] - window_size + 1, stride)
    ]
    return np.asarray(windows, dtype=float)


def build_epigo_window_features(
    signals: np.ndarray,
    window_size: int,
    step: int | None = None,
    histogram_bins: int = 16,
) -> np.ndarray:
    """Build EPIGO descriptors for each multi-track genomic window.

    For every channel and window, the output contains mean, standard
    deviation, entropy, and dominant frequency. These compact descriptors are
    the recommended input for neural representation learning.
    """
    values = np.asarray(signals, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must be a non-empty (channels, bins) matrix")
    if window_size < 4:
        raise ValueError("window_size must be at least 4")
    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be positive")
    stride = window_size if step is None else int(step)
    if stride <= 0:
        raise ValueError("step must be positive")
    if values.shape[1] < window_size:
        raise ValueError("window_size cannot exceed the number of bins")

    rows: list[list[float]] = []
    for start in range(0, values.shape[1] - window_size + 1, stride):
        row: list[float] = []
        for channel in values:
            segment = channel[start : start + window_size]
            histogram = np.histogram(segment, bins=histogram_bins)[0]
            frequencies, power = spectrum_1d(segment)
            row.extend(
                (
                    float(np.mean(segment)),
                    float(np.std(segment)),
                    float(shannon_entropy(histogram)),
                    float(frequencies[int(np.argmax(power))]),
                )
            )
        rows.append(row)
    return np.asarray(rows, dtype=float)


class Autoencoder:
    """A small fully-connected autoencoder implemented with NumPy.

    Each row of ``X`` is one genomic window and each column is one EPIGO
    feature or signal bin. The encoder compresses a window into a latent
    blueprint. The decoder expands that blueprint and rebuilds the row.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        learning_rate: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if input_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Network dimensions must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
            rng.normal(
                0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, latent_dim)
            ),
            rng.normal(
                0.0, np.sqrt(2.0 / latent_dim), (latent_dim, hidden_dim)
            ),
            rng.normal(
                0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, input_dim)
            ),
        ]
        self.biases = [
            np.zeros(hidden_dim, dtype=float),
            np.zeros(latent_dim, dtype=float),
            np.zeros(hidden_dim, dtype=float),
            np.zeros(input_dim, dtype=float),
        ]

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_gradient(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(float)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        z1 = X @ self.weights[0] + self.biases[0]
        a1 = self._relu(z1)
        z2 = a1 @ self.weights[1] + self.biases[1]
        latent = self._relu(z2)
        z3 = latent @ self.weights[2] + self.biases[2]
        a3 = self._relu(z3)
        reconstruction = a3 @ self.weights[3] + self.biases[3]
        return reconstruction, [X, z1, a1, z2, latent, z3, a3]

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return the learned compact blueprint for each input row."""
        values = self._validate_matrix(X)
        _, cache = self._forward(values)
        return cache[4].copy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Rebuild input rows from their learned latent blueprints."""
        values = self._validate_matrix(X)
        reconstruction, _ = self._forward(values)
        return reconstruction

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 500,
        validation_data: np.ndarray | None = None,
        verbose: bool = False,
    ) -> TrainingHistory:
        """Train by minimizing mean squared reconstruction error."""
        values = self._validate_matrix(X)
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        validation = (
            None
            if validation_data is None
            else self._validate_matrix(validation_data)
        )

        train_losses: list[float] = []
        validation_losses: list[float] = []
        sample_count = values.shape[0]

        for epoch in range(epochs):
            reconstruction, cache = self._forward(values)
            error = reconstruction - values
            train_losses.append(float(np.mean(error**2)))

            d4 = 2.0 * error / sample_count
            d3 = d4 @ self.weights[3].T * self._relu_gradient(cache[5])
            d2 = d3 @ self.weights[2].T * self._relu_gradient(cache[3])
            d1 = d2 @ self.weights[1].T * self._relu_gradient(cache[1])

            gradients_w = [
                cache[0].T @ d1,
                cache[2].T @ d2,
                cache[4].T @ d3,
                cache[6].T @ d4,
            ]
            gradients_b = [
                np.sum(d1, axis=0),
                np.sum(d2, axis=0),
                np.sum(d3, axis=0),
                np.sum(d4, axis=0),
            ]
            for i in range(4):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            if validation is not None:
                validation_losses.append(self._loss(validation))
            if verbose and (
                epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0
            ):
                print(
                    f"epoch {epoch + 1}/{epochs} loss={train_losses[-1]:.6f}"
                )

        return TrainingHistory(tuple(train_losses), tuple(validation_losses))

    def evaluate(self, X: np.ndarray) -> ReconstructionReport:
        """Measure reconstruction quality on held-out rows."""
        values = self._validate_matrix(X)
        difference = self.reconstruct(values) - values
        scale = max(float(np.linalg.norm(values)), np.finfo(float).eps)
        return ReconstructionReport(
            mean_squared_error=float(np.mean(difference**2)),
            mean_absolute_error=float(np.mean(np.abs(difference))),
            relative_error=float(np.linalg.norm(difference) / scale),
        )

    def assert_reconstruction(
        self,
        X: np.ndarray,
        max_relative_error: float = 0.5,
    ) -> ReconstructionReport:
        """Evaluate held-out rows and raise if reconstruction misses tolerance."""
        report = self.evaluate(X)
        if not report.within_tolerance(max_relative_error):
            raise ValueError(
                "Reconstruction failed: relative error "
                f"{report.relative_error:.4f} exceeds "
                f"{max_relative_error:.4f}"
            )
        return report

    def _loss(self, X: np.ndarray) -> float:
        return float(np.mean((self.reconstruct(X) - X) ** 2))

    def _validate_matrix(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                "X must be a 2D matrix: rows=samples, columns=features"
            )
        if values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {values.shape[1]}"
            )
        if values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("X must contain finite, non-empty data")
        return values


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    epochs: int = 500,
    validation_fraction: float = 0.2,
    learning_rate: float = 1e-2,
    seed: int = 0,
) -> tuple[Autoencoder, TrainingHistory, ReconstructionReport]:
    """Split rows, train an autoencoder, and evaluate held-out reconstruction."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(
            "X must contain at least two rows and be two-dimensional"
        )
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(values.shape[0])
    split = int(values.shape[0] * (1.0 - validation_fraction))
    split = min(max(split, 1), values.shape[0] - 1)
    train = values[order[:split]]
    validation = values[order[split:]]

    model = Autoencoder(
        input_dim=values.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    history = model.fit(train, epochs=epochs, validation_data=validation)
    report = model.evaluate(validation)
    return model, history, report
