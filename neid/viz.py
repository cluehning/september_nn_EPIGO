from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .analysis import (
    build_feature_matrix,
    normalize_matrix,
    shannon_entropy,
    spectrum_1d,
)
from .io import ensure_dir

log = logging.getLogger(__name__)


def _plt():
    """Lazy matplotlib import with clear error."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: python -m pip install matplotlib"
        ) from exc
    return plt


def project_root() -> Path:
    """Resolve project root robustly."""
    return Path(__file__).resolve().parents[1]


def out_dir() -> Path:
    """Central output directory."""
    out = project_root() / "out"
    ensure_dir(out)
    return out


# ---------------------------------------------------------------------
# Core visualizations
# ---------------------------------------------------------------------

def plot_time_series(x: np.ndarray) -> Path:
    """Plot raw time series."""
    plt = _plt()
    path = out_dir() / "time_series.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, lw=1.2)
    ax.set_title("Time Series")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_power_spectrum(x: np.ndarray) -> Path:
    """Plot log-scaled power spectrum."""
    plt = _plt()
    path = out_dir() / "power_spectrum.png"

    freqs, power = spectrum_1d(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(freqs, power, lw=1.2)
    ax.set_yscale("log")
    ax.set_title("Power Spectrum (log scale)")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_entropy_over_windows(x: np.ndarray) -> Path:
    """
    Entropy vs window size.

    This is extremely useful for detecting structure vs noise.
    """
    plt = _plt()
    path = out_dir() / "entropy_vs_window.png"

    window_sizes = np.unique(
        np.logspace(1, np.log10(len(x) // 4), num=20).astype(int)
    )

    entropies: list[float] = []
    for w in window_sizes:
        chunks = [
            x[i:i + w]
            for i in range(0, len(x) - w + 1, w)
        ]
        if not chunks:
            continue
        counts = np.array([np.histogram(c, bins=32)[0] for c in chunks])
        e = np.mean(shannon_entropy(counts, axis=1))
        entropies.append(e)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window_sizes[:len(entropies)], entropies, marker="o")
    ax.set_xscale("log")
    ax.set_title("Shannon Entropy vs Window Size")
    ax.set_xlabel("Window size (log)")
    ax.set_ylabel("Entropy (bits)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


def plot_feature_heatmap() -> Path:
    """
    Feature matrix heatmap (normalized).

    Demonstrates multivariate structure clearly.
    """
    plt = _plt()
    path = out_dir() / "feature_heatmap.png"

    # Example synthetic features (replace with real ones later)
    rng = np.random.default_rng(0)
    rows = [
        {
            "mean": float(np.mean(rng.normal(size=200))),
            "std": float(np.std(rng.normal(size=200))),
            "entropy": float(
                shannon_entropy(
                    np.histogram(rng.normal(size=200), bins=32)[0]
                )
            ),
        }
        for _ in range(20)
    ]

    fm = build_feature_matrix(rows)
    Xn = normalize_matrix(fm.X, method="zscore", axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Xn, aspect="auto", cmap="viridis")

    ax.set_title("Normalized Feature Matrix")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample")

    ax.set_xticks(range(len(fm.feature_names)))
    ax.set_xticklabels(fm.feature_names, rotation=45, ha="right")

    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    log.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run full visualization suite on demo data.

    Replace `x` with real EPIGO signal when ready.
    """
    logging.basicConfig(level=logging.INFO)

    rng = np.random.default_rng(42)
    x = rng.normal(size=2048)

    plot_time_series(x)
    plot_power_spectrum(x)
    plot_entropy_over_windows(x)
    plot_feature_heatmap()

    print(f"All plots written to: {out_dir().resolve()}")


if __name__ == "__main__":
    main()
