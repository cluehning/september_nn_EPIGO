from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .analysis import normalize_matrix
from .io import ensure_dir
from .neural import (
    build_epigo_window_features,
    build_signal_windows,
    Autoencoder,
)
from .reduction import fit_pca, learn_dictionary
from .tracks import load_bedgraph_region_binned
from .viz import _plt


def run_pipeline(
    track_paths: tuple[Path, Path],
    output_dir: Path,
    chrom: str = "chr1",
    start: int = 0,
    end: int = 1_000_000,
    bin_size: int = 100,
    window_size: int = 128,
    step: int = 64,
    latent_dim: int = 3,
    hidden_dim: int = 16,
    epochs: int = 1_500,
    seed: int = 42,
) -> dict[str, object]:
    """Run EPIGO descriptors, PCA, dictionary learning, and autoencoding."""
    ensure_dir(output_dir)
    signals = np.vstack(
        [
            load_bedgraph_region_binned(path, chrom, start, end, bin_size)
            for path in track_paths
        ]
    )
    raw_features = build_epigo_window_features(signals, window_size, step)
    features = normalize_matrix(raw_features, method="zscore", axis=0)
    raw_windows = build_signal_windows(signals, window_size, step)
    window_tensor = raw_windows.reshape(
        raw_windows.shape[0], signals.shape[0], window_size
    )
    window_starts = (
        np.arange(raw_windows.shape[0], dtype=int) * step * bin_size
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(features.shape[0])
    split = min(max(int(features.shape[0] * 0.8), 1), features.shape[0] - 1)
    train_idx = order[:split]
    validation_idx = order[split:]
    train = features[train_idx]
    validation = features[validation_idx]

    pca = fit_pca(train, n_components=min(latent_dim, features.shape[1]))
    pca_validation = pca.reconstruct(validation)
    pca_error = _relative_error(validation, pca_validation)

    dictionary = learn_dictionary(
        train,
        n_atoms=min(8, train.shape[0]),
        sparsity=min(2, train.shape[0]),
        iterations=30,
        seed=seed,
    )
    dictionary_validation_codes = _sparse_encode(
        validation,
        dictionary.dictionary,
        sparsity=min(2, dictionary.dictionary.shape[0]),
    )
    dictionary_validation = dictionary_validation_codes @ dictionary.dictionary
    dictionary_error = _relative_error(validation, dictionary_validation)

    model = Autoencoder(
        input_dim=features.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=0.01,
        seed=seed,
    )
    history = model.fit(train, epochs=epochs, validation_data=validation)
    autoencoder_validation = model.reconstruct(validation)
    autoencoder_error = _relative_error(validation, autoencoder_validation)

    np.save(output_dir / "signals_binned.npy", signals)
    np.save(output_dir / "window_tensor.npy", window_tensor)
    np.save(output_dir / "window_starts_bp.npy", window_starts)
    np.save(output_dir / "epigo_features_raw.npy", raw_features)
    np.save(output_dir / "epigo_features_normalized.npy", features)
    np.save(output_dir / "pca_latent.npy", pca.transform(features))
    np.save(output_dir / "dictionary_codes_train.npy", dictionary.codes)
    np.save(
        output_dir / "dictionary_codes_validation.npy",
        dictionary_validation_codes,
    )
    np.save(output_dir / "autoencoder_latent.npy", model.encode(features))
    np.save(
        output_dir / "autoencoder_validation_reconstruction.npy",
        autoencoder_validation,
    )
    np.save(output_dir / "validation_indices.npy", validation_idx)
    _write_plots(
        output_dir,
        history,
        model.encode(features),
        validation_idx,
        pca_error,
        dictionary_error,
        autoencoder_error,
    )

    metrics: dict[str, object] = {
        "chrom": chrom,
        "region_start": start,
        "region_end": end,
        "bin_size": bin_size,
        "window_size": window_size,
        "step": step,
        "n_tracks": int(signals.shape[0]),
        "n_windows": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "train_windows": int(train.shape[0]),
        "validation_windows": int(validation.shape[0]),
        "pca_relative_error": pca_error,
        "dictionary_relative_error": dictionary_error,
        "autoencoder_relative_error": autoencoder_error,
        "autoencoder_within_50_percent": bool(autoencoder_error <= 0.5),
        "autoencoder_initial_loss": float(history.train_loss[0]),
        "autoencoder_final_loss": float(history.train_loss[-1]),
        "explained_variance_ratio": pca.explained_variance_ratio.tolist(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the reproducible EPIGO neural pipeline."
    )
    parser.add_argument(
        "--track-a",
        type=Path,
        default=Path("data/tracks/E071_H3K27ac.bedGraph.gz"),
    )
    parser.add_argument(
        "--track-b",
        type=Path,
        default=Path("data/tracks/E071_H3K4me3.bedGraph.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/nn_epigo")
    )
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1_000_000)
    parser.add_argument("--bin-size", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_argparser().parse_args(
        list(argv) if argv is not None else None
    )
    metrics = run_pipeline(
        (args.track_a, args.track_b),
        args.output_dir,
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        bin_size=args.bin_size,
        window_size=args.window_size,
        step=args.step,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


def _relative_error(target: np.ndarray, prediction: np.ndarray) -> float:
    difference = prediction - target
    scale = max(float(np.linalg.norm(target)), np.finfo(float).eps)
    return float(np.linalg.norm(difference) / scale)


def _sparse_encode(
    X: np.ndarray, dictionary: np.ndarray, sparsity: int
) -> np.ndarray:
    correlations = X @ dictionary.T
    codes = np.zeros((X.shape[0], dictionary.shape[0]), dtype=float)
    active = np.argpartition(np.abs(correlations), -sparsity, axis=1)[
        :, -sparsity:
    ]
    rows = np.arange(X.shape[0])[:, None]
    codes[rows, active] = correlations[rows, active]
    return codes


def _write_plots(
    output_dir: Path,
    history: object,
    latent: np.ndarray,
    validation_idx: np.ndarray,
    pca_error: float,
    dictionary_error: float,
    autoencoder_error: float,
) -> None:
    plt = _plt()
    training_loss = history.train_loss
    validation_loss = history.validation_loss

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(training_loss, label="train")
    ax.plot(
        np.linspace(0, len(training_loss) - 1, len(validation_loss)),
        validation_loss,
        label="validation",
    )
    ax.set(
        title="Autoencoder reconstruction loss",
        xlabel="Epoch",
        ylabel="Mean squared error",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "autoencoder_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    if latent.shape[1] >= 2:
        ax.scatter(latent[:, 0], latent[:, 1], s=16, alpha=0.7)
        ax.set(
            xlabel="Latent 1",
            ylabel="Latent 2",
            title="Learned EPIGO latent space",
        )
    else:
        ax.plot(latent[:, 0], marker=".", linewidth=0)
        ax.set(
            xlabel="Window",
            ylabel="Latent 1",
            title="Learned EPIGO latent space",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "latent_space.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["PCA", "Dictionary", "Autoencoder"]
    errors = [pca_error, dictionary_error, autoencoder_error]
    ax.bar(names, errors, color=["#718096", "#d69e2e", "#464feb"])
    ax.axhline(0.5, color="#b83280", linestyle="--", label="0.5 tolerance")
    ax.set(
        title="Held-out relative reconstruction error", ylabel="Relative error"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "method_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
