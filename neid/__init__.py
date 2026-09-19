"""
NEID - simple, robust analysis utilities.

This package is intentionally minimal and "fail-proof":
- flat layout (no src/ import surprises)
- defensive imports with helpful error messages
- strong typing and clear APIs
"""

from __future__ import annotations

from .analysis import (
    build_feature_matrix,
    normalize_matrix,
    shannon_entropy,
    spectrum_1d,
)
from .config import Config, load_config
from .io import (
    ensure_dir,
    read_csv,
    read_json,
    read_npy,
    read_parquet,
    read_yaml,
    write_csv,
    write_json,
    write_npy,
    write_parquet,
)
from .neural import (
    Autoencoder,
    ReconstructionReport,
    TrainingHistory,
    build_epigo_window_features,
    build_signal_windows,
    train_autoencoder,
)
from .reduction import DictionaryResult, PCAResult, fit_pca, learn_dictionary
from .tracks import load_bigwig_region_binned

__all__ = [
    "Autoencoder",
    "build_epigo_window_features",
    "build_signal_windows",
    "Config",
    "DictionaryResult",
    "build_feature_matrix",
    "ensure_dir",
    "load_config",
    "load_bigwig_region_binned",
    "normalize_matrix",
    "PCAResult",
    "read_csv",
    "read_json",
    "read_npy",
    "read_parquet",
    "read_yaml",
    "ReconstructionReport",
    "shannon_entropy",
    "TrainingHistory",
    "train_autoencoder",
    "fit_pca",
    "learn_dictionary",
    "spectrum_1d",
    "write_csv",
    "write_json",
    "write_npy",
    "write_parquet",
]

__version__ = "0.1.0"
