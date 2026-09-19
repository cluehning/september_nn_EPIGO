from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def _missing(dep: str, extra: str = "") -> ImportError:
    msg = f"Missing dependency '{dep}'. Install it with: pip install {dep}"
    if extra:
        msg = f"{msg} {extra}"
    return ImportError(msg)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML into dict."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _missing("pyyaml") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data)}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON into dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object, got {type(data)}")
    return data


def write_json(path: Path, obj: Mapping[str, Any], indent: int = 2) -> None:
    """Write JSON, ensuring parent exists."""
    ensure_dir(path.parent)
    text = json.dumps(obj, indent=indent, sort_keys=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> "Any":
    """Read CSV via pandas."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise _missing("pandas") from exc
    return pd.read_csv(path)


def write_csv(path: Path, df: "Any") -> None:
    """Write CSV via pandas."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def read_parquet(path: Path) -> "Any":
    """Read Parquet via pandas (requires pyarrow or fastparquet)."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise _missing("pandas") from exc
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise _missing("pyarrow", " (recommended) or pip install fastparquet") from exc


def write_parquet(path: Path, df: "Any") -> None:
    """Write Parquet via pandas."""
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise _missing("pyarrow", " (recommended) or pip install fastparquet") from exc


def read_npy(path: Path) -> np.ndarray:
    """Read NumPy .npy array."""
    return np.load(path)


def write_npy(path: Path, arr: np.ndarray) -> None:
    """Write NumPy .npy array."""
    ensure_dir(path.parent)
    np.save(path, arr)
