from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import read_yaml


@dataclass(frozen=True, slots=True)
class Config:
    """
    Central configuration.

    Keep file paths as Path objects.
    Store only primitives or nested dicts for easy serialization.
    """

    project_root: Path
    data_dir: Path
    output_dir: Path
    random_seed: int = 0
    log_level: str = "INFO"

    def resolved(self) -> "Config":
        """Return a config with absolute, resolved paths."""
        return Config(
            project_root=self.project_root.resolve(),
            data_dir=self.data_dir.resolve(),
            output_dir=self.output_dir.resolve(),
            random_seed=int(self.random_seed),
            log_level=str(self.log_level).upper(),
        )


def setup_logging(level: str = "INFO") -> None:
    """Idempotent logging setup."""
    level_name = str(level).upper()
    numeric = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(numeric)
        return

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_config(path: str | Path, project_root: str | Path | None = None) -> Config:
    """
    Load config from YAML or JSON.

    - If project_root not provided, infer it as the parent of config file.
    - Returns resolved paths.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    root = Path(project_root) if project_root is not None else cfg_path.parent

    raw: Mapping[str, Any]
    if cfg_path.suffix.lower() in {".yml", ".yaml"}:
        raw = read_yaml(cfg_path)
    elif cfg_path.suffix.lower() == ".json":
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("JSON config root must be an object")
    else:
        raise ValueError("Config must be .yaml/.yml or .json")

    data_dir = Path(raw.get("data_dir", root / "data"))
    output_dir = Path(raw.get("output_dir", root / "out"))

    cfg = Config(
        project_root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        random_seed=int(raw.get("random_seed", 0)),
        log_level=str(raw.get("log_level", "INFO")),
    ).resolved()

    setup_logging(cfg.log_level)
    logging.getLogger(__name__).info("Loaded config from %s", cfg_path)
    return cfg
