from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Prefer package-relative imports, but allow running as a script too
try:
    from .analysis import shannon_entropy, spectrum_1d
except ImportError:  # pragma: no cover
    from neid.analysis import shannon_entropy, spectrum_1d  # type: ignore


__all__ = [
    "Track",
    "TrackFeatures",
    "extract_track_features",
    "load_bigwig_region_binned",
    "load_tracks_json_gz",
    "load_bedgraph_region_dense",
    "load_bedgraph_region_binned",
]


# ---------------------------------------------------------------------
# 1) Segment tracks (array index based) - ONLY if you have JSON.gz tracks
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Track:
    """
    A generic segment track for 1D array signals.

    start/end are indices into a 1D array x, using Python slicing:
      segment = x[start:end]
    """
    track_id: str
    start: int
    end: int

    def slice(self, x: np.ndarray) -> np.ndarray:
        return x[self.start:self.end]


def load_tracks_json_gz(path: str | Path) -> list[Track]:
    """
    Load segment tracks from a gzipped JSON file.

    Expected JSON:
      [
        {"track_id": "T1", "start": 10, "end": 100},
        {"track_id": "T2", "start": 200, "end": 350}
      ]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Track file not found: {p}")

    with gzip.open(p, "rt", encoding="utf-8") as f:
        data: Any = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Track JSON must be a list of objects")

    tracks: list[Track] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Track entry {i} must be an object")

        try:
            track_id = str(item["track_id"])
            start = int(item["start"])
            end = int(item["end"])
        except KeyError as exc:
            raise ValueError(f"Track entry {i} missing key: {exc}") from exc

        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid track bounds for {track_id}: start={start}, end={end}"
            )

        tracks.append(Track(track_id=track_id, start=start, end=end))

    return tracks


@dataclass(frozen=True, slots=True)
class TrackFeatures:
    track_id: str
    mean: float
    std: float
    entropy: float
    dominant_freq: float


def extract_track_features(
    x: np.ndarray,
    tracks: Iterable[Track],
) -> list[TrackFeatures]:
    """
    Compute features per segment track using existing analysis utilities.

    Notes:
    - entropy is computed from a histogram (32 bins) of the segment
    - dominant_freq comes from argmax of the rFFT power spectrum
    """
    x = np.asarray(x, dtype=float).ravel()
    features: list[TrackFeatures] = []

    for t in tracks:
        seg = t.slice(x)
        if seg.size < 4:
            continue

        hist = np.histogram(seg, bins=32)[0]
        ent = float(shannon_entropy(hist))

        freqs, power = spectrum_1d(seg)
        dom_freq = float(freqs[int(np.argmax(power))])

        features.append(
            TrackFeatures(
                track_id=t.track_id,
                mean=float(np.mean(seg)),
                std=float(np.std(seg)),
                entropy=ent,
                dominant_freq=dom_freq,
            )
        )

    return features


# ---------------------------------------------------------------------
# 2) bedGraph tracks (genomic signal tracks) - THIS is your E071 data
# ---------------------------------------------------------------------
def load_bigwig_region_binned(
    path: str | Path,
    chrom: str,
    region_start: int,
    region_end: int,
    bin_size: int = 100,
) -> np.ndarray:
    """Load and average a BigWig interval into fixed-width bins.

    BigWig values are aggregated by base-pair overlap, matching the behavior
    of ``load_bedgraph_region_binned`` without requiring conversion first.
    """
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Direct BigWig loading requires pyBigWig. "
            "Install it with: python -m pip install pyBigWig"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BigWig file not found: {p}")
    if region_end <= region_start:
        raise ValueError("Invalid region: end must be > start")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    length_bp = region_end - region_start
    n_bins = int(np.ceil(length_bp / bin_size))
    bigwig = pyBigWig.open(str(p))
    try:
        if chrom not in bigwig.chroms():
            raise ValueError(f"Chromosome not found in BigWig: {chrom}")
        values = bigwig.stats(
            chrom,
            region_start,
            region_end,
            nBins=n_bins,
            type="mean",
        )
    finally:
        bigwig.close()

    result = np.zeros(n_bins, dtype=float)
    for index, value in enumerate(values):
        if value is not None and np.isfinite(value):
            result[index] = float(value)
    return result


def load_bedgraph_region_dense(
    path: str | Path,
    chrom: str,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    """
    Load a genomic interval from a bedGraph.gz file as a dense 1D array.

    bedGraph lines are typically:
      chrom  chromStart  chromEnd  value

    WARNING:
    This allocates one element per base pair in the region.
    Use only for SMALL regions (e.g., <= a few million bp).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bedGraph file not found: {p}")

    length = region_end - region_start
    if length <= 0:
        raise ValueError("Invalid region: end must be > start")

    x = np.zeros(length, dtype=float)

    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith(("track", "#", "browser")):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            c, start_s, end_s, val_s = parts[:4]
            if c != chrom:
                continue

            try:
                s0 = int(start_s)
                e0 = int(end_s)
                v = float(val_s)
            except ValueError:
                continue

            s = max(s0, region_start)
            e = min(e0, region_end)
            if s < e:
                x[s - region_start:e - region_start] = v

    return x


def load_bedgraph_region_binned(
    path: str | Path,
    chrom: str,
    region_start: int,
    region_end: int,
    bin_size: int = 100,
) -> np.ndarray:
    """
    Load a genomic interval from a bedGraph.gz file as a BINNED 1D array.

    Instead of bp resolution, we create bins of bin_size bp and compute the
    average signal per bin. This is the recommended loader for large regions.

    Returns:
        Array of length ceil((region_end - region_start) / bin_size)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bedGraph file not found: {p}")

    if region_end <= region_start:
        raise ValueError("Invalid region: end must be > start")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    length_bp = region_end - region_start
    n_bins = int(np.ceil(length_bp / bin_size))

    tot = np.zeros(n_bins, dtype=float)
    cov = np.zeros(n_bins, dtype=float)

    def add_interval(rel_s: int, rel_e: int, v: float) -> None:
        if rel_e <= rel_s:
            return
        b0 = rel_s // bin_size
        b1 = (rel_e - 1) // bin_size
        for b in range(b0, b1 + 1):
            bin_s = b * bin_size
            bin_e = min((b + 1) * bin_size, length_bp)
            s = max(rel_s, bin_s)
            e = min(rel_e, bin_e)
            w = float(e - s)
            if w > 0:
                tot[b] += v * w
                cov[b] += w

    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith(("track", "#", "browser")):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            c, start_s, end_s, val_s = parts[:4]
            if c != chrom:
                continue

            try:
                s0 = int(start_s)
                e0 = int(end_s)
                v = float(val_s)
            except ValueError:
                continue

            s = max(s0, region_start)
            e = min(e0, region_end)
            if e <= s:
                continue

            add_interval(s - region_start, e - region_start, v)

    x = np.zeros(n_bins, dtype=float)
    m = cov > 0
    x[m] = tot[m] / cov[m]
    return x


if __name__ == "__main__":  # pragma: no cover
    print(
        "neid.tracks is a library module. Run visualizations with:\n"
        "  python -m neid.viz_tracks\n"
    )
