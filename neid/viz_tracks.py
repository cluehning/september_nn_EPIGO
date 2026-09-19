from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Imports: prefer package-relative, but allow running as a script too.
# ---------------------------------------------------------------------
try:
    from .analysis import shannon_entropy, spectrum_1d
    from .io import ensure_dir
    from .viz import _plt, project_root
except ImportError:  # pragma: no cover
    # Allows: python neid/viz_tracks.py
    from neid.analysis import shannon_entropy, spectrum_1d  # type: ignore
    from neid.io import ensure_dir  # type: ignore
    from neid.viz import _plt, project_root  # type: ignore


# ---------------------------------------------------------------------
# BedGraph reading (gzipped)
# bedGraph format is 4 columns: chrom, start, end, value. Track lines
# may appear and should be skipped. [1](https://de.python-3.com/?p=49988)[2](https://stackoverflow.com/questions/73193119/python-filenotfounderror-winerror-2-the-system-cannot-find-the-file-specifie)
# ---------------------------------------------------------------------
def load_bedgraph_region_binned(
    path: str | Path,
    chrom: str,
    region_start: int,
    region_end: int,
    bin_size: int = 100,
) -> np.ndarray:
    """
    Load a genomic interval from a bedGraph.gz file into a dense 1D signal.

    The file is assumed to have columns:
      chrom  chromStart  chromEnd  value    (tab/space separated) [1](https://de.python-3.com/?p=49988)[2](https://stackoverflow.com/questions/73193119/python-filenotfounderror-winerror-2-the-system-cannot-find-the-file-specifie)

    To keep things fast, we bin the region into fixed-width bins and fill each
    bin with the average value across covered bases (missing bins stay 0).

    Args:
        path: path to .bedGraph.gz
        chrom: chromosome name (e.g., "chr1")
        region_start: 0-based start (inclusive)
        region_end: 0-based end (exclusive)
        bin_size: bin width in bp (e.g., 50/100/200/1000)

    Returns:
        1D array of length ceil((region_end - region_start)/bin_size)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Track file not found: {p}")

    if region_end <= region_start:
        raise ValueError("region_end must be > region_start")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    length_bp = region_end - region_start
    n_bins = int(np.ceil(length_bp / bin_size))

    # We'll accumulate total signal and total covered bp per bin
    tot = np.zeros(n_bins, dtype=float)
    cov = np.zeros(n_bins, dtype=float)

    def add_interval(s: int, e: int, v: float) -> None:
        """Add [s, e) with value v in bp coordinates relative to region_start."""
        if e <= s:
            return
        bs = s // bin_size
        be = (e - 1) // bin_size
        for b in range(bs, be + 1):
            bin_s = b * bin_size
            bin_e = min((b + 1) * bin_size, length_bp)
            overlap_s = max(s, bin_s)
            overlap_e = min(e, bin_e)
            w = float(overlap_e - overlap_s)
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

            # Clip to requested region
            s = max(s0, region_start)
            e = min(e0, region_end)
            if e <= s:
                continue

            # Convert to region-relative coordinates
            add_interval(s - region_start, e - region_start, v)

    # Average value per bin (0 if uncovered)
    x = np.zeros(n_bins, dtype=float)
    mask = cov > 0
    x[mask] = tot[mask] / cov[mask]
    return x


def _out_dir() -> Path:
    out = project_root() / "out"
    ensure_dir(out)
    return out


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def plot_mark_overlay(
    x1: np.ndarray,
    x2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    out_path: Path,
) -> Path:
    """Overlay two 1D signals in a single plot."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(x1, lw=1.2, label=label1, alpha=0.9)
    ax.plot(x2, lw=1.2, label=label2, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Signal (binned average)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_entropy_vs_window(
    x1: np.ndarray,
    x2: np.ndarray,
    label1: str,
    label2: str,
    out_path: Path,
    bins: int = 32,
) -> Path:
    """
    Compare Shannon entropy vs window size for two signals.
    Uses histogram counts per window as a distribution input to entropy.
    """
    plt = _plt()

    def entropy_curve(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Window sizes from ~8 bins up to len(x)//4, log-spaced
        max_w = max(8, len(x) // 4)
        ws = np.unique(np.logspace(np.log10(8), np.log10(max_w), 18).astype(int))
        es: list[float] = []
        ws2: list[int] = []
        for w in ws:
            if w < 4 or w > len(x):
                continue
            chunks = [x[i:i + w] for i in range(0, len(x) - w + 1, w)]
            if not chunks:
                continue
            counts = np.array([np.histogram(c, bins=bins)[0] for c in chunks])
            e = float(np.mean(shannon_entropy(counts, axis=1)))
            ws2.append(int(w))
            es.append(e)
        return np.array(ws2, dtype=int), np.array(es, dtype=float)

    w1, e1 = entropy_curve(x1)
    w2, e2 = entropy_curve(x2)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(w1, e1, marker="o", label=label1)
    ax.plot(w2, e2, marker="o", label=label2)
    ax.set_xscale("log")
    ax.set_title("Entropy vs window size (binned signal)")
    ax.set_xlabel("Window size (bins, log)")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_spectrum_compare(
    x1: np.ndarray,
    x2: np.ndarray,
    label1: str,
    label2: str,
    out_path: Path,
) -> Path:
    """Compare power spectra of two signals."""
    plt = _plt()
    f1, p1 = spectrum_1d(x1)
    f2, p2 = spectrum_1d(x2)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(f1, p1, lw=1.2, label=label1)
    ax.plot(f2, p2, lw=1.2, label=label2)
    ax.set_yscale("log")
    ax.set_title("Power spectrum comparison (log power)")
    ax.set_xlabel("Frequency (cycles per bin)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize epigenomic bedGraph.gz tracks (H3K27ac / H3K4me3)."
    )
    p.add_argument("--chrom", default="chr1", help="Chromosome (e.g. chr1)")
    p.add_argument("--start", type=int, default=0, help="Region start (0-based)")
    p.add_argument("--end", type=int, default=1_000_000, help="Region end (exclusive)")
    p.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Bin size in bp for downsampling (e.g. 50/100/200/1000)",
    )
    p.add_argument(
        "--k27ac",
        default=str(project_root() / "data" / "tracks" / "E001-H3K9m3.bedGraph.gz"),
        help="Path to E001-H3K9m3.pval.signal.bedGraph.gz",
    )
    p.add_argument(
        "--k4me3",
        default=str(project_root() / "data" / "tracks" / "E001-H3K36me3.bedGraph.gz"),
        help="Path to E001-H3K36me3.pval.signal.bedGraph.gz",
    )
    return p


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = build_argparser().parse_args(list(argv) if argv is not None else None)

    out = _out_dir()
    chrom = str(args.chrom)
    region_start = int(args.start)
    region_end = int(args.end)
    bin_size = int(args.bin_size)

    log.info("Loading region %s:%d-%d (bin=%d bp)", chrom, region_start, region_end, bin_size)

    x_k27 = load_bedgraph_region_binned(
        args.k27ac, chrom=chrom, region_start=region_start, region_end=region_end, bin_size=bin_size
    )
    x_k4 = load_bedgraph_region_binned(
        args.k4me3, chrom=chrom, region_start=region_start, region_end=region_end, bin_size=bin_size
    )

    p1 = plot_mark_overlay(
        x_k27,
        x_k4,
        label1="E001 H3K9me9",
        label2="E001 H3K36me3",
        title=f"E001 marks overlay: {chrom}:{region_start}-{region_end} (bin {bin_size}bp)",
        out_path=out / "E001_marks_overlay.png",
    )
    p2 = plot_entropy_vs_window(
        x_k27,
        x_k4,
        label1="H3K9me3",
        label2="H3K36me3",
        out_path=out / "E001_entropy_vs_window.png",
    )
    p3 = plot_spectrum_compare(
        x_k27,
        x_k4,
        label1="H3K9me3",
        label2="H3K36me3",
        out_path=out / "E001_spectrum_compare.png",
    )

    print(f"Saved: {p1.resolve()}")
    print(f"Saved: {p2.resolve()}")
    print(f"Saved: {p3.resolve()}")


if __name__ == "__main__":
    main()
