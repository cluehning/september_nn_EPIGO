# EPIGO Neural Network Extension

EPIGO treats epigenetic tracks as structured objects rather than as unrelated
lists of numbers. This project adds a neural representation-learning layer to
that analysis: it measures genomic signal structure, compresses the
measurements into a smaller representation, rebuilds them, and checks how much
information was preserved.

The current reproducible example uses two E071 Roadmap tracks:

- `E071_H3K27ac.bedGraph.gz`
- `E071_H3K4me3.bedGraph.gz`

The code is intentionally interpretable. The neural network does not replace
the EPIGO measurements; it learns from them.

## The Lego explanation

Imagine that each epigenetic track is a long Lego construction laid out along
the genome. A signal value at a genomic position is part of the construction,
and a different histone mark is a different Lego color.

### Stage 1: Get the Lego constructions

The input files contain signal intervals such as:

```text
chr1    0       9946    0.03128
chr1    9946    9950    0.12771
```

These are not yet convenient neural-network inputs. They are long,
variable-length genomic constructions.

Implemented by [`neid.tracks`](neid/tracks.py):

- read compressed bedGraph tracks,
- select a chromosome and genomic interval,
- average signal into fixed-width bins.

For the included run, every 100 base pairs becomes one bin.

### Stage 2: Cut the construction into comparable windows

The binned tracks are cut into overlapping windows. A window is a small Lego
section that can be compared with other sections.

The default run uses:

```text
bin size:    100 bp
window size: 128 bins = 12,800 bp
step:        64 bins = 6,400 bp
```

Two views of each window are retained:

1. A raw tensor with shape `(windows, tracks, bins)`. This preserves the
   physical track-by-position structure.
2. A feature row used by the statistical and neural models.

Simply reshaping a vector into a tensor would not create information. The
tensor is meaningful here because its axes are explicitly tracks and genomic
position.

### Stage 3: Measure each Lego section with EPIGO

For every track in every window, EPIGO records four interpretable properties:

- **Mean:** the average signal level.
- **Standard deviation:** how much the signal varies.
- **Entropy:** how diverse or irregular the signal distribution is.
- **Dominant frequency:** the strongest repeating spatial scale.

With two tracks, each genomic window becomes an eight-dimensional row:

```text
H3K27ac: mean, standard deviation, entropy, dominant frequency
H3K4me3: mean, standard deviation, entropy, dominant frequency
```

This is the EPIGO Lego inventory: instead of handing the model every tiny
brick, we give it a structured description of each section.

The feature calculations are implemented in
[`neid.analysis`](neid/analysis.py) and
`build_epigo_window_features` in [`neid.neural`](neid/neural.py).

### Stage 4: Put measurements on comparable scales

Mean signal, entropy, and frequency do not naturally use the same numerical
scale. The feature matrix is therefore z-score normalized before learning.

This is like weighing and labeling every Lego type using a common measuring
system. It prevents a large-valued feature from dominating simply because of
its units.

### Stage 5: Try a linear Lego blueprint with PCA

PCA/SVD is the first baseline. It asks:

> Can a small number of linear directions preserve most of the information?

The PCA coordinates are a linear change of basis. If PCA performs as well as
the neural network, the current feature space may already be mostly linear.

Implemented in [`neid/reduction.py`](neid/reduction.py).

### Stage 6: Try explicit learned Lego blocks with dictionary learning

Dictionary learning creates a small set of reusable blocks. Each window is
represented by coefficients saying how much of each block it contains:

```text
window ~= block_1 * coefficient_1
       + block_2 * coefficient_2
       + ...
```

This is the most literal version of the Lego analogy. It is included as a
baseline, not assumed to be the best model.

Also implemented in [`neid/reduction.py`](neid/reduction.py).

### Stage 7: Let the neural network learn a nonlinear blueprint

The autoencoder has two parts:

```text
8 EPIGO features -> hidden layer -> latent blueprint -> hidden layer -> 8 rebuilt features
```

The **encoder** compresses each window into a three-dimensional latent vector.
This is the learned blueprint: a compact representation of the combination of
signal level, variability, entropy, and spectral behavior across both marks.

The **decoder** expands that blueprint and tries to rebuild the original eight
features.

During training, the model changes its weights to reduce reconstruction error:

```text
reconstruction error = rebuilt features - original features
```

The model is not proving that the biology is correct. It is checking whether
its compact blueprint retained enough information to reconstruct the measured
EPIGO structure. Held-out windows provide a check against simply memorizing
the training windows.

Implemented in [`neid/neural.py`](neid/neural.py).

## What the code currently does

The complete pipeline is:

```text
bedGraph tracks
    -> fixed genomic bins
    -> overlapping windows
    -> raw track tensor
    -> EPIGO descriptors
    -> z-score normalization
    -> PCA baseline
    -> dictionary baseline
    -> neural autoencoder
    -> held-out reconstruction comparison
```

The neural network currently learns an autoencoder representation. It is not a
supervised classifier yet: there are no biological class labels in this
dataset. Its current task is reconstruction and representation learning.

## Quick run

Run from the project root:

```powershell
python -m neid.pipeline --epochs 1500 --output-dir out/nn_epigo
```

For a faster smoke test:

```powershell
python -m neid.pipeline --epochs 300 --output-dir out/nn_epigo_quick
```

The pipeline uses the included `.bedGraph.gz` files, so `pyBigWig` is not
needed for this command.

To run the older visualization demos:

```powershell
python -m neid.viz
python -m neid.viz_tracks --chrom chr1 --start 0 --end 100000 --bin-size 100
```

## Outputs

The pipeline writes results to `out/nn_epigo/`:

- `metrics.json`: configuration, losses, explained variance, and errors.
- `signals_binned.npy`: the binned input tracks.
- `window_tensor.npy`: raw data shaped as `(windows, tracks, bins)`.
- `window_starts_bp.npy`: genomic start coordinate for every window.
- `epigo_features_raw.npy`: interpretable feature rows before normalization.
- `epigo_features_normalized.npy`: model input matrix.
- `pca_latent.npy`: PCA coordinates.
- `dictionary_codes_*.npy`: sparse Lego-block coefficients.
- `autoencoder_latent.npy`: neural latent blueprints.
- `autoencoder_validation_reconstruction.npy`: rebuilt held-out features.
- `autoencoder_loss.png`: training and validation loss.
- `latent_space.png`: two-dimensional view of the learned latent space.
- `method_comparison.png`: PCA, dictionary, and autoencoder errors.

## E071 result

The complete run on the first 1 Mb of `chr1` produced 155 windows and eight
EPIGO features per window:

```text
PCA relative error:          0.3753
Dictionary relative error:   0.9528
Autoencoder relative error:  0.3780
Autoencoder tolerance:       passed at 0.5
Autoencoder loss:             1.4605 -> 0.0936
```

PCA and the autoencoder perform almost identically on this small feature set.
That is an informative result: the current EPIGO descriptor space is already
mostly linear. The neural network becomes more valuable when more marks,
regions, or biological labels are added and nonlinear interactions become
harder for PCA to represent.

## Project structure

```text
NN_EPIGO/
├── data/tracks/
│   ├── E071_H3K27ac.bedGraph.gz
│   ├── E071_H3K4me3.bedGraph.gz
│   └── E071_tracks.json.gz
├── neid/
│   ├── analysis.py       # entropy, spectra, matrices, normalization
│   ├── tracks.py         # genomic track loading and binning
│   ├── neural.py         # EPIGO features and autoencoder
│   ├── reduction.py      # PCA/SVD and dictionary baselines
│   ├── pipeline.py       # reproducible end-to-end run
│   ├── viz.py            # general plots
│   └── viz_tracks.py     # track comparison plots
├── out/nn_epigo/         # generated results
└── README.md
```

## Data source

The example tracks come from the NIH Roadmap Epigenomics Project. Roadmap
signal files are available at:

```text
https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/pval/
```

The loaders expect standard bedGraph columns:

```text
chromosome    start    end    value
```

Compressed `.bedGraph.gz` files are supported directly. BigWig files are also
supported through `load_bigwig_region_binned`, but direct BigWig loading
requires a compatible `pyBigWig` installation.

## Scientific interpretation

EPIGO does not claim that a low reconstruction error proves a biological
hypothesis. Reconstruction only shows that the learned representation
preserves the measured signal descriptors. Biological interpretation still
requires comparisons across marks, regions, conditions, and, eventually,
independent labels or experimental evidence.

## Development Note: 
Parts of the codebase were created with AI assistance ("vibe coding"), but the underlying ideas, research direction, experimental design, mathematical reasoning, and interdisciplinary extensions are my own. AI was used as an implementation and exploration tool, with all major decisions, modifications, and interpretations guided by the author.
