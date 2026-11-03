# 2026_Pallares: Organoid Drug Screen Image Analysis

Image analysis pipeline for a time-lapse, multi-well organoid drug screen (DENV-2 infection with antiviral, immune-modulator and toxicity compound panels). It takes raw high-content microscope images and produces:

- stitched per-well, per-timepoint montages
- organoid segmentation masks for every timepoint (DINOv2 features plus a trained linear classifier)
- per-well intensity and organoid-area measurements over time
- interactive Plotly plots per compound panel, linked to a napari viewer for looking at the underlying images

## Pipeline overview

```
Raw microscope TIFFs (rXXcYYfZZpNN-chN...tiff)
        │
        ▼
A-MedianBackgroundFinder.ipynb ──► ch1_median.tif, ch2_median.tif, ch3_median.tif
        │
        ▼
B-MontageViewer.ipynb ──► concatenated/<well>_montage.tif      (T, Y, X, 3)
        │
        ▼
organoid_segmentation/ ──► concatenated/<well>_organoid_labels.tiff
        │
        ▼
C-Compute.ipynb (SLURM) ──► Quantification.csv
                            <well>_montage_ds.tiff, <well>_organoid_labels_ds.tiff
        │
        ▼
D-Display.ipynb + Map.csv ──► PerWell.csv, Aggregated.csv, *.html plots
```

## Channels

| Index | Raw name | Contents (as labeled in `D-Display`) |
|-------|----------|--------------------------------------|
| 0 | `ch1` | nucSpot (nuclear stain) |
| 1 | `ch2` | mScarlet (fluorescent reporter) |
| 2 | `ch3` | Transmitted light, used for organoid segmentation |

## Steps

### A. `A-MedianBackgroundFinder.ipynb`: flat-field / background estimate
Samples every 1000th raw image for each channel under `<root_dir>/*/Images/` and writes a per-pixel median image (`ch1_median.tif`, `ch2_median.tif`, `ch3_median.tif`). Only the ch1 and ch2 medians are used later.

### B. `B-MontageViewer.ipynb`: stitching
For each well, loads the 9 fields (`_01` … `_09`) from `projected_good/`, subtracts the median background from channels 0 and 1, reorders the fields into the correct 3×3 layout, and builds a montage with `skimage.util.montage` (3 × 1080 px = 3240 × 3240 px). Montages from every acquisition directory matching `../2026*/` are concatenated along time and saved as `concatenated/<well>_montage.tif`.

The "Testing" section at the bottom is scratch work, including an earlier Cellpose (`cpdino`) segmentation attempt.

### Organoid segmentation (`organoid_segmentation/`)
Each transmitted-light plane is turned into a well mask, flat-field normalized (divided by a σ=80 Gaussian background), padded to a multiple of 14, and passed through **DINOv2 ViT-B/14 (with registers)** at full resolution. That gives a 232×232 grid of 768-dim patch tokens, one per 14×14 px patch. A logistic-regression head classifies each token as organoid or background. The probability map is upsampled, thresholded at 0.5, limited to the well, and cleaned up (min object 3000 px, holes < 5000 px filled), then labeled.

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `extract_fr.py` | Cache DINOv2 features for every montage in the current directory | `*_montage.tif` | `output/feats_fr/<well>_t<t>.npz` (`feats`, `norm`, `well`) |
| `extract_two.py` | Same, for the hard-negative wells I10 and H08 (fibrous mesh, glare) | `SRC/{I10,H08}_montage.tif` | `output/feats_fr/…` |
| `train_fr.py` | Train the classifier head from hand-picked positive/negative boxes, verified organoid interiors from a previous round, and random smooth-background tokens. Prints 5-fold balanced accuracy. | cached features, `output/{A02,E16,K22}_organoid_labels.tif` | `output/organoid_head_fr.joblib` |
| `predict_fr.py` | Predict masks from cached features (assumes 16 timepoints) and write QC figures | cached features, head | `output/<well>_organoid_mask.tif`, `output/<well>_organoid_labels.tif`, `output/qc3/<well>_qc.png` |
| `batch_labels.py` | **Production run.** Computes features on the fly and applies the head to every montage in `SRC`. Can be resumed (skips wells that already have output), writes atomically through a `.part` file, and prefetches the next file while the GPU works. | `SRC/*_montage.tif`, head | `SRC/<well>_organoid_labels.tiff` (uint16, zstd), `output/qc_batch/<well>.jpg` |

Usage:
```bash
cd organoid_segmentation
python batch_labels.py          # all wells
python batch_labels.py A02      # a single well
```
Training box coordinates in `train_fr.py` are `(well, t, y0, y1, x0, x1)` in full-resolution montage pixels.

### C. `C-Compute.ipynb`: quantification (SLURM cluster)
Starts a `dask_jobqueue.SLURMCluster` (512 single-core, 16 GB jobs on the `compute` partition). For each montage it:
1. downsamples 4× (to 810 × 810)
2. applies a rolling-ball-style top-hat background subtraction (radius 60) to channels 0 and 1
3. resizes the organoid labels to match
4. saves `_montage_ds.tiff` and `_organoid_labels_ds.tiff`
5. records per-timepoint totals: `c0`, `c1`, `mask` (organoid area in px), `masked_c0`, `masked_c1`

All results go to `Quantification.csv`.

### D. `D-Display.ipynb`: plate map merge and plots
Joins `Quantification.csv` with the plate map `Map.csv` (columns `Row`, `Column`, `Infection`, `Panel`, `Compound`, `Dose`), applies fixed background offsets, and computes:
- `c0_corrected`, `c1_corrected`, `Ratio` (c1/c0)
- `c1_masked_corrected`: mean mScarlet intensity inside organoids minus a fixed offset (`null_count`)

Writes `PerWell.csv` and `Aggregated.csv` (mean and std per Infection | Compound | Dose and timepoint), plus HTML plots:
- plate overviews: `nucSpot_Intensity.html`, `MaskedOrganoid_Quantity.html`, `mScarlet_Intensity.html`
- per panel (`Toxicity`, `Antiviral`, `Immune modulator`, each shown alongside `Control` and `LO (no Mac)`): `<Panel>.html`, `<Panel>_error.html`, `<Panel>_organoid.html`, `<Panel>_organoid_error.html`

In a live Jupyter session, clicking a point on a `FigureWidget` plot opens every replicate well for that condition, with its labels, in napari.



