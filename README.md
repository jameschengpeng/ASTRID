# ASTRID

**ASTRocyte-Informed Demixing** of neuronal calcium traces.

In calcium imaging where neurons and astrocytes share one fluorescence channel, an ROI drawn on a neuron
also collects light from nearby astrocytes, neighbouring neurons and the neuropil. ASTRID fits, for each ROI,
a small model of the patch of pixels around it, with the neuron's own signal, the neighbouring neurons,
astrocyte components placed on astrocyte territory, and the local neuropil, each with its own spatial
support and temporal assumptions. It returns the ROI's demixed trace (ΔF and ΔF/F0) with the other sources
removed. Without astrocyte territory it runs as a neuron-plus-neuropil demixing.

The method and its evaluation are described in the paper (citation to come). This repository holds the
package and the code to reproduce the paper's analysis (`reproduce/`).

## Install

```bash
pip install astrid-demix            # once released; until then, from a clone:
pip install -e .                    # numpy, scipy, h5py, pyyaml, numba; add [tiff] to read TIFF movies
```

Python 3.10 or later. AQuA2 (MATLAB) is needed only to make astrocyte territory from a movie; it is not part of
this package (see [Astrocyte territory](#astrocyte-territory)).

## Quick start

```python
import numpy as np
import astrid

# movie: (T, H, W) array (or an HDF5 dataset, a .npy/.tif path); rois: (n, H, W) boolean masks
res = astrid.demix(movie, rois, fps=30.0, um_per_px=0.8, tau_r=0.045, tau_d=0.142)
dff = res.dff                      # (n, T) demixed dF/F0, NaN rows for ROIs not fitted
res.save("astrid_out")             # astrid_out/result.npz and result.json (the provenance)
```

The same from the command line:

```bash
astrid demix --movie movie.h5:data --rois rois.npz --fps 30 --um-per-px 0.8 \
             --tau-r 0.045 --tau-d 0.142 --out astrid_out
```

## The endpoints

All five return an `astrid.Result` and take the keywords of `demix` (time constants, `settings`, `border_px`,
`workers`, `export_components`, ...). Each has a command (`astrid <command> --help`).

| Endpoint | What it takes | Command |
|---|---|---|
| `demix(movies, rois, *, fps, um_per_px, ...)` | a movie (or several recordings of one field) and ROI masks; optionally astrocyte territory, an evidence map, candidate cells | `astrid demix` |
| `from_suite2p(folder, *, um_per_px, ...)` | a Suite2p plane folder: its registered movie, cells and frame rate | `astrid suite2p` |
| `from_caiman(path, ...)` | a CaImAn result file (`cnm.save`): footprints made into masks, its movie, frame rate and pixel size | `astrid caiman` |
| `territory_from_aqua2(movies, rois, aqua2_results, *, fps, um_per_px, ...)` | a movie and ROI masks, with the territory from AQuA2's result files | `astrid aqua2` |
| `run_pipeline(movies, rois=None, *, fps, um_per_px, aqua2_results=None, marker=None, ...)` | one imaging location as in the paper's Fig. 1A: its recordings, the ROI masks (or the paper's ROI detection when none are given), AQuA2's results and a structural astrocyte marker | `astrid pipeline location.yaml` |

```python
# 1. a movie and ROI masks, with astrocyte territory from any source
res = astrid.demix(movie, rois, fps=30.9, um_per_px=2.0, tau_r=0.05, tau_d=0.4, territory=astro_masks)

# 2. a Suite2p run (its cells, data.bin and ops['fs'])
res = astrid.from_suite2p("suite2p/plane0", um_per_px=0.8, tau_r=0.045, tau_d=0.142)

# 3. a CaImAn run (estimates.A thresholded as CaImAn draws contours; its memmap, fr and dxy)
res = astrid.from_caiman("cnm_results.hdf5", tau_r=0.045, tau_d=0.142)

# 4. territory from AQuA2: pixels touched by >= 2 events over the result files given
res = astrid.territory_from_aqua2(movie, rois, ["rec1_AQuA2.mat", "rec2_AQuA2.mat"],
                                  fps=30.9, um_per_px=2.0, tau_r=0.05, tau_d=0.4)

# 5. one imaging location: two recordings fitted jointly, territory from AQuA2, evidence from a marker
res = astrid.run_pipeline(["rec1.h5:mov", "rec2.h5:mov"], rois, fps=30.9, um_per_px=2.0,
                          tau_r=0.05, tau_d=0.4, aqua2_results=["rec1_AQuA2.mat", "rec2_AQuA2.mat"],
                          marker=[red_mean_1, red_mean_2], out="astrid_out")
```

`astrid pipeline` reads the same arguments from a YAML file (paths relative to it):

```yaml
movies: [rec1.h5:mov, rec2.h5:mov]
rois: rois.npz                               # optional: without it, the ROIs are detected
fps: 30.9
um_per_px: 2.0
tau_r: 0.05
tau_d: 0.4
aqua2_results: [rec1_AQuA2.mat, rec2_AQuA2.mat]
marker: [rec1_red.h5:mov, rec2_red.h5:mov]   # optional
out: astrid_out
```

## Inputs

- **Movies.** (T, H, W), one recording or a list of recordings of the same field registered to one pixel
  grid (fitted jointly, with footprints shared across recordings). In memory, or anything that slices along
  time (`h5py.Dataset`, `numpy.memmap`), or a path: `file.h5:dataset`, `file.npy`, `file.tif` (needs
  `tifffile`). Raw fluorescence, not ΔF/F: ASTRID computes each pixel's baseline itself. A MATLAB v7.3 file
  read through h5py comes out with MATLAB's axes reversed, (T, W, H) for an (H, W, T) movie: transpose its
  frames (or the masks) so that movie and masks share one (H, W) frame.
- **ROI masks.** (n, H, W) booleans. Overlapping masks are refused unless `overlaps="nearest"`, which gives
  each shared pixel to the mask with the nearest centroid. The demixed trace averages the ROI's pixels.
- **`fps` and `um_per_px`** are required. Lengths (patch margin, supports, ring) are set in micrometres and
  converted with the pixel size; time scales (baseline, noise band, kernel, astrocyte prior) in seconds.
- **`tau_r`, `tau_d`**: the rise and decay time constants (s) of the indicator's impulse response. They have
  no default, because they depend on the indicator. The paper used 0.045 and 0.142 s for GCaMP6f, 0.179
  and 0.550 s for GCaMP6s, and 0.05 and 0.4 s on D1 (GCaMP6m) and D6 (jRGECO1a). On D7 they were estimated
  from the recordings (`reproduce/datasets.yaml`).
- **`territory`** (optional): astrocyte territory as an (H, W) map or (m, H, W) masks, from any source. In
  each patch, the up to three largest regions of territory (at least 80 µm²) become astrocyte components.
  Without territory there are no astrocyte components.
- **`evidence`** (optional): a structural evidence map ρ in [0, 1], (H, W), from an astrocyte marker
  imaged in another channel (`astrid.evidence.evidence_map(mean_image)`). Astrocyte footprints on pixels with
  weak evidence are penalized. Without a map the penalty is off.
- **`candidates`** (optional): an (H, W) map of candidate cells that are not ROIs (a detection's rejected
  proposals); its pieces near an ROI are fitted as unclassified sources and subtracted.
- **`border_px`** (default 0): a margin at the image edge, such as one left by registration, whose pixels are
  left out of every patch; ROIs within it are not fitted. The paper's D1, D6 and D7 had a 5-pixel margin.

## Outputs

`Result.arrays()` (and `result.npz`) hold, per ROI (row i is mask i):

| Key | Shape | Content |
|---|---|---|
| `dff__<rec>` | (n, T) | the demixed trace as ΔF/F0 (ΔF over the ROI average of the baseline F0) |
| `delta_f__<rec>` | (n, T) | the demixed trace in fluorescence units: the ROI average of the baseline-subtracted movie minus the fitted contributions of every other component (the residual is kept) |
| `baseline__<rec>` | (n, T) | the ROI average of F0 |
| `ring__<rec>` | (n, T) | the ring subtraction for comparison: the ROI mean − 0.7 × the mean of a ring 4–16 µm outside the ROI that leaves out the pixels of every ROI and candidate, both of the baseline-subtracted movie (fluorescence units) |
| `fitted`, `converged`, `iterations`, `stop`, `error` | (n,) | the fit's outcome; a fit that fails is recorded and the others go on |
| `n_neighbours`, `n_unclassified`, `n_astro`, `pixels`, `seconds` | (n,) | the patch's components and size, and the time |

`<rec>` is the recording's name (`recording_names=`, default `rec0`, `rec1`, ...). With
`export_components=True`, also:

- `contrib_<kind>__<rec>` (n, T), for `neighbours`, `unclassified`, `astro`, `neuropil`: the ROI-average
  contribution of each other kind of component, as ΔF/F0 on the ROI's baseline (0 where the patch has none).
  `dff` plus these is the ROI average of the movie's ΔF/F0.
- one row per component of every fitted patch (the ROI's own first): `component_roi`, `component_name`,
  `component_kind`, `component_dff__<rec>` (the component's trace as ΔF/F0 averaged over its spatial
  support, as in the paper's Fig. 1E), and its footprint as a sparse list (`component_fp_offset`,
  `component_fp_row`, `component_fp_col`, `component_fp_weight`; `astrid.component_footprint` rebuilds it).

The astrocyte components are nuisance terms that keep astrocyte signal out of the neuronal traces; their
traces are not validated as astrocyte readouts.

## Settings

`astrid.Settings` holds the model's settings in the paper's names; the defaults are those of the paper's
Supplementary Table of default settings.

| Setting | Default | Meaning | Name in the analysis code |
|---|---|---|---|
| `kappa_d` | 2 | drive: the height, in noise s.d., a transient must reach before any drive is assigned to it | `kappa_s` |
| `kappa_a` | 10 | astrocytes: at the reference starting point, a cosine coefficient above 2 f_a shrinks by 1/(1 + κ_a) | `kappa_a` |
| `kappa_np` | 0.1 | neuropil: smoothness of its footprint | `kappa_b` |
| `kappa_rho` | 1 | evidence: astrocyte footprint on weak structural evidence | `kappa_t` |
| `f_a` | 0.2 Hz | cutoff of the astrocytes' low-frequency prior | `f_a` |
| `neuropil_cap` | True | the neuropil footprint may not be higher on the ROI than around it | `neuropil_cap` |
| `max_iterations`, `footprint_steps`, `drive_steps` | 600, 200, 300 | iteration limits | `n_outer`, `n_h`, `n_deconv` |
| `check_every`, `tolerance` | 10, 1e-3 | the convergence check | `block`, `tol_target` |
| `baseline_bin_s`, `baseline_window_s`, `baseline_percentile` | 1 s, 60 s, 10 | the baseline F0 of each pixel: the 10th percentile of its 1-s means in a moving 60-s window | |
| `trend_pct_per_s` | 0 | with `baseline_window_s=None`: one percentile along a line of this slope (the paper's short one-photon recordings: 3-s bins) | |

```python
s = astrid.Settings(kappa_d=4, kappa_np=0.3)       # or Settings.from_yaml("settings.yaml")
res = astrid.demix(movie, rois, fps=30, um_per_px=1.0, tau_r=0.045, tau_d=0.142, settings=s)
```

The fixed geometry (patch margin 16 µm, neuronal support margin 4 µm, astrocyte support margin 2 µm, up to
three regions of territory of at least 80 µm², the 5-Hz noise band) is in `astrid/patch.py`.

## Astrocyte territory

ASTRID does not detect astrocytes. The paper's territory came from
[AQuA2](https://github.com/yu-lab-vt/AQuA2) (Mi et al. 2025; MATLAB, GPL-2; tested at commit 6e9a965): the
pixels touched by at least two AQuA2 events over the recordings of an imaging location. Run AQuA2 on your
movie (`aqua2_settings/` holds the settings the paper used) and pass its result files to
`territory_from_aqua2` or `run_pipeline`, or make the map yourself with `astrid.aqua2.territory(files)`.
Any other source of astrocyte masks works too.

## ROI detection

ASTRID accepts ROI masks from any source. The package also holds the ROI detection of the paper's pipeline
(Supplementary, "ROI detection"), which the paper does not evaluate:

```python
det = astrid.detect_rois(["rec1.h5:mov", "rec2.h5:mov"], fps=30.9,
                         aqua2_results=["rec1_AQuA2.mat", "rec2_AQuA2.mat"])   # one AQuA2 result per movie
res = astrid.demix(["rec1.h5:mov", "rec2.h5:mov"], det.rois, fps=30.9, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                   candidates=det.candidates, territory=territory)
```

The steps (`astrid/detection.py`):
- **Proposals.** SynQuant (Wang et al. 2020) proposes candidate ROIs on each recording's maximum and mean
  projections and on a map of its fast activity (0.2–5 Hz). Pixels at the image edge and in the recording's
  AQuA2 astrocyte mask are removed.
- **Acceptance.** A candidate is accepted if its average trace, smoothed with a Gaussian kernel of SD 6
  frames, rises above a moving 6-s baseline by at least 3 times its noise level and to a ΔF/F0 of at least
  50%, and its longest period above a ΔF/F0 of 10% lasts at most 5 s.
- **Merging.** The ROIs of a location's recordings are merged into one set.

`astrid detect` does the same from the command line.

SynQuant's scoring is in `astrid/synquant.py`. Its order-statistics test is adapted from the SynQuant (Wang et al.
2020) and AQuA2 (Mi et al. 2025) MATLAB code of Guoqiang Yu's lab at Virginia Tech (now at Tsinghua University),
with modifications by James Cheng Peng, used under the MIT license with the permission of Guoqiang Yu's lab.

- Wang Y, Wang C, Ranefall P, et al. SynQuant: an automatic tool to quantify synapses from microscopy images.
  *Bioinformatics* 36(5):1599–1606 (2020).
- Mi X, Chen AB-Y, Duarte D, et al. Fast, accurate, and versatile data analysis platform for the quantification
  of molecular spatiotemporal signals. *Cell* 188(10):2794–2809 (2025).

## Spike inference

ASTRID does not infer spikes. Its demixed ΔF/F0 can go to any spike inference method. In the paper's pipeline it
went to [CASCADE](https://github.com/HelmchenLabSoftware/Cascade) (Rupprecht et al. 2021, *Nature Neuroscience*
24:1324–1337), with its general model for excitatory neurons imaged at 30 Hz (`Global_EXC_30Hz_smoothing200ms`).
CASCADE is not part of this package.

## Performance

Each ROI is fitted separately; `workers=N` fits N ROIs at a time in separate processes (set one BLAS thread per
process, e.g. `OMP_NUM_THREADS=1`), the largest patches first.

- **The drive update runs compiled** (numba). The first run compiles it, which takes several seconds, and numba
  caches it for later runs.
- **Memory:** where processes fork (Linux, macOS), `workers` > 1 also high-passes each pixel once for the whole
  field instead of once per patch, which doubles the memory of the pixel data.

The results are the same bitwise either way.

## Reproducing the paper

See [`reproduce/README.md`](reproduce/README.md): the paper's evaluation on its seven datasets, its tables, and
the code that draws the data panels of its figures.

- **From public data alone:** D2–D5 (the Allen Brain Observatory, CRCNS cai-1, NAOMi and one-photon datasets),
  from the public movies to every method's traces, the tables and the figures (Fig. 4 and the supplementary
  figure of D2, D4 and D5). For D3, the export every method read is built from the CRCNS download, with the
  ROI of each cell from `reproduce/cai1_rois`, drawn by hand once, blind to all methods (the dataset ships
  none).
- **With the authors' recordings:** D1, D6 and D7, whose availability is to be announced. Their settings, the
  statistics of their tables and the ROI detection are here; the builders of their injection and leakage tests
  and their figures will follow with the data. D6's table in the paper scores every peer trace after its own
  running baseline is removed: `python -m astrid_repro.stats d6 --baseline equal`, after `d6-scores --baseline
  equal` on each recording.

## License

MIT (see `LICENSE`), except two folders of the reproduction code, which derive from TUnCaT's GPL-2 code and so
keep its license (GPL-2, each with its own `LICENSE`): `reproduce/tuncat_eval/`, the paper's F1 scorer of D2, D4
and D5 (and its per-ROI view, drawn in their supplementary figure), a line-by-line port of TUnCaT's MATLAB
evaluation; and `reproduce/tuncat_peers/`, which reruns the peer methods on D2, D4 and D5 with the scripts of
TUnCaT's paper-reproduction archive, changed as the paper's runs changed them. The `astrid` package and the MIT
reproduction code never import them. AQuA2, CASCADE, the peer methods
used in the paper's benchmark, and their data are not included and keep their own licenses.
