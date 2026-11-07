# Reproducing the paper's analysis

This folder holds what is needed to rerun the paper's evaluation of ASTRID on its seven datasets, on top of the
`astrid` package. **Status: draft.** The parts marked *to come* are being ported from the analysis code.

## What is here

| File | Content |
|---|---|
| `datasets.yaml` | ASTRID's inputs and settings on each dataset as they were run: frame rate, pixel size, time constants, border, territory, evidence map, candidates, the cross-validation grids of D2–D5, D7's baseline |
| `astrid_repro/metrics.py` | the measures and statistics: the injection test's error (below 0.5 Hz), the leakage, the gain, the failure rule, the paired Wilcoxon signed-rank test and Holm's correction |
| `../aqua2_settings/` | the AQuA2 settings that made the astrocyte territory of D1, D6 and D7 |

ASTRID itself is the `astrid` package. On a real case of every dataset type, its traces were checked against the
analysis code that produced the paper's results: they are bitwise equal (the public datasets' route through
`demix`; D1 with its evidence map, candidates and territory; D6's configuration; D7 with its baseline).

## The route, per dataset

Every method received the same movies and ROI masks. On D1, D6 and D7, ROIs within 5 pixels of the image edge, a
margin left by registration, were not used as cases, and the pixels of this margin were left out of the input of
every method.

1. **Inputs.**
   - D2–D5 are public. D2 is the Allen Brain Observatory videos (AllenSDK) with TUnCaT's manual masks and
     transient labels. D4 and D5 are TUnCaT's NAOMi and one-photon videos with their labels. TUnCaT's data are on
     figshare, linked from its repository, https://github.com/YijunBao/TUnCaT. D3 is the CRCNS cai-1 dataset
     (https://crcns.org, a free account).
   - D1, D6 and D7 were recorded by the authors: *their availability is to be announced.*
2. **ASTRID** on each case with `astrid.demix` and the settings of `datasets.yaml`.
   - The baseline F0 of each pixel is, by default, the 10th percentile of its 1-s means in a moving 60-s window
     (D7's own rule is in `datasets.yaml`).
   - The fits were allowed at most 600 iterations, the default, except in the leakage tests of D1 and D7, where
     they were allowed at most 2,000.
   - On D2–D5, every (kappa_d, kappa_np) pair of the dataset's grid is run on every video. Leave-one-video-out
     cross-validation (leave-one-cell-out on D3) then picks the setting that is applied to each held-out
     video.
3. **Peer methods**, with their authors' code: FISSA 1.0.0, TUnCaT (its paper's v1.0 scripts), CNMF in CaImAn
   1.13.2 (seeded with the masks), Suite2p 1.1.0 (its neuropil subtraction on the given masks), and the ring
   subtraction (`ring__<rec>` in ASTRID's output). TUnCaT and CNMF also ran with the astrocyte territory as
   extra components on D1, D6 and D7 ("+ terr."). *Runners to come.*
4. **Tests and scores** (`astrid_repro/metrics.py`).
   - **D1 and D7:** the injection and leakage tests, with error and leakage.
   - **D6:** the leakage of the real astrocyte channel, from the sum of the two channels against the neuronal
     channel alone, below 5 Hz.
   - **D2, D4 and D5:** the F1 score of transient detection with TUnCaT's evaluation.
   - **D3:** the correlation with the calcium predicted from the recorded spikes.
   - *The case builders (adding known signals to the movie before the fit) and the scorers of D2–D5 are to
     come.*
5. **Statistics:** the mean ± SD over the units (imaging locations on D1 and D6, recordings on D7, videos on D2,
   D4 and D5, cells on D3). ASTRID is compared with each other method by a paired two-sided Wilcoxon
   signed-rank test, with Holm's correction over the comparisons of each measure (`astrid_repro/metrics.py`).
   *The scripts that write the paper's tables and figures are to come.*

## Running ASTRID as in the paper

```python
import yaml, astrid
cfg = yaml.safe_load(open("reproduce/datasets.yaml"))["datasets"]["D7_location1"]
settings = astrid.Settings(tau_r=cfg["tau_r"], tau_d=cfg["tau_d"], **cfg["settings"])
res = astrid.demix(movie, rois, fps=cfg["fps"], um_per_px=cfg["um_per_px"], territory=territory,
                   border_px=cfg["border_px"], settings=settings)
```
