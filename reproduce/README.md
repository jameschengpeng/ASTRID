# Reproducing the paper's analysis

This folder holds what is needed to rerun the paper's evaluation of ASTRID on its seven datasets, on top of the
`astrid` package. **Status: draft.** The parts marked *to come* wait on the decisions noted there.

Install the extra `reproduce` (pandas) with the package (`pip install -e ".[reproduce]"`) and run the modules
from this folder (`python -m astrid_repro.<module> --help`). The peer methods (FISSA, TUnCaT, CaImAn, Suite2p)
are GPL and are not part of this repository: install them in an environment of their own.

## What is here

| File | Content |
|---|---|
| `datasets.yaml` | ASTRID's inputs and settings on each dataset as they were run: frame rate, pixel size, time constants, border, territory, evidence map, candidates, the cross-validation grids of D2–D5, D7's baseline, the iteration limits |
| `astrid_repro/metrics.py` | the measures: the injection test's error (below 0.5 Hz), the leakage, the gain, the failure rule; the paired Wilcoxon signed-rank test and Holm's correction |
| `astrid_repro/peers.py` | FISSA, TUnCaT and TUnCaT + terr., CNMF and CNMF + terr. (CaImAn) on the exported cases of D1, D6 and D7, at the paper's settings |
| `astrid_repro/suite2p_runner.py` | Suite2p's neuropil subtraction on given masks: whole videos (D2, D4, D5), cai-1 (D3) and exported cases |
| `astrid_repro/stats.py` | the tables of D1 (ASTRID against seven methods), the ablations, D6 (its per-recording scoring and table) and D7 (by recording and by location) |
| `astrid_repro/public_data.py`, `ring.py`, `cv.py`, `tables_public.py` | D2–D5: the readers of the public data and of their ground truth, the ring subtraction, the leave-one-out cross-validation over settings and detection thresholds, the tables |
| `../aqua2_settings/` | the AQuA2 settings that made the astrocyte territory of D1, D6 and D7 |

**Checked against the analysis code that produced the paper's results.**
- ASTRID's traces are bitwise equal on a real case of every dataset type: the public datasets' route through
  `demix`, D1 with its evidence map, candidates and territory, D6's configuration, and D7 with its baseline.
- From the saved per-case scores and traces, every number of the paper's tables is recomputed exactly: D1, the
  ablations, D2–D5 with the cross-validation, D6 and D7.
- The ROI detection gives the ROIs of the paper's D1 run.

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
     cross-validation (leave-one-cell-out on D3) then picks the setting that is applied to each held-out video
     (`cv.py`).
3. **Peer methods**, with their authors' code (`peers.py`, `suite2p_runner.py`):
   - FISSA 1.0.0;
   - TUnCaT, its paper's v1.0 scripts;
   - CNMF in CaImAn 1.13.2, seeded with the masks;
   - Suite2p 1.1.0, its neuropil subtraction on the given masks;
   - the ring subtraction, `ring__<rec>` in ASTRID's output, or `ring.py` on the public data.

   TUnCaT and CNMF also ran with the astrocyte territory as extra components on D1, D6 and D7 ("+ terr.").
4. **Tests and scores.**
   - **D1 and D7:** the injection and leakage tests, scored by error and leakage (`metrics.py`). *The case
     builders, which add the known signals to the movies before the fits, wait on the release of D1 and D7.*
   - **D6:** the leakage of the real astrocyte channel, from the sum of the two channels against the neuronal
     channel alone, below 5 Hz (`stats.py d6-scores`).
   - **D2, D4 and D5:** the F1 score of transient detection with TUnCaT's evaluation.
     *That scorer is a port of TUnCaT's GPL-2 MATLAB code and is not included yet.* `cv.f1_grid` takes it as an
     argument.
   - **D3:** the correlation with the calcium predicted from the recorded spikes (`public_data.py`).
5. **Statistics and tables.** The mean ± SD is taken over the units: the imaging locations on D1 and D6, the
   recordings on D7, the videos on D2, D4 and D5, the cells on D3. ASTRID is compared with each other method by a
   paired two-sided Wilcoxon signed-rank test, with Holm's correction over the comparisons of each measure
   (`stats.py`, `tables_public.py`). *The figure scripts are to come.*

## Running ASTRID as in the paper

```python
import yaml, astrid
cfg = yaml.safe_load(open("reproduce/datasets.yaml"))["datasets"]["D7_location1"]
settings = astrid.Settings(tau_r=cfg["tau_r"], tau_d=cfg["tau_d"], **cfg["settings"])
res = astrid.demix(movie, rois, fps=cfg["fps"], um_per_px=cfg["um_per_px"], territory=territory,
                   border_px=cfg["border_px"], settings=settings)
```
