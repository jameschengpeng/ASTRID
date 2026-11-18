# Reproducing the paper's analysis

This folder holds what is needed to rerun the paper's evaluation of ASTRID on its seven datasets, on top of the
`astrid` package. The parts in italics wait on the decisions noted there.

Install the extra `reproduce` (pandas, matplotlib) with the package (`pip install -e ".[reproduce]"`) and run
the modules from this folder (`python -m astrid_repro.<module> --help`). The peer methods (FISSA, TUnCaT, CaImAn, Suite2p)
are GPL and are not part of this repository: install them in an environment of their own.

**Licenses.** Everything here is MIT, like the `astrid` package, except two folders that are GPL-2, each with
its own `LICENSE`, because they derive from the GPL-2 code of TUnCaT (Bao et al. 2022,
https://github.com/YijunBao/TUnCaT):
- `tuncat_eval/`, the F1 scorer of D2, D4 and D5, a line-by-line port of TUnCaT's MATLAB evaluation code. The
  `astrid` package and `astrid_repro` never import it; it is handed to them at run time (below), as the scorer of
  the tables and, scored ROI by ROI, to the example panels of the supplementary figure of D2, D4 and D5;
- `tuncat_peers/`, which stages the scripts of TUnCaT's paper-reproduction archive with the changes the paper's
  runs made to them and runs FISSA, TUnCaT and CNMF on D2, D4 and D5 (`python -m tuncat_peers`).
A port or a change of GPL-2 code keeps its license, so these folders are kept apart from the MIT code.

## What is here

| File | Content |
|---|---|
| `datasets.yaml` | ASTRID's inputs and settings on each dataset as they were run: frame rate, pixel size, time constants, border, territory, evidence map, candidates, the cross-validation grids of D2–D5, D7's baseline, the iteration limits |
| `astrid_repro/metrics.py` | the measures: the injection test's error (below 0.5 Hz), the leakage, the gain, the failure rule; the paired Wilcoxon signed-rank test and Holm's correction |
| `astrid_repro/run.py` | ASTRID on D2–D5 as the paper ran it: every video at every setting of the dataset's grid (`python -m astrid_repro.run D2 --data <archive> --out <results>`) |
| `astrid_repro/abo_crop.py` | D2's movies: the URLs of the ten public Allen movies and TUnCaT's crop of them (the first 20 % of frames, the central 200 × 200 pixels) |
| `astrid_repro/peers.py` | FISSA, TUnCaT and TUnCaT + terr., CNMF and CNMF + terr. (CaImAn) on the exported cases of D1, D6 and D7, at the paper's settings |
| `astrid_repro/cai1_peers.py` | FISSA, TUnCaT and CNMF (CaImAn) on the recordings of D3's export, at the paper's settings (CNMF also at each setting of its cross-validation grid) |
| `astrid_repro/suite2p_runner.py` | Suite2p's neuropil subtraction on given masks: whole videos (D2, D4, D5), cai-1 (D3) and exported cases |
| `astrid_repro/cai1_export.py`, `cai1_rois/` | D3: the export every method read, built from the CRCNS download (the frames placed on the processed traces' grid, rigid registration, 4 × 4 binning, the scored frames, the spike-predicted calcium), with the ROI of each cell, drawn by hand once, blind to all methods |
| `astrid_repro/stats.py` | the tables of D1 (ASTRID against seven methods), the ablations, D6 (its per-recording scoring and table) and D7 (by recording and by location) |
| `astrid_repro/public_data.py`, `ring.py`, `cv.py`, `tables_public.py` | D2–D5: the readers of the public data and of their ground truth, the ring subtraction, the leave-one-out cross-validation over settings and detection thresholds, the tables |
| `astrid_repro/figures/` | the data panels of the paper's figures, drawn from the outputs above: Figure 4 (`cai1.py`) and Supplementary Figure S6 (`public.py`, D2, D4 and D5), with the rules that chose their examples (section "Figures" below) |
| `tuncat_peers/` (**GPL-2**) | FISSA, TUnCaT and CNMF on D2, D4 and D5 with the scripts of TUnCaT's paper-reproduction archive, staged with the paper's changes (CNMF at its published setting, one background component and AR(1), or at any setting of its cross-validation grid), and the record of those changes |
| `tuncat_eval/` (**GPL-2**) | the F1 scorer of D2, D4 and D5: TUnCaT's transient detection and matching, the ground-truth transients of the NAOMi videos, the template filtering, the rule for CNMF results with removed components, and the same scoring ROI by ROI for the figures (`evaluation.view`) |
| `../aqua2_settings/` | the AQuA2 settings that made the astrocyte territory of D1, D6 and D7 |

**Checked against the analysis code that produced the paper's results.**
- ASTRID's traces are bitwise equal on a real case of every dataset type: the public datasets' route through
  `demix`, D1 with its evidence map, candidates and territory, D6's configuration, and D7 with its baseline.
- From the saved per-case scores and traces, every number of the paper's tables is recomputed exactly: D1, the
  ablations, D2–D5 with the cross-validation, D6 and D7.
- The ROI detection gives the ROIs of the paper's D1 run.
- The figure code draws Figure 4 and Supplementary Figure S6 with the values of the paper's figures: every point,
  line, image, outline, band and text the paper's panels draw, compared one by one, and the examples, windows
  and scores that their legends print.

## The route, per dataset

All methods received the same ROI masks and the same movie of each recording, with two differences in how they
received them. Where the released masks of D2, D4 and D5 overlapped, ASTRID gave each shared pixel to the ROI with
the nearest centroid, whereas the other methods used the masks as released. On D1, D6 and D7, FISSA, TUnCaT, CNMF
and Suite2p were run on each ROI separately, on a crop of the movie that extended about 30 µm beyond the ROI,
farther than the 16 µm margin of ASTRID's patch, and TUnCaT, CNMF and Suite2p also received the masks of the
neighboring ROIs and, on D1 and D6, of the candidates within ASTRID's patch. On D1, D6 and D7, ROIs within 5
pixels of the image edge, a margin left by registration, were not used as cases, and the pixels of this margin
were left out of the input of every method.

1. **Inputs.**
   - D2–D5 are public. D2 is ten Allen Brain Observatory experiments, cropped as TUnCaT cropped them
     (`abo_crop.py`, from the public movies, about 60 GB each), with TUnCaT's manual masks and transient labels.
     D4 and D5 are TUnCaT's NAOMi and one-photon videos with their labels. TUnCaT's data are on
     figshare, linked from its repository, https://github.com/YijunBao/TUnCaT. D3 is the CRCNS cai-1 dataset
     (https://crcns.org, a free account). Every method read one export of D3: per recording, the registered
     movie binned 4 × 4, the ROI drawn by hand on the cell, the scored frames and the spike-predicted calcium
     (`public_data.py`). `cai1_export.py` builds it from the CRCNS download, with the masks of `cai1_rois/`
     (the dataset ships none).
   - D1, D6 and D7 were recorded by the authors and are available from the corresponding author, Axel
     Nimmerjahn, on reasonable request.
2. **ASTRID** on each case with `astrid.demix` and the settings of `datasets.yaml`.
   - The baseline F0 of each pixel is, by default, the 10th percentile of its 1-s means in a moving 60-s window
     (D7's own rule is in `datasets.yaml`).
   - The fits were allowed at most 600 iterations, the default, except in the leakage tests of D1 and D7, where
     they were allowed at most 2,000.
   - On D2–D5, every (kappa_d, kappa_np) pair of the dataset's grid is run on every video (`run.py`, into the
     layout `tables_public` reads). Leave-one-video-out cross-validation (leave-one-cell-out on D3) then picks
     the setting that is applied to each held-out video (`cv.py`).
3. **Peer methods**, with their authors' code (`peers.py` on D1, D6 and D7, `cai1_peers.py` on D3,
   `tuncat_peers` on D2, D4 and D5, `suite2p_runner.py`). On D2, D4 and D5, the paper reran the scripts of
   TUnCaT's paper-reproduction archive (`tuncat_peers`); the traces those scripts' authors released, in
   TUnCaT's archive, match the reruns closely but, for CNMF, not exactly, so they are a close second route. A
   rerun through `tuncat_peers` in the paper's environments gives FISSA's and CNMF's traces of the paper bit for
   bit; TUnCaT's unmixing is not deterministic from run to run, so its traces come within about 2e-4 of the
   paper's and its F1 within about 0.006 at single thresholds (`tuncat_peers/__init__.py`). CNMF with more than
   one background component is not deterministic either: CaImAn starts the background from scikit-learn's NMF
   with `init='nndsvda'`, whose randomized SVD is not seeded (with one component CaImAn starts from the mean, so
   the published setting reproduces bit for bit). Two runs at the settings chosen on D4 and D5 gave the same mean
   F1 within 0.001 (0.8423 and 0.8425 on D4, 0.6825 and 0.6819 on D5), although single traces differ more.
   The methods:
   - FISSA 1.0.0, at its defaults on D1, D6 and D7; on D2–D5 its alpha is chosen by the same cross-validation as
     the other methods' settings, over its authors' grid for the dataset. On D4 the paper also ran alpha 0.0005,
     below that grid, in a run of its own whose output folder `tables_public grids` reads as a second `--peers` root:

     ```bash
     python -m tuncat_peers --scripts <archive scripts> --data <archive> --dataset NAOMi --method fissa \
         --fissa-alphas 0.0005 --out <peers2>
     ```
   - TUnCaT, its paper's v1.0 scripts, with floating alpha: alpha 1 on D1, D6 and D7; on D2–D5 chosen by the
     cross-validation over its authors' grid for the dataset;
   - CNMF in CaImAn 1.13.2, seeded with the masks. On D2–D5 its settings are chosen by the same cross-validation
     as the other methods', over the number of background components (1, 2, 3) and the order of the
     autoregressive model (1, 2), ties going to the first in the order (1, 1), (1, 2), (2, 1), (2, 2), (3, 1),
     (3, 2) as (gnb, p). The paper's choices: (1, 1) on D2 (10 of 10 videos), (1, 2) on D3 (20 of 20 cells),
     (3, 2) on D4 (10 of 10) and (2, 2) on D5 (9 of 9). Each setting runs into its own folder and the tables
     read them all (`--cnmf-grid`); the published setting, (1, 1), stays in the tables as the reference row
     `CNMF_fixed`:

     ```bash
     python -m tuncat_peers --scripts <archive scripts> --data <archive> --dataset NAOMi --method cnmf \
         --cnmf-gnb 3 --cnmf-p 2 --out <grid>/gnb3_p2 [--smoke <video>]          # D2, D4, D5
     python -m astrid_repro.cai1_peers --data <D3 export> --out <grid>/gnb1_p2/cai1 --methods cnmf \
         --cnmf-gnb 1 --cnmf-p 2 <recording> ...                                 # D3
     ```

     CaImAn writes each memory-mapped movie to `$CAIMAN_TEMP` (else to `~/caiman_data/temp`), so both runners
     point it to a temporary folder that is deleted when the fit ends (for `tuncat_peers`, when the run ends:
     one video per run with `--smoke`);
   - Suite2p 1.1.0, its neuropil subtraction on the given masks (`suite2p_runner.py`, in Suite2p's environment).
     On D2, D4 and D5 it runs on each video with the released masks, into the folder that
     `public_data.suite2p_path` reads, `<root>/<dataset>/<video>/suite2p/<video>.npz` (`<dataset>` ABO, NAOMi or
     1p), and `tables_public` reads `--suite2p <root>`:

     ```bash
     python -I astrid_repro/suite2p_runner.py video --movie <archive>/data/NAOMi/<video>.h5:mov \
         --masks "<archive>/data/NAOMi/GT Masks/FinalMasks_<video>.mat" --name <video> --threads 8 \
         --out <root>/NAOMi/<video>/suite2p
     ```

     On D3 `suite2p_runner.py cai1` runs on the export's recordings, and on D1, D6 and D7 `suite2p_runner.py cases`
     on each recording's exported cases;
   - the ring subtraction, `ring__<rec>` in ASTRID's output, or `ring.py` on the public data.

   The alpha grids of FISSA and TUnCaT on D2–D5 are those of the scripts of TUnCaT's paper-reproduction archive
   for each dataset. D3 was not in TUnCaT's benchmark, so it took D2's (Allen) grids, with TUnCaT's extended by
   50 and 100 (`cai1_peers.py`):

   | Dataset | FISSA alpha | TUnCaT alpha (floating) |
   |---|---|---|
   | D2 (ABO) | 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1 (9 values, `FISSA_ABO.py`) | 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30 (11 values, `TUnCaT_multi_ABO.py`) |
   | D3 (cai-1) | D2's (9 values) | D2's and 50, 100 (13 values) |
   | D4 (NAOMi) | 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1 (13 values, `FISSA_NAOMi.py`), and 0.0005 | 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10 (13 values, `TUnCaT_multi_NAOMi.py`) |
   | D5 (1p) | 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1 (13 values, `FISSA_1p.py`) | 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10 (13 values, `TUnCaT_multi_1p.py`) |

   TUnCaT and CNMF also ran with the astrocyte territory as extra components on D1, D6 and D7 ("+ terr.").
4. **Tests and scores.**
   - **D1 and D7:** the injection and leakage tests, scored by error and leakage (`metrics.py`). *The case
     builders, which add the known signals to the movies before the fits, wait on the release of D1 and D7.*
   - **D6:** the leakage of the real astrocyte channel, from the sum of the two channels against the neuronal
     channel alone, below 5 Hz and below 0.5 Hz, and the kept size of the neurons' own transients
     (`stats.py`). ASTRID's and the ring subtraction's traces are F - F0 of each movie. The paper's table
     (Supplementary Table S7) scores every other trace after its own running baseline is removed by ASTRID's
     rule, on each movie (`--baseline equal`); without this step (`--baseline reported`, the default) the means
     differ by at most 0.015:

     ```bash
     python -m astrid_repro.stats d6-scores --run <root>/<location>_<recording> --baseline equal  # each recording
     python -m astrid_repro.stats d6 --root <root> --baseline equal --out <tables>                 # Table S7
     ```
   - **D2, D4 and D5:** the F1 score of transient detection with TUnCaT's evaluation (`tuncat_eval/`, GPL-2),
     handed to the MIT code at run time:

     ```bash
     python -m astrid_repro.tables_public grids --dataset ABO --archive <TUnCaT archive> --ours <root> ... \
         --peers <root> ... --cnmf-grid <grid> \
         --scorer tuncat_eval.evaluation:scorer --cnmf-fill tuncat_eval.evaluation:cnmf_fill --out <grids>
     ```
   - **D3:** the correlation with the calcium predicted from the recorded spikes (`public_data.py`), scored two
     ways, both reported. ASTRID's trace is ΔF/F0, its running baseline removed by the fit. As first scored,
     every other trace has only its mean removed. With the same baseline, every other trace first has its own
     running baseline removed by ASTRID's rule (the 10th percentile of 1-s means in a moving 60-s window):

     ```bash
     python -m astrid_repro.tables_public cai1 --export <D3 export> --ours <root> ... --peers <folder> \
         --cnmf-grid <grid> --suite2p <root> --out <tables>                     # cv_cai1
     python -m astrid_repro.tables_public cai1 ... --baseline equal --out <tables>  # cv_cai1_equal_baseline
     ```
5. **Statistics and tables.** The mean ± SD is taken over the units: the imaging locations on D1 and D6, the
   recordings on D7, the videos on D2, D4 and D5, the cells on D3. ASTRID is compared with each other method by a
   paired two-sided Wilcoxon signed-rank test, with Holm's correction over the comparisons of each measure
   (`stats.py`, `tables_public.py`). The test is `metrics.wilcoxon_p`, zero differences dropped, with SciPy's
   default method in every table but the ablations', which used its exact method.
6. **Figures**: below.

## Running ASTRID as in the paper

On D2–D5, the runner does every fit of the paper (on D3, after `cai1_export`):

```bash
python -m astrid_repro.run D3 --data <D3 export> --out <results> --workers 8      # 58 recordings x 19 settings
```

On one case, with a dataset's settings:

```python
import yaml, astrid
cfg = yaml.safe_load(open("reproduce/datasets.yaml"))["datasets"]["D7_location1"]
settings = astrid.Settings(tau_r=cfg["tau_r"], tau_d=cfg["tau_d"], **cfg["settings"])
res = astrid.demix(movie, rois, fps=cfg["fps"], um_per_px=cfg["um_per_px"], territory=territory,
                   border_px=cfg["border_px"], settings=settings)
```

## Figures

`astrid_repro/figures` draws the data panels of the paper's figures from the outputs above, each with the values,
examples, windows, marks and colors of the paper's figure, and saves each panel on its own
(`<panel>_<dataset>_<part>.png` and `.svg`) with a record of what it shows (`.json`: the values drawn, the
example and the numbers its legend prints). The schematics (Fig. 1A and 1D, 2A, 3A, Supp. Fig. S2A) and the
paper's typography, panel sizes and page layout are not part of it: the panels are drawn with matplotlib's
defaults.
Run from this folder:

```bash
# Figure 4 (D3): A, the correlation per cell; B, the example. The table sets the scoring: the paper's Figure 4
# is drawn from the equal-baseline table (cv_cai1_equal_baseline.json), and cv_cai1.json draws the first scoring
python -m astrid_repro.figures cai1 --table <tables>/cv_cai1_equal_baseline.json --export <D3 export> \
    --ours <root> [...] --peers <folder> --cnmf-grid <grid> --suite2p <root> --out <figures>
# Supplementary Figure S6, one row per dataset: D2 (ABO: A, B), D4 (NAOMi: C, D), D5 (1p: E, F)
python -m astrid_repro.figures public --dataset ABO --table <tables>/cv_ABO.json --archive <TUnCaT archive> \
    --ours <root> [...] --peers <root> [...] --cnmf-grid <grid> --suite2p <root> \
    --evaluator tuncat_eval.evaluation:view --cnmf-fill tuncat_eval.evaluation:cnmf_fill --out <figures>
```

The tables are those `tables_public` writes; the other paths are those given to it.

| Paper panel | Shows | Code |
|---|---|---|
| Fig. 4A | D3: each method's correlation with the spike-predicted calcium per cell, GCaMP6f and GCaMP6s cells marked, the Holm-corrected marks | `cai1.draw_scores` |
| Fig. 4B | D3: the example recording's mean image and ROI, and each method's standardized trace over the predicted calcium, with the spikes | `cai1.example`, `cai1.draw_image`, `cai1.draw_traces` |
| Supp. Fig. S6A, C, E | D2, D4, D5: each method's F1 per video, the Holm-corrected marks | `public.draw_scores` |
| Supp. Fig. S6B, D, F | D2, D4, D5: the example video's mean image and ROI masks, and the example ROI's trace as the evaluation sees it, with the labeled or simulated transients and each method's detections | `public.example`, `public.draw_image`, `public.draw_traces` |
| Figs. 1-3; Supp. Figs. S1-S5 | D1, D6, D7, and the simulations of Supp. Fig. S5 | *not part of this code yet* |

**Example rules** (Supplementary Section "Example panels"; each module's docstring gives the rule in full):

- Fig. 4B: the cell on which ASTRID's correlation is furthest above the best other method's, its recording on
  which ASTRID is furthest ahead, and the 30 s within the electrophysiology where the methods' standardized
  traces differ most.
- Supp. Fig. S6B, D, F: the video with ASTRID's median F1; among its ROIs with at least 8 labeled or simulated
  transients and an ASTRID F1 of at least 0.5, the ROI on which ASTRID's F1 is furthest above the best other
  method's; the 120 s (or the whole recording) with the most transients on which the methods disagree.
