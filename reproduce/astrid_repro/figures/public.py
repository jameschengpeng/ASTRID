"""Supplementary Figure S6: D2 (Allen Brain Observatory), D4 (NAOMi) and D5 (one-photon), the public datasets
scored by transient detection, one row each. Every panel is drawn from the outputs of ``reproduce/``: the
table of the dataset (``cv_<dataset>.json``, written by ``tables_public public``), the traces every method gave
(ASTRID's ``result.npz``, the peers', Suite2p's) and TUnCaT's figshare archive (the movies, masks and labels).

- **Scores** (panels B, D and F; ``draw_scores``): each method's F1 per video, at the setting and threshold chosen for
  that video by leave-one-video-out cross-validation, the mean as a bar, and above each method the mark of the
  Holm-corrected p of ASTRID against it over the videos, Holm over the five methods. The y axis names the
  reference, the labeled (D2, D5) or simulated (D4) transients, and a note under the axis says why the "+ terr."
  variants of the paper's D1 figures are missing (``common.NO_TERRITORY``).
- **Example** (panels A, C and E; ``example``, ``draw_image``, ``draw_traces``): the mean image of one video, every
  ROI mask in white and the example ROI in yellow, the dataset named below with its descriptor
  (``common.image_name``), and the example ROI with one row per method: its trace as
  the evaluation sees it (template-filtered, in noise SDs above its baseline), the method's threshold, the
  labeled (D2, D5) or simulated (D4) transients, and the transients the method detected (matched or false),
  with the ROI's F1 on the right.

**The example rule** (Supplementary Section "Example panels"). The video is the one on which ASTRID's F1 is
closest to the dataset's median (the first such video in the table's order). Among its ROIs with at least
``MIN_TRANSIENTS`` labeled or simulated transients and an ASTRID F1 of at least ``MIN_ASTRID_F1``, the example
is the ROI on which ASTRID's F1 is furthest above the best other method's (ties: more transients, then the
lower index). Every method is scored at its cross-validated setting and threshold for that video, and the F1
of the video recomputed here must be the table's for every method. The traces show the ``WINDOW_S`` s (or the
whole recording, if shorter) with the most transients on which the methods disagree: the labeled transients
missed by at least one method plus the false detections of every method (ties: more labeled transients, then
the earlier window).

The scoring ROI by ROI is TUnCaT's evaluation, which is GPL-2: it is handed in at run time, as the scorer of
the tables is (``--evaluator tuncat_eval.evaluation:view --cnmf-fill tuncat_eval.evaluation:cnmf_fill``, from
``reproduce/tuncat_eval``).

    python -m astrid_repro.figures public --dataset ABO --table <tables>/cv_ABO.json --archive <archive>
        --ours <root> [...] --peers <root> [...] --suite2p <root>
        --evaluator tuncat_eval.evaluation:view --cnmf-fill tuncat_eval.evaluation:cnmf_fill --out <folder>
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from astrid_repro import public_data as D
from astrid_repro.figures import common as F
from astrid_repro.ring import video_ring

ORDER = F.PUBLIC_ORDER
NAMES = {"ABO": "ABO", "NAOMi": "NAOMi", "1p": "one-photon"}       # printed after the dataset's ID
PANELS = {"ABO": ("figS6b", "figS6a"), "NAOMi": ("figS6d", "figS6c"), "1p": ("figS6f", "figS6e")}  # scores, example
FPS = {"ABO": 30.0, "NAOMi": 30.0, "1p": 20.0}
MIN_TRANSIENTS = 8
MIN_ASTRID_F1 = 0.5                  # an ROI that ASTRID detects well, not only less badly than the others
WINDOW_S = 120.0                     # or the whole recording, if shorter
TRUTH_BAND, TRUTH_LINE = "#DADADA", "#C8C8C8"
THRESHOLD_COLOR = "#606060"

# (dataset, video, archive) -> (traces (n, T), threshold) -> the scoring ROI by ROI (tuncat_eval's View)
Evaluator = Callable[[str, str, str], Callable[[np.ndarray, float], dict[str, Any]]]


def name_of(dataset: str) -> str:
    return f"{D.PAPER_NAMES[dataset]} {NAMES[dataset]}"


# ---- the table ----

def read_table(path: str | Path, dataset: str) -> dict[str, Any]:
    """A dataset's table as ``tables_public public`` writes it, checked: every method's videos in ASTRID's
    order, Suite2p's entry with ASTRID's F1, and the five methods' Holm family. Returns {"record", "suite2p",
    "videos", "values" (method -> F1 per video), "tests" (method -> its test)}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    record, s2p = d["table"], d["suite2p"]
    videos = record["ours"]["videos"]
    for m in ORDER[:5]:
        if record[m]["videos"] != videos:
            raise ValueError(f"{dataset}: {m}'s videos are not in ASTRID's order")
    if s2p is None or s2p["videos"] != videos or not np.allclose(s2p["ours"]["f1"], record["ours"]["f1"],
                                                                  rtol=0, atol=1e-12):
        raise ValueError(f"{dataset}: Suite2p's entry is missing, or its videos or ASTRID's F1 are not the table's")
    if tuple(record["_meta"].get("holm", {}).get("family", ())) != F.PUBLIC_FAMILY:
        raise ValueError(f"{dataset}: the Holm family is not {F.PUBLIC_FAMILY}")
    values = {m: record[m]["f1"] for m in ORDER[:5]} | {"suite2p": s2p["suite2p"]["f1"]}
    tests = {m: record["_meta"]["stats"][F.TEST_KEYS[m]] for m in ORDER[1:5]} | \
        {"suite2p": s2p["stats"]["Suite2p"]}
    F.check_holm(tests, f"{dataset} F1")
    return {"record": record, "suite2p": s2p, "videos": videos, "values": values, "tests": tests}


def draw_scores(ax: Any, dataset: str, table: dict[str, Any]) -> dict[str, Any]:
    """Panel B, D or F: each method's F1 per video, with the mark of each comparison (module docstring)."""
    marks = {m: F.stars(float(t["p_holm"])) for m, t in table["tests"].items()}
    kind = "simulated" if dataset == "NAOMi" else "labeled"
    shown = F.paired_dots(ax, table["values"], ORDER, marks, f"F1 score against {kind}\ntransients (higher is better)",
                          f"{name_of(dataset)}, {len(table['videos'])} videos")
    note = F.NO_TERRITORY.format(name=D.PAPER_NAMES[dataset])
    ax.figure.supxlabel(note)                      # under the axis's label
    return {"dataset": D.PAPER_NAMES[dataset], "measure": "F1 of transient detection per video, at the "
            "cross-validated setting and threshold", "videos": table["videos"], **shown,
            "p_holm": {m: float(t["p_holm"]) for m, t in table["tests"].items()},
            "holm_family": list(F.PUBLIC_FAMILY), "note": note}


# ---- the example ----

def _traces(dataset: str, video: str, table: dict[str, Any], vi: int, archive: Path,
            ours: Sequence[str | Path], peers: Sequence[str | Path] | None, suite2p: str | Path,
            masks: np.ndarray, cnmf_fill: D.CnmfFill | None, cnmf_grid: str | Path | None = None
            ) -> dict[str, tuple[np.ndarray, str, float, float]]:
    """Each method's traces of the video at its chosen setting: method -> (traces, setting, threshold, the
    table's F1 of the video). CNMF's come from its cross-validation grid (``cnmf_grid``, as tables_public reads
    it) where the table chose its setting there."""
    record, s2p = table["record"], table["suite2p"]
    out: dict[str, tuple[np.ndarray, str, float, float]] = {}
    for m in ORDER:
        if m == "suite2p":
            x = D.npz_trace(D.suite2p_path(suite2p, dataset, video))
            out[m] = (x, "suite2p", float(s2p["suite2p"]["threshold"][vi]), float(s2p["suite2p"]["f1"][vi]))
            continue
        setting, thr, f1 = record[m]["setting"][vi], float(record[m]["threshold"][vi]), float(record[m]["f1"][vi])
        if m == "ring":
            x = video_ring(D.movie_path(archive, dataset, video), masks, D.UM_PER_PX[dataset])["trace"]
        elif m == "ours":
            x = D.ours_traces(list(ours), dataset, video)[setting]
        elif m == "CNMF" and cnmf_grid is not None:
            x = D.cnmf_grid_traces(cnmf_grid, dataset, video, masks=masks, cnmf_fill=cnmf_fill)[setting]
        else:
            x = D.peer_traces(list(peers) if peers else D.released_root(archive, dataset), m, video, masks=masks,
                              cnmf_fill=cnmf_fill)[setting]
        out[m] = (np.asarray(x, dtype=float), setting, thr, f1)
    return out


def _window(ex: dict[str, Any], roi: int, n_eval: int, win: int) -> tuple[int, int]:
    """The window of ``win`` frames with the most transients on which the methods disagree (module
    docstring)."""
    events = np.zeros(n_eval, float)
    gt = ex["truth"][roi]
    all_hit = np.all([ex["methods"][m]["truth_matched"][roi] for m in ORDER], axis=0)
    for (s, _), hit in zip(gt, all_hit, strict=True):
        if s < n_eval:                       # a transient starting after the last evaluated frame is not shown
            events[s] += 0 if hit else 1
    for m in ORDER:
        d = ex["methods"][m]
        for (s, _), ok in zip(d["detections"][roi], d["detection_matched"][roi], strict=True):
            events[s] += 0 if ok else 1
    counts = np.convolve(events, np.ones(win), mode="valid")
    starts = np.bincount(gt[:, 0], minlength=n_eval)[:n_eval].astype(float)
    gt_counts = np.convolve(starts, np.ones(win), mode="valid")
    start = int(np.lexsort((np.arange(counts.size), -gt_counts, -counts))[0])
    return start, start + win


def example(dataset: str, table: dict[str, Any], *, archive: str | Path, ours: Sequence[str | Path],
            peers: Sequence[str | Path] | None, suite2p: str | Path, evaluator: Evaluator,
            cnmf_fill: D.CnmfFill | None = None, cnmf_grid: str | Path | None = None) -> dict[str, Any]:
    """The example of one dataset by the rule of the module docstring: its video, ROI and window, and every
    method's scoring of the video ROI by ROI (``evaluator``)."""
    archive = Path(archive)
    f1 = np.asarray(table["values"]["ours"], dtype=float)
    vi = int(np.argmin(np.abs(f1 - np.median(f1))))
    video = table["videos"][vi]
    masks = D.read_masks(D.masks_path(archive, dataset, video))
    raw = _traces(dataset, video, table, vi, archive, ours, peers, suite2p, masks, cnmf_fill, cnmf_grid)
    n = min(x.shape[1] for x, _, _, _ in raw.values())
    score = evaluator(dataset, video, str(archive))
    methods: dict[str, dict[str, Any]] = {}
    for m, (x, setting, thr, table_f1) in raw.items():
        x = x[:, :n]
        x = np.where(np.isfinite(x).all(axis=1)[:, None], x, 0.0)     # the failure rule of the tables
        v = score(x, thr)
        if abs(v["video_f1"] - table_f1) > 1e-9:
            raise ValueError(f"{dataset} {video} {m}: F1 {v['video_f1']} here, {table_f1} in the table")
        methods[m] = {"setting": setting, "threshold": thr, **v}
    truth = methods["ours"]["truth"]
    n_gt = np.array([0 if g is None else len(g) for g in truth])
    labelled = np.array([g is not None for g in truth])
    ours_f1 = methods["ours"]["roi_f1"]
    others = np.array([methods[m]["roi_f1"] for m in ORDER[1:]])
    gain = np.full(len(truth), np.nan)
    gain[labelled] = ours_f1[labelled] - others[:, labelled].max(axis=0)
    eligible = np.flatnonzero((n_gt >= MIN_TRANSIENTS) & np.isfinite(gain) & (ours_f1 >= MIN_ASTRID_F1))
    if eligible.size == 0:
        raise ValueError(f"{dataset} {video}: no ROI has {MIN_TRANSIENTS} transients and an ASTRID F1 of "
                         f"{MIN_ASTRID_F1}")
    roi = int(sorted(eligible, key=lambda i: (-gain[i], -n_gt[i], i))[0])
    n_eval = min(methods[m]["z"].shape[1] for m in ORDER)              # the filtered traces' frames
    fps = FPS[dataset]
    win = min(int(round(WINDOW_S * fps)), n_eval)
    ex: dict[str, Any] = {"dataset": dataset, "video": video, "video_index": vi, "roi": roi, "masks": masks,
                          "truth": truth, "n_gt": n_gt, "gain": gain, "methods": methods, "fps": fps,
                          "frames": n_eval, "movie": D.movie_path(archive, dataset, video),
                          "um_per_px": D.UM_PER_PX[dataset],
                          "best_other": ORDER[1 + int(np.argmax(others[:, roi]))]}
    ex["window"] = _window(ex, roi, n_eval, win)
    return ex


def mean_image(movie: str | Path, block: int = 500) -> np.ndarray:
    """The mean over all frames of an HDF5 movie (``mov``, (T, H, W))."""
    with h5py.File(movie, "r") as f:
        ds = f["mov"]
        acc = np.zeros(ds.shape[1:], dtype=np.float64)
        for s in range(0, ds.shape[0], block):
            acc += np.asarray(ds[s:s + block], dtype=np.float64).sum(axis=0)
        return acc / ds.shape[0]


def draw_image(ax: Any, ex: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    """The left part of panel A, C or E: the video's mean image (gray, from its 1st to its 99.5th
    percentile), every ROI mask in white and the example in yellow, and a scale bar below the image (20 um
    on a field narrower than 120 um, else 50 um)."""
    masks = np.asarray(ex["masks"], dtype=bool)
    lo, hi = np.percentile(image, [1, 99.5])
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    for i, m in enumerate(masks):
        if i != ex["roi"]:
            ax.contour(m.astype(float), levels=[0.5], colors="white", linewidths=0.3, alpha=0.6)
    ax.contour(masks[ex["roi"]].astype(float), levels=[0.5], colors=[F.EXAMPLE_COLOR], linewidths=0.8)
    um = ex["um_per_px"]
    h, w = image.shape
    bar_um = 20 if w * um < 120 else 50
    bar_px = bar_um / um
    y_bar = h + 0.04 * h                                     # below the image
    ax.plot([w - bar_px, w], [y_bar, y_bar], color="black", lw=1.2, solid_capstyle="butt", clip_on=False)
    ax.text(w - bar_px / 2, y_bar + 0.02 * h, f"{bar_um} µm", color="black", ha="center", va="top",
            clip_on=False)
    ax.set_ylim(h + 0.16 * h, -0.5)
    ax.set_axis_off()
    return {"video": ex["video"], "example ROI": ex["roi"] + 1, "ROIs": len(masks), "scale bar (um)": bar_um,
            "gray scale": [float(lo), float(hi)]}


def draw_traces(fig: Any, ex: dict[str, Any]) -> dict[str, Any]:
    """The right part of panel A, C or E: the example ROI, one row per method (module docstring)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    axes = fig.subplots(len(ORDER), 1, sharex=True)
    roi, (s, e), fps = ex["roi"], ex["window"], ex["fps"]
    t = np.arange(s, e) / fps
    gt = ex["truth"][roi]
    rows = []
    for r, (ax, m) in enumerate(zip(axes, ORDER, strict=True)):
        d = ex["methods"][m]
        for (gs, ge) in gt:
            if ge >= s and gs < e:
                ax.axvspan(max(gs, s) / fps, min(ge + 1, e) / fps, color=TRUTH_BAND, lw=0, zorder=0)
                ax.axvline((gs + ge + 1) / 2 / fps, color=TRUTH_LINE, lw=0.5, zorder=0)
        z = d["z"][roi, s:e]
        ax.plot(t, z, color=F.color(m), lw=0.6, zorder=2)
        ax.axhline(d["threshold"], color=THRESHOLD_COLOR, lw=0.5, ls=(0, (2, 2)), zorder=1)
        top = max(float(np.nanmax(z)), d["threshold"]) * 1.05
        bottom = min(float(np.nanmin(z)), 0.0)
        bar_y = bottom - 0.14 * (top - bottom)
        shown = [0, 0]
        for (ds_, de_), ok in zip(d["detections"][roi], d["detection_matched"][roi], strict=True):
            if de_ >= s and ds_ < e:
                lo_, hi_ = max(ds_, s), min(de_, e - 1)
                peak = (lo_ + int(np.argmax(d["z"][roi, lo_:hi_ + 1]))) / fps
                if ok:
                    ax.plot(peak, bar_y, "o", ms=3.5, color=F.color(m), mew=0, zorder=3)
                else:
                    ax.plot(peak, bar_y, "x", ms=3.5, color="black", mew=0.8, zorder=3)
                shown[0 if ok else 1] += 1
        ax.set_ylim(bar_y - 0.1 * (top - bottom), top)
        F.trace_row_style(ax, F.label(m), True, r == len(ORDER) - 1, None if m == "ours" else F.color(m))
        F.score_label(ax, f"{d['roi_f1'][roi]:.2f}")
        rows.append({"method": m, "setting": d["setting"], "threshold": d["threshold"],
                     "roi_f1": float(d["roi_f1"][roi]), "video_f1": d["video_f1"],
                     "detections shown (matched, false)": shown})
    F.column_footer(axes[-1], "F1")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].spines["bottom"].set_position(("outward", 3))          # clear of the detection markers
    kind = "Simulated" if ex["dataset"] == "NAOMi" else "Labeled"
    key = [Patch(facecolor=TRUTH_BAND, edgecolor=TRUTH_LINE, lw=0.5, label=f"{kind} transient"),
           Line2D([], [], ls="", marker="o", ms=3.5, mew=0, color=THRESHOLD_COLOR, label="Detected, matched"),
           Line2D([], [], ls="", marker="x", ms=3.5, mew=0.8, color="black", label="Detected, false"),
           Line2D([], [], ls=(0, (2, 2)), lw=0.5, color=THRESHOLD_COLOR, label="Threshold")]
    fig.legend(handles=key, loc="outside lower center", ncols=2, frameon=False)
    return {"dataset": D.PAPER_NAMES[ex["dataset"]], "video": ex["video"], "example ROI": roi + 1,
            "transients of the ROI": int(ex["n_gt"][roi]), "best other method": ex["best_other"],
            "ASTRID's F1 above the best other method's": float(ex["gain"][roi]),
            "window (frames)": [s, e], "window (s)": [s / fps, (e - 1) / fps],
            "whole recording": e - s >= ex["frames"], "rows": rows,
            "rule": "astrid_repro.figures.public (module docstring); ROI numbers are 1-based"}


# ---- the panels of one dataset ----

def panels(dataset: str, table_path: str | Path, *, archive: str | Path, ours: Sequence[str | Path],
           peers: Sequence[str | Path] | None, suite2p: str | Path, evaluator: Evaluator,
           cnmf_fill: D.CnmfFill | None = None, cnmf_grid: str | Path | None = None) -> list[F.Panel]:
    """The dataset's row of the figure: (file stem, figure, record) for its scores, its example's
    image and its example's traces."""
    import matplotlib.pyplot as plt

    table = read_table(table_path, dataset)
    scores_panel, example_panel = PANELS[dataset]
    tag = D.PAPER_NAMES[dataset]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    out = [(f"{scores_panel}_{tag}_scores", fig, draw_scores(ax, dataset, table))]
    ex = example(dataset, table, archive=archive, ours=ours, peers=peers, suite2p=suite2p, evaluator=evaluator,
                 cnmf_fill=cnmf_fill, cnmf_grid=cnmf_grid)
    fig, ax = plt.subplots(figsize=(3.2, 3.4), layout="constrained")
    fig.supxlabel(F.image_name(tag))
    out.append((f"{example_panel}_{tag}_image", fig, draw_image(ax, ex, mean_image(ex["movie"]))))
    fig = plt.figure(figsize=(7.0, 5.6), layout="constrained")
    out.append((f"{example_panel}_{tag}_traces", fig, draw_traces(fig, ex)))
    return out
