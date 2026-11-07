"""Leave-one-out cross-validation over a method's settings and the detection thresholds (the paper's Methods,
"Benchmark protocol").

- **D2, D4, D5:** per video, a method's grid of recall, precision and F1 over its settings (rows) and the
  detection thresholds (columns) (``f1_grid``); for each held-out video, the (setting, threshold) with the
  highest mean F1 over the other videos is applied to it (``loo_cv``). The F1 is that of TUnCaT's transient
  detection and matching, computed by a scorer passed in: TUnCaT's evaluation code (GPL-2) is not part of this
  package. ``read_tuncat_scores`` reads the grids its MATLAB evaluation saves.
- **D3:** per recording, the correlation with the spike-predicted calcium; per cell, the mean over its
  recordings; for each held-out cell, the setting with the highest mean over the other cells (``cell_cv``).

Every method's threshold, like its settings, is chosen this way, on one grid of thresholds per dataset
(``THRESHOLDS``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

# one grid per dataset for every method, a superset of the ranges TUnCaT used per method
THRESHOLDS = {"ABO": tuple(range(10, 155, 5)), "NAOMi": tuple(range(10, 120, 10)),
              "1p": tuple(range(0, 820, 20))}

# (traces (n, T), thresholds (k,)) -> (recall, precision, F1) per threshold, the number of labelled ROIs
# scored and the number of ground-truth transients
Scorer = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]

# TUnCaT's published grids of the peers on the raw videos (results/<ds>/evaluation in its archive)
NAOMI_SIM = "120s_30Hz_N=200_100mW_noise10+23_NA0.8,0.6_GCaMP6f"
TUNCAT_SCORE_FILES = {
    "ABO": {"FISSA": "scores_split_FISSA_RawVideo_UnmixSigma_ksd-psd.mat",
            "TUnCaT": "scores_split_ours_RawVideo_UnmixSigma_ksd-psd.mat",
            "CNMF": "scores_split_CNMF_RawVideo_p1_sum_ksd-psd.mat",
            "allen": "scores_split_AllenSDK_RawVideo_Unmix_ksd-psd.mat",
            "bgsubs": "scores_split_bgsubs_RawVideo_ksd-psd.mat"},
    "1p": {"FISSA": "scores_split_FISSA_RawVideo_UnmixSigma_ksd-psd.mat",
           "TUnCaT": "scores_split_ours_RawVideo_UnmixSigma_ksd-psd.mat",
           "CNMF": "scores_split_CNMF_RawVideo_p1_sumSigma_ksd-psd.mat",
           "allen": "scores_split_AllenSDK_RawVideo_Unmix_ksd-psd.mat",
           "bgsubs": "scores_split_bgsubs_RawVideo_ksd-psd.mat"},
    "NAOMi": {m: f"scores_split_{n}_{NAOMI_SIM}_RawVideo_{s}_ksd-psd.mat"
              for m, n, s in (("FISSA", "FISSA", "Unmix_Sigma"), ("TUnCaT", "ours", "Unmix_Sigma"),
                              ("CNMF", "CNMF", "p1_sumSigma"), ("allen", "AllenSDK", "Unmix_Sigma"),
                              ("bgsubs", "bgsubs", "Raw_Sigma"))},
}


@dataclass
class Grid:
    """One method on one video: recall, precision and F1, each (settings, thresholds)."""
    settings: list[str]
    thresholds: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    f1: np.ndarray
    n_neurons: int
    n_gt: int


def save_grid(path: str | Path, g: Grid) -> None:
    np.savez(path, settings=np.array(g.settings), thresholds=g.thresholds, recall=g.recall,
             precision=g.precision, f1=g.f1, n_neurons=g.n_neurons, n_gt=g.n_gt)


def load_grid(path: str | Path) -> Grid:
    with np.load(path, allow_pickle=False) as z:
        return Grid(z["settings"].tolist(), z["thresholds"], z["recall"], z["precision"], z["f1"],
                    int(z["n_neurons"]), int(z["n_gt"]))


def trim(traces: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every setting's traces cut to the shortest one's frames."""
    n = min(x.shape[1] for x in traces.values())
    return {k: x[:, :n] for k, x in traces.items()}


def f1_grid(traces: dict[str, np.ndarray], scorer: Scorer, thresholds: Sequence[float] | np.ndarray,
            n_rois: int | None = None) -> Grid:
    """The grid of one method on one video: ``traces`` maps each setting to its (n, T) traces, ``scorer`` scores
    them at every threshold (``Scorer``). The failure rule: an ROI whose trace is missing or not finite gets a
    zero trace, so it detects nothing and its transients count as missed."""
    names = list(traces)
    thr = np.asarray(thresholds, dtype=float)
    rec, pre, f1 = (np.zeros((len(names), thr.size)) for _ in range(3))
    n_neurons = n_gt = 0
    for i, name in enumerate(names):
        x = np.atleast_2d(np.asarray(traces[name], dtype=float))
        if n_rois is not None and x.shape[0] != n_rois:
            raise ValueError(f"{name}: {x.shape[0]} traces for {n_rois} ROIs")
        bad = ~np.isfinite(x).all(axis=1)
        x = np.where(bad[:, None], 0.0, x)
        r, p, f, n_neurons, n_gt = scorer(x, thr)
        rec[i], pre[i], f1[i] = r, p, f
    return Grid(names, thr, rec, pre, f1, n_neurons, n_gt)


@dataclass
class CV:
    videos: list[str]
    setting: list[str]           # chosen for each held-out video
    threshold: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    f1: np.ndarray               # held-out F1 per video

    @property
    def mean_f1(self) -> float:
        return float(self.f1.mean())


def loo_cv(videos: Sequence[str], grids: Sequence[Grid]) -> CV:
    """Leave one video out: the (setting, threshold) with the highest mean F1 over the other videos (the first
    maximum, settings before thresholds), applied to the held-out one. Every video needs the same settings and
    thresholds."""
    names, thr = grids[0].settings, grids[0].thresholds
    for g in grids:
        if g.settings != names or not np.array_equal(g.thresholds, thr):
            raise ValueError("every video needs the same settings and thresholds")
    F = np.stack([g.f1 for g in grids])                       # (videos, settings, thresholds)
    n = len(videos)
    chosen_s, chosen_t, rec, pre, f1 = [], [], [], [], []
    for v in range(n):
        train = [i for i in range(n) if i != v]
        mean = F[train].mean(axis=0)
        s, t = np.unravel_index(int(np.argmax(mean)), mean.shape)
        chosen_s.append(names[s])
        chosen_t.append(thr[t])
        rec.append(grids[v].recall[s, t])
        pre.append(grids[v].precision[s, t])
        f1.append(grids[v].f1[s, t])
    return CV(list(videos), chosen_s, np.array(chosen_t), np.array(rec), np.array(pre), np.array(f1))


def read_tuncat_scores(path: str | Path, setting: str = "fixed") -> list[Grid]:
    """The per-video grids TUnCaT's MATLAB evaluation saves (``list_recall``, ``list_precision``, ``list_F1``
    as (videos, alphas, thresholds), or (videos, thresholds) for a method with one setting;
    ``list_thred_ratio``; ``list_alpha``), one Grid per video. Settings are named "a=<alpha>", else
    ``setting``."""
    d = scipy.io.loadmat(path)
    thr = np.asarray(d["list_thred_ratio"], dtype=float).ravel()
    alpha = np.asarray(d["list_alpha"], dtype=float).ravel() if "list_alpha" in d else None
    arrays = []
    for key in ("list_recall", "list_precision", "list_F1"):
        a = np.asarray(d[key], dtype=float)
        arrays.append(a[:, None, :] if a.ndim == 2 else a)
    rec, pre, f1 = arrays
    names = [f"a={x:g}" for x in alpha] if alpha is not None and f1.shape[1] == alpha.size else [setting]
    return [Grid(list(names), thr, rec[v], pre[v], f1[v], 0, 0) for v in range(f1.shape[0])]


def cell_cv(per_recording: dict[str, dict[str, float]], cell_of: dict[str, str],
            cells: Sequence[str]) -> dict[str, Any] | None:
    """D3's leave-one-cell-out CV of one method. ``per_recording``: setting -> {recording: r}, the recordings
    in the export's order. Each cell's score is the mean of its recordings' r; for each held-out cell, the
    setting (in sorted order, the first maximum) with the highest mean over the other cells. None if every
    score is NaN."""
    names = sorted(per_recording)
    table = np.full((len(names), len(cells)), np.nan)
    for i, s in enumerate(names):
        for j, c in enumerate(cells):
            vals = [r for n, r in per_recording[s].items() if cell_of[n] == c]
            table[i, j] = np.nanmean(vals) if vals else np.nan
    if np.isnan(table).all():
        return None
    chosen, held = [], np.full(len(cells), np.nan)
    for j in range(len(cells)):
        train = [k for k in range(len(cells)) if k != j]
        mean = np.nanmean(table[:, train], axis=1)
        i = int(np.nanargmax(mean))
        chosen.append(names[i])
        held[j] = table[i, j]
    return {"settings": names, "cells": list(cells), "chosen": chosen, "r": held.tolist(),
            "mean_r": float(np.nanmean(held)), "n_recordings": len(per_recording[names[0]])}
