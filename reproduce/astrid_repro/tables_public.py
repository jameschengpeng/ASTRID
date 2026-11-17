"""The D2-D5 tables: every method's score per unit after leave-one-out cross-validation (``cv``), its mean and
SD, and ASTRID against each other method by a paired two-sided Wilcoxon signed-rank test over the units (videos;
cells on D3), with Holm's correction over the paper's five comparisons: TUnCaT, FISSA, CNMF, Ring and Suite2p.

The records keep the layout of the analysis that made the paper's tables. ``public_record`` gives a dataset's
table (each method's CV result; ``_meta.stats``, ASTRID's tests) and Suite2p's entry; ``cai1_record`` the same for
D3. Each test keeps its raw ``p``; for TUnCaT, FISSA and CNMF also ``p_holm_declared`` (Holm over those three,
the family declared first) and ``p_holm_suite2p`` (with Suite2p, four); ``p_holm`` is the paper's (five).

From traces to grids (``public_grids``) the F1 comes from a scorer passed in (``cv.Scorer``): TUnCaT's
evaluation code (GPL-2) is not part of this package.

CNMF's settings are chosen by the same cross-validation as the other methods' (``--cnmf-grid``: its six settings,
``cv.CNMF_GRID``, each in its own folder; ``public_data.cnmf_grid_traces``, ``cai1_cnmf_grid``): the row ``CNMF``
is then the cross-validated one, and ``CNMF_fixed``, read from ``--peers`` (the published setting, gnb 1 and
p 1), is a reference row, not tested. Without ``--cnmf-grid`` the row ``CNMF`` is the fixed setting, as before.

D3 has two scorings (``cai1 --baseline``), and the paper reports both. ``reported`` (the default): ASTRID's ΔF/F0,
its running baseline removed by the fit, and every other trace with only its mean removed by the correlation.
``equal``: every trace but ASTRID's has its own running baseline removed by ASTRID's rule
(``public_data.minus_baseline``) before the correlation, the cross-validation and the tests then as in
``reported``; written as ``cv_cai1_equal_baseline``.

D5's nine sub-videos are crops of three recordings (``public_data.one_photon_recording``). By default each
sub-video is held out alone, as in TUnCaT's evaluation. ``public --folds recording`` holds out the three crops
of a recording together (``cv.group_cv``), for every method, so no crop of the held-out recording helps choose
its settings or threshold; written as ``cv_1p_by_recording``.

Run, with ``reproduce/`` on the path (every path is an argument):
    python -m astrid_repro.tables_public grids --dataset ABO --archive <TUnCaT archive> --ours <root> [...]
        --peers <root> [...] --cnmf-grid <grid> --suite2p <root> --scorer <module:function> --out <grids>
    python -m astrid_repro.tables_public public --dataset ABO --grids <grids> --out <folder>
    python -m astrid_repro.tables_public public --dataset 1p --grids <grids> --folds recording --out <folder>
    python -m astrid_repro.tables_public cai1 --export <export> --ours <root> [...] --peers <folder>
        --cnmf-grid <grid> --suite2p <root> [--baseline equal] --out <folder>
``--scorer`` names a function ``(dataset, video, archive) -> cv.Scorer``; ``--cnmf-fill`` one
``(mat, masks) -> (n, T)`` for CNMF results with dropped components (``public_data.peer_traces``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from astrid_repro import cv as C
from astrid_repro import public_data as D
from astrid_repro.metrics import holm, wilcoxon_p
from astrid_repro.ring import cai1_reference_rows, video_ring

PEERS = ("TUnCaT", "FISSA", "CNMF")                  # the declared comparisons
CNMF_FIXED = "CNMF_fixed"                            # CNMF at the published setting, beside its grid: a reference
REFERENCE = ("raw", "bgsubs", "allen")               # reference rows, not tested (D2, D4, D5)
METHODS = ("ours", *PEERS, CNMF_FIXED, *REFERENCE, "ring", "suite2p")
CUTS = (0.05, 0.01, 0.001)                           # the figures' stars
BASELINES = ("reported", "equal")                    # D3's scorings (cai1_scores)
EQUAL_BASELINE = ("every trace but ASTRID's minus its own running baseline under ASTRID's rule (the 10th percentile "
                  "of 1-s means in a moving 60-s window) before the correlation; ASTRID's trace as reported")

ScorerFactory = Callable[[str, str, str], C.Scorer]


# ---- the tests ----

def cv_entry(res: C.CV) -> dict[str, Any]:
    return {"videos": res.videos, "setting": res.setting, "threshold": res.threshold.tolist(),
            "recall": res.recall.tolist(), "precision": res.precision.tolist(), "f1": res.f1.tolist(),
            "mean_f1": res.mean_f1}


def video_tests(results: dict[str, C.CV]) -> dict[str, dict[str, float]]:
    """ASTRID minus each declared peer over the videos: mean difference, videos where ASTRID is better, the
    Wilcoxon p and Holm's over the three (``p_holm``)."""
    stats: dict[str, dict[str, float]] = {}
    raw_p: dict[str, float] = {}
    for m in PEERS:
        if m in results:
            d = results["ours"].f1 - results[m].f1
            p = wilcoxon_p(d)
            raw_p[m] = p
            stats[m] = {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "p": p}
    for m, ph in holm(raw_p).items():
        stats[m]["p_holm"] = ph
    return stats


def paired_test(ours: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    """ASTRID against another method over the units where both are finite, higher better."""
    d = np.asarray(ours, float) - np.asarray(other, float)
    d = d[np.isfinite(d)]
    return {"mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "n": int(d.size), "p": wilcoxon_p(d)}


def added_test(ours: np.ndarray, other: np.ndarray, better: str) -> dict[str, Any]:
    """ASTRID against Suite2p over paired units, as first reported (uncorrected; ``amend`` adds Holm's)."""
    d = np.asarray(ours, float) - np.asarray(other, float)
    wins = int((d < 0).sum()) if better == "lower" else int((d > 0).sum())
    return {"mean_diff": float(d.mean()), "wins": wins, "n": int(d.size), "p": wilcoxon_p(d), "added": True,
            "holm": False}


def amend(stats: dict[str, dict[str, Any]], s2p: dict[str, Any], s2p_key: str = "Suite2p",
          ring_key: str | None = None, where: str = "") -> dict[str, Any]:
    """Holm over the declared comparisons of ``stats``, over them with Suite2p (``s2p``, its test) and, with
    ``ring_key``, with Ring too (its test in ``stats``). Writes ``p_holm_declared``, ``p_holm`` and, with Ring,
    ``p_holm_suite2p`` into the tests (in place; the stored declared value must be Holm's of the raw p) and
    returns the family: its comparisons in order and each one's p before and after."""
    declared = [m for m in stats if m != ring_key]
    raw = {m: float(stats[m]["p"]) for m in declared}
    old, mid = holm(raw), holm({**raw, s2p_key: float(s2p["p"])})
    new = holm({**raw, s2p_key: float(s2p["p"]), ring_key: float(stats[ring_key]["p"])}) if ring_key else mid
    for m in declared:
        s = stats[m]
        stored = float(s.get("p_holm_declared", s["p_holm"]))
        if not math.isclose(stored, old[m], rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(f"{where}, {m}: the stored declared p_holm {stored!r} is not Holm's of raw p")
        s["p_holm_declared"], s["p_holm"] = old[m], new[m]
        if ring_key:
            s["p_holm_suite2p"] = mid[m]
    s2p.update(p_holm=new[s2p_key], p_reported_as_added=float(s2p["p"]), added=False, holm=True)
    comparisons = {m: {"n": stats[m].get("n"), "p": raw[m], "p_before": old[m], "p_holm": new[m]}
                   for m in raw}
    comparisons[s2p_key] = {"n": s2p.get("n"), "p": float(s2p["p"]), "p_before": float(s2p["p"]),
                            "p_holm": new[s2p_key], "before": "uncorrected (added)"}
    family = [*raw, s2p_key]
    if ring_key:
        s2p["p_holm_suite2p"] = mid[s2p_key]
        stats[ring_key]["p_holm"] = new[ring_key]
        for m in comparisons:
            comparisons[m]["p_suite2p"] = mid[m]
        comparisons[ring_key] = {"n": stats[ring_key].get("n"), "p": float(stats[ring_key]["p"]),
                                 "p_before": None, "p_suite2p": None, "p_holm": new[ring_key],
                                 "before": "no test (a reference row, or no result)"}
        family.insert(len(raw), ring_key)                       # TUnCaT, FISSA, CNMF, Ring, Suite2p
    return {"family": family, "comparisons": comparisons}


def crossed(before: float | None, after: float) -> list[float]:
    return [] if before is None else [c for c in CUTS if before < c <= after]


def holm_rows(dataset: str, measure: str, family: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per comparison of a dataset's family: raw p, before (the declared family), with Suite2p, the
    paper's, and the stars' thresholds crossed."""
    rows = []
    for m, c in family["comparisons"].items():
        row = {"scope": "scores", "dataset": dataset, "measure": measure, "comparison": m, **c,
               "changed": c["p_holm"] != c["p_before"], "crosses": crossed(c["p_before"], c["p_holm"])}
        if "p_suite2p" in c:
            row["crosses_since_suite2p"] = crossed(c["p_suite2p"], c["p_holm"])
        rows.append(row)
    return rows


# ---- D2, D4, D5 ----

def public_record(dataset: str, cvs: dict[str, C.CV], thresholds: Sequence[float] | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """A dataset's table from each method's CV (``cvs``: "ours", the peers, the reference rows, "ring",
    "suite2p"): (the record, Suite2p's entry, the Holm family), each None where its methods are missing."""
    record: dict[str, Any] = {m: cv_entry(cvs[m]) for m in ("ours", *PEERS, CNMF_FIXED, *REFERENCE) if m in cvs}
    stats = video_tests(cvs) if "ours" in cvs else {}
    thr = list(C.THRESHOLDS[dataset] if thresholds is None else thresholds)
    if "ring" in cvs:
        record["ring"] = cv_entry(cvs["ring"])
        if "ours" in cvs:
            stats["Ring"] = paired_test(np.asarray(record["ours"]["f1"]), cvs["ring"].f1)
    record["_meta"] = {"stats": stats, "thresholds": thr}
    if "suite2p" not in cvs or "ours" not in cvs:
        return record, None, None
    s2p, ours = cvs["suite2p"], np.asarray(record["ours"]["f1"], float)
    entry = {"dataset": D.PAPER_NAMES[dataset], "unit": "video", "videos": record["ours"]["videos"],
             "suite2p": {"threshold": s2p.threshold.tolist(), "recall": s2p.recall.tolist(),
                         "precision": s2p.precision.tolist(), "f1": s2p.f1.tolist(), "mean_f1": s2p.mean_f1,
                         "sd_f1": float(s2p.f1.std(ddof=1))},
             "ours": {"f1": ours.tolist(), "mean_f1": float(ours.mean()), "sd_f1": float(ours.std(ddof=1))},
             "stats": {"Suite2p": added_test(ours, s2p.f1, "higher")}}
    family = amend(stats, entry["stats"]["Suite2p"], "Suite2p", "Ring" if "Ring" in stats else None,
                   f"{D.PAPER_NAMES[dataset]} {dataset}")
    record["_meta"]["holm"] = {"family": family["family"]}
    return record, entry, family


FOLDS = ("video", "recording")


def cv_by_method(grids: dict[str, list[C.Grid]], videos: Sequence[str], folds: str = "video"
                 ) -> dict[str, C.CV]:
    """Each method's CV, for the methods with a grid on every video: leave one video out, or, with ``folds``
    "recording" (D5 only), the sub-videos of a recording held out together."""
    if folds not in FOLDS:
        raise ValueError(f"folds must be one of {FOLDS}, got {folds!r}")
    if folds == "video":
        return {m: C.loo_cv(videos, g) for m, g in grids.items() if len(g) == len(videos) and len(videos) > 1}
    groups = [D.one_photon_recording(v) for v in videos]
    return {m: C.group_cv(videos, g, groups) for m, g in grids.items() if len(g) == len(videos)}


def _grid_task(args: tuple[Any, ...]) -> tuple[str, str, C.Grid | None]:
    method, dataset, video, archive, ours, peers, suite2p, factory, cnmf_fill, ring_trace, cnmf_grid = args
    masks = D.read_masks(D.masks_path(archive, dataset, video))
    released = D.released_root(archive, dataset)
    if method == "ours":
        traces = D.ours_traces(ours, dataset, video)
    elif method == "CNMF" and cnmf_grid:
        traces = D.cnmf_grid_traces(cnmf_grid, dataset, video, masks=masks, cnmf_fill=cnmf_fill)
    elif method == CNMF_FIXED:
        traces = D.peer_traces(peers or released, "CNMF", video, masks=masks, cnmf_fill=cnmf_fill)
    elif method in PEERS:
        traces = D.peer_traces(peers or released, method, video, masks=masks, cnmf_fill=cnmf_fill)
    elif method in REFERENCE:
        traces = D.peer_traces(released, method, video, masks=masks, cnmf_fill=cnmf_fill)
    elif method == "ring":
        traces = {"ring": ring_trace}
    else:
        p = D.suite2p_path(suite2p, dataset, video)
        traces = {"suite2p": D.npz_trace(p)} if p.exists() else {}
    if not traces:
        return method, video, None
    grid = C.f1_grid(C.trim(traces), factory(dataset, video, str(archive)), C.THRESHOLDS[dataset],
                     n_rois=len(masks))
    return method, video, grid


def public_grids(dataset: str, archive: str | Path, scorer: ScorerFactory, *,
                 ours: Sequence[str | Path] | None = None, peers: Sequence[str | Path] | None = None,
                 suite2p: str | Path | None = None, videos: Sequence[str] | None = None,
                 cnmf_fill: D.CnmfFill | None = None, ring: bool = True, workers: int = 1,
                 log: Callable[[str], None] = print, cnmf_grid: str | Path | None = None
                 ) -> tuple[dict[str, list[C.Grid]], dict[str, Any]]:
    """Every method's grid on every video of a dataset: ASTRID's settings (``ours`` roots), the peers (their
    reruns, ``peers``; else the archive's released traces), CNMF over its grid (``cnmf_grid``, with
    ``CNMF_fixed`` from the peers), the reference rows (released), the ring (computed here from the archive's
    movie) and Suite2p (``suite2p`` root). ``scorer(dataset, video, archive)`` gives the video's scorer. A method
    missing on a video is left out. Returns (method -> grids in video order, the ring traces per video)."""
    videos = list(videos or D.VIDEOS[dataset])
    rings: dict[str, Any] = {}
    if ring:
        for v in videos:
            masks = D.read_masks(D.masks_path(archive, dataset, v))
            rings[v] = video_ring(D.movie_path(archive, dataset, v), masks, D.UM_PER_PX[dataset])
            log(f"{dataset} {v}: ring of {len(masks)} ROIs")
    todo = [m for m in METHODS if (m != "ours" or ours) and (m != "suite2p" or suite2p) and (m != "ring" or ring)
            and (m != CNMF_FIXED or cnmf_grid)]
    tasks = [(m, dataset, v, str(archive), [str(r) for r in ours or []], [str(r) for r in peers or []],
              None if suite2p is None else str(suite2p), scorer, cnmf_fill,
              rings[v]["trace"] if m == "ring" else None, None if cnmf_grid is None else str(cnmf_grid))
             for m in todo for v in videos]
    got: dict[tuple[str, str], C.Grid | None] = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for method, video, g in ex.map(_grid_task, tasks):
                got[method, video] = g
                log(f"{dataset} {method} {video}: {'grid' if g is not None else 'no traces'}")
    else:
        for t in tasks:
            method, video, g = _grid_task(t)
            got[method, video] = g
            log(f"{dataset} {method} {video}: {'grid' if g is not None else 'no traces'}")
    out: dict[str, list[C.Grid]] = {}
    for m in todo:
        gs = [got[m, v] for v in videos]
        if all(g is not None for g in gs):
            out[m] = [g for g in gs if g is not None]
        else:
            log(f"{dataset} {m}: no traces for some videos; row left out")
    return out, rings


# ---- D3 ----

def _check_baseline(baseline: str) -> None:
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")


def _cai1_r(trace: np.ndarray, pred: np.ndarray, ok: np.ndarray, fps: float | None) -> float:
    """A trace's D3 score; with ``fps``, after its own running baseline is removed (``D.minus_baseline``)."""
    return D.corr(trace if fps is None else D.minus_baseline(trace, fps), pred, ok)


def cai1_scores(export: str | Path, ours: Sequence[str | Path] | None = None, peers: str | Path | None = None,
                cnmf_grid: str | Path | None = None, baseline: str = "reported"
                ) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    """Every method's r per recording on D3: (the export's index, method -> setting -> {recording: r}), the
    reference rows first ("raw", "ring", "authors_ring"), then ASTRID and the peers; with ``cnmf_grid``, CNMF
    over its grid and ``CNMF_fixed`` from ``peers``. ``baseline`` (module docstring): "reported", or "equal",
    every trace but ASTRID's scored after its own running baseline is removed by ASTRID's rule."""
    _check_baseline(baseline)
    index = D.read_cai1_index(export)
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for row in index:
        name = row["name"]
        pred, ok = D.read_cai1_scored(export, name)
        fps = D.read_cai1_fps(export, name) if baseline == "equal" else None
        for key, tr in cai1_reference_rows(Path(export) / f"{name}.h5").items():
            scores.setdefault(key, {}).setdefault(key, {})[name] = _cai1_r(tr, pred, ok, fps)
        for method in (["ours"] if ours else []) + list(PEERS) + ([CNMF_FIXED] if cnmf_grid else []):
            if method == "CNMF" and cnmf_grid:
                traces = D.cai1_cnmf_grid(cnmf_grid, name)
            else:
                traces = D.cai1_traces("CNMF" if method == CNMF_FIXED else method, name, peers, ours)
            for setting, tr in traces.items():
                r = _cai1_r(tr, pred, ok, None if method == "ours" else fps)    # ASTRID's: ΔF/F0 as it is
                scores.setdefault(method, {}).setdefault(setting, {})[name] = r
    return index, scores


def cai1_suite2p_scores(export: str | Path, root: str | Path, index: list[dict[str, Any]],
                        baseline: str = "reported") -> dict[str, float]:
    """Suite2p's r per recording on D3 (its trace ``<root>/cai1/<name>/suite2p/<name>.npz``), scored as
    ``cai1_scores`` scores the peers under ``baseline``."""
    _check_baseline(baseline)
    out: dict[str, float] = {}
    for row in index:
        name = row["name"]
        pred, ok = D.read_cai1_scored(export, name)
        fps = D.read_cai1_fps(export, name) if baseline == "equal" else None
        out[name] = _cai1_r(D.npz_trace(D.suite2p_path(root, "cai1", name))[0], pred, ok, fps)
    return out


def cai1_record(index: list[dict[str, Any]], scores: dict[str, dict[str, dict[str, float]]],
                suite2p: dict[str, float] | None = None
                ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """D3's table from the per-recording r (``cai1_scores``; Suite2p's from ``cai1_suite2p_scores``): (the
    record, Suite2p's entry, the Holm family). A method missing a recording is refused."""
    cells = sorted({row["cell"] for row in index})
    cell_of = {row["name"]: row["cell"] for row in index}
    every = {row["name"] for row in index}
    incomplete = {f"{m} {s}": sorted(every - set(recs))[:3] for m, settings in scores.items()
                  for s, recs in settings.items() if set(recs) != every}
    if incomplete:
        raise ValueError(f"incomplete results (method setting: missing recordings): {incomplete}")
    results: dict[str, dict[str, Any]] = {}
    for method, settings in scores.items():
        res = C.cell_cv(settings, cell_of, cells)
        if res is not None:
            results[method] = res
    stats: dict[str, dict[str, Any]] = {}
    if "ours" in results:
        raw_p = {}
        for m in PEERS:
            if m in results:
                stats[m] = paired_test(np.array(results["ours"]["r"]), np.array(results[m]["r"]))
                raw_p[m] = stats[m]["p"]
        for m, ph in holm(raw_p).items():
            stats[m]["p_holm"] = ph
    record: dict[str, Any] = {"results": results, "stats": stats}
    if "ring" in results and "ours" in results:
        per_rec = dict(scores["ring"]["ring"])
        mine = np.array([np.nanmean([r for n, r in per_rec.items() if cell_of[n] == c]) for c in cells])
        if not np.allclose(mine, results["ring"]["r"], rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError("D3: the ring's per-recording r do not give its per-cell r")
        results["ring"]["per_recording"] = per_rec
        stats["Ring"] = paired_test(np.asarray(results["ours"]["r"]), mine)
    if suite2p is None or "ours" not in results:
        return record, None, None
    by_cell = {c: [r for n, r in suite2p.items() if cell_of[n] == c] for c in results["ours"]["cells"]}
    r = np.array([np.nanmean(by_cell[c]) for c in results["ours"]["cells"]])
    ours = np.asarray(results["ours"]["r"], float)
    ok = np.isfinite(r) & np.isfinite(ours)
    entry = {"dataset": "D3", "unit": "cell", "cells": results["ours"]["cells"], "n_recordings": len(suite2p),
             "suite2p": {"r": r.tolist(), "mean_r": float(np.nanmean(r)), "sd_r": float(np.nanstd(r, ddof=1)),
                         "per_recording": dict(suite2p)},
             "ours": {"r": ours.tolist(), "mean_r": float(ours.mean()), "sd_r": float(ours.std(ddof=1))},
             "stats": {"Suite2p": added_test(ours[ok], r[ok], "higher")}}
    family = amend(stats, entry["stats"]["Suite2p"], "Suite2p", "Ring" if "Ring" in stats else None, "D3 cai1")
    record["holm"] = {"family": family["family"]}
    return record, entry, family


# ---- reports ----

# the settings chosen, as the paper's table of them gives them (tab:cv-settings)
CV_LABELS = (("ours", "ASTRID (kappa_d, kappa_np)"), ("TUnCaT", "TUnCaT (alpha)"), ("FISSA", "FISSA (alpha)"),
             ("CNMF", "CNMF (n_b, p)"))
SETTING_FORMS = {"ours": r"ks([0-9.]+)_kb([0-9.]+)", "CNMF": r"gnb([0-9]+)_p([0-9]+)"}


def setting_label(method: str, s: str) -> str | None:
    """A chosen setting as the paper's table prints it: "ks16_kb0.03" -> "16, 0.03", "a=0.2" -> "0.2",
    "gnb3_p2" -> "3, 2"; None for a setting of another form (CNMF's fixed one)."""
    m = re.fullmatch(SETTING_FORMS.get(method, r"a=([0-9.]+)"), s)
    return None if m is None else ", ".join(m.groups())


def settings_lines(chosen: dict[str, list[str]], unit: str) -> list[str]:
    """The settings the cross-validation chose, each with the number of held-out units for which it was
    chosen, the most frequent first (a method at a fixed setting is left out)."""
    lines = ["", f"Settings chosen by the cross-validation, each with the number of held-out {unit} for which it "
             "was chosen (the threshold of transient detection, chosen the same way, is not listed):", "",
             "| method | settings |", "|---|---|"]
    for m, label in CV_LABELS:
        labels = [setting_label(m, s) for s in chosen.get(m, [])]
        if not labels or None in labels:
            continue
        times = Counter(labels)
        order = sorted(times, key=lambda x: (-times[x], [float(y) for y in str(x).split(", ")]))
        lines.append(f"| {label} | " + "; ".join(f"{x} ({times[x]})" for x in order) + " |")
    return lines


def _p(s: dict[str, Any]) -> str:
    return f"{s['p']:.3g} | {s.get('p_holm', float('nan')):.3g}"


def public_markdown(dataset: str, record: dict[str, Any], s2p: dict[str, Any] | None,
                    folds: str = "video") -> str:
    rows = {m: e for m, e in record.items() if m != "_meta"}
    videos = rows["ours"]["videos"] if "ours" in rows else next(iter(rows.values()))["videos"]
    how = ("leave-one-video-out cross-validation" if folds == "video" else
           "cross-validation by recording, the sub-videos of a recording held out together")
    lines = [f"# {D.PAPER_NAMES[dataset]} ({dataset}): F1 by {how} ({len(videos)} videos)", "",
             "| method | mean F1 | SD | min | max | settings chosen (video: setting @ threshold) |",
             "|---|---|---|---|---|---|"]
    entries = dict(rows)
    if s2p is not None:
        entries["suite2p"] = {"videos": videos, "setting": ["suite2p"] * len(videos), **s2p["suite2p"]}
    for m, e in entries.items():
        f = np.asarray(e["f1"])
        chosen = "; ".join(f"{v}: {s} @ {int(t)}" for v, s, t in zip(e["videos"], e["setting"], e["threshold"],
                                                                      strict=True))
        lines.append(f"| {m} | {f.mean():.4f} | {f.std(ddof=1):.4f} | {f.min():.4f} | {f.max():.4f} | {chosen} |")
    lines += settings_lines({m: e["setting"] for m, e in rows.items()}, "videos")
    if CNMF_FIXED in rows:
        lines += ["", f"{CNMF_FIXED}: CNMF at the published setting (gnb 1, p 1), a reference, not tested."]
    tests = dict(record["_meta"]["stats"])
    if s2p is not None:
        tests["Suite2p"] = s2p["stats"]["Suite2p"]
    if tests:
        lines += ["", "ASTRID minus each method, paired over videos (two-sided Wilcoxon signed-rank; Holm over "
                  f"{len(tests)}):", "", "| method | mean diff | videos ASTRID better | p | p (Holm) |",
                  "|---|---|---|---|---|"]
        for m, s in tests.items():
            lines.append(f"| {m} | {s['mean_diff']:+.4f} | {s['wins']}/{len(videos)} | {_p(s)} |")
    return "\n".join(lines) + "\n"


def cai1_markdown(record: dict[str, Any], s2p: dict[str, Any] | None) -> str:
    res = record["results"]
    cells = next(iter(res.values()))["cells"]
    lines = [f"# D3 (cai-1): correlation with the spike-predicted calcium, leave-one-cell-out CV ({len(cells)} "
             "cells)", ""]
    if "baseline" in record:
        lines += [f"Equal baseline: {record['baseline']}.", ""]
    lines += ["| method | mean r | SD | min | max | recordings |", "|---|---|---|---|---|---|"]
    entries = {m: (np.array(e["r"]), e["n_recordings"]) for m, e in res.items()}
    if s2p is not None:
        entries["suite2p"] = (np.array(s2p["suite2p"]["r"]), s2p["n_recordings"])
    for m, (r, n) in entries.items():
        lines.append(f"| {m} | {np.nanmean(r):.4f} | {np.nanstd(r, ddof=1):.4f} | {np.nanmin(r):.4f} "
                     f"| {np.nanmax(r):.4f} | {n} |")
    lines += settings_lines({m: e["chosen"] for m, e in res.items()}, "cells")
    if CNMF_FIXED in res:
        lines += ["", f"{CNMF_FIXED}: CNMF at the published setting (gnb 1, p 1), a reference, not tested."]
    tests = dict(record["stats"])
    if s2p is not None:
        tests["Suite2p"] = s2p["stats"]["Suite2p"]
    if tests:
        lines += ["", f"ASTRID minus each method, paired over cells (two-sided Wilcoxon signed-rank; Holm over "
                  f"{len(tests)}):", "", "| method | mean diff | cells ASTRID better | p | p (Holm) |",
                  "|---|---|---|---|---|"]
        for m, s in tests.items():
            lines.append(f"| {m} | {s['mean_diff']:+.4f} | {s['wins']}/{s['n']} | {_p(s)} |")
    return "\n".join(lines) + "\n"


def _write(out: Path, stem: str, record: dict[str, Any], s2p: dict[str, Any] | None, family: Any,
           text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps({"table": record, "suite2p": s2p, "holm": family}, indent=1),
                                      encoding="utf-8")
    (out / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text)


def _callable(spec: str | None) -> Any:
    if not spec:
        return None
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grids", help="every method's F1 grid per video (D2, D4, D5), with an external scorer")
    g.add_argument("--dataset", choices=("ABO", "NAOMi", "1p"), required=True)
    g.add_argument("--archive", type=Path, required=True, help="TUnCaT's figshare archive")
    g.add_argument("--ours", type=Path, nargs="+", help="ASTRID's result roots (<root>/<ds>/<setting>/<video>)")
    g.add_argument("--peers", type=Path, nargs="+", help="the peers' reruns (default: the archive's released)")
    g.add_argument("--suite2p", type=Path, help="Suite2p's traces (<root>/<ds>/<video>/suite2p/<video>.npz)")
    g.add_argument("--scorer", required=True, help="module:function, (dataset, video, archive) -> scorer")
    g.add_argument("--cnmf-fill", help="module:function, (mat, masks) -> (n, T)")
    g.add_argument("--cnmf-grid", type=Path, help="CNMF's six settings (<grid>/gnb<g>_p<p>/<dataset>/...)")
    g.add_argument("--videos", nargs="*")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("public", help="a D2, D4 or D5 table from the grids")
    p.add_argument("--dataset", choices=("ABO", "NAOMi", "1p"), required=True)
    p.add_argument("--grids", type=Path, required=True, help="grid_<method>_<video>.npz files")
    p.add_argument("--videos", nargs="*")
    p.add_argument("--folds", choices=FOLDS, default="video",
                   help="recording (D5 only): the sub-videos of a recording held out together (written as "
                        "cv_1p_by_recording)")
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("cai1", help="the D3 table from the export and the traces")
    c.add_argument("--export", type=Path, required=True)
    c.add_argument("--ours", type=Path, nargs="+")
    c.add_argument("--peers", type=Path)
    c.add_argument("--cnmf-grid", type=Path, help="CNMF's six settings (<grid>/gnb<g>_p<p>/cai1/<name>.npz)")
    c.add_argument("--suite2p", type=Path)
    c.add_argument("--baseline", choices=BASELINES, default="reported",
                   help="equal: every trace but ASTRID's minus its own running baseline under ASTRID's rule before "
                        "the correlation (written as cv_cai1_equal_baseline)")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "grids":
        grids, _ = public_grids(a.dataset, a.archive, _callable(a.scorer), ours=a.ours, peers=a.peers,
                                suite2p=a.suite2p, videos=a.videos, cnmf_fill=_callable(a.cnmf_fill),
                                workers=a.workers, cnmf_grid=a.cnmf_grid)
        a.out.mkdir(parents=True, exist_ok=True)
        for m, gs in grids.items():
            for v, grid in zip(a.videos or D.VIDEOS[a.dataset], gs, strict=True):
                C.save_grid(a.out / f"grid_{m}_{v}.npz", grid)
        return 0
    if a.cmd == "public":
        videos = list(a.videos or D.VIDEOS[a.dataset])
        grids = {m: [C.load_grid(a.grids / f"grid_{m}_{v}.npz") for v in videos] for m in METHODS
                 if all((a.grids / f"grid_{m}_{v}.npz").exists() for v in videos)}
        if a.folds == "recording" and a.dataset != "1p":
            raise SystemExit("--folds recording is D5's (1p): only its sub-videos share recordings")
        record, s2p, family = public_record(a.dataset, cv_by_method(grids, videos, a.folds))
        stem = f"cv_{a.dataset}" if a.folds == "video" else f"cv_{a.dataset}_by_recording"
        _write(a.out, stem, record, s2p, family, public_markdown(a.dataset, record, s2p, a.folds))
        return 0
    index, scores = cai1_scores(a.export, a.ours, a.peers, a.cnmf_grid, a.baseline)
    s2p_r = cai1_suite2p_scores(a.export, a.suite2p, index, a.baseline) if a.suite2p else None
    record, s2p, family = cai1_record(index, scores, s2p_r)
    if a.baseline == "equal":
        record["baseline"] = EQUAL_BASELINE
    stem = "cv_cai1" if a.baseline == "reported" else "cv_cai1_equal_baseline"
    _write(a.out, stem, record, s2p, family, cai1_markdown(record, s2p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
