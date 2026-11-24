"""Figure 5: D3, the CRCNS cai-1 dataset, where each imaged neuron's spikes were recorded electrically. Every
panel is drawn from the outputs of ``reproduce/``: the table (``cv_cai1.json``, written by ``tables_public
cai1``), the export every method read (one ``<recording>.h5`` per recording and its ``index.json``) and the
traces every method gave (ASTRID's ``result.npz``, the peers', Suite2p's).

- **Scores** (4A; ``draw_scores``): each method's correlation with the calcium predicted from the spikes, per
  cell (the mean over its recordings), at the setting chosen for the cell by leave-one-cell-out
  cross-validation, the mean as a bar, the cells imaged with GCaMP6f as circles and those with GCaMP6s as
  triangles, and above each method the mark of the Holm-corrected p of ASTRID against it over the cells,
  Holm over the five methods.
- **Example** (4B; ``example``, ``draw_image``, ``draw_traces``): one recording of one cell. On the left, the
  mean over the scored frames of the registered, 4 x 4 binned movie the methods were run on, from its 1st to
  its 99.5th percentile, with the cell's hand-drawn ROI in yellow and a scale bar of about 10 um (the pixel
  size of D3, about 0.47 um per binned pixel, is an estimate). On the right, each method's demixed trace over
  the predicted calcium, both standardized to mean 0 and SD 1 over the recording's scored frames (missing
  frames left as gaps), the spikes below, and the recording's correlation on the right.

**The example rule** (Supplementary Section "Example panels"). The cell is the one on which ASTRID's
correlation is furthest above the best other method's, and its recording the one of that cell on which
ASTRID's correlation is furthest above the best other method's on that recording (the first in name order on
ties). Every method is at the setting chosen for the cell. The traces show the ``WINDOW_S`` s within the
electrophysiology where the methods' standardized traces differ most: the largest mean, over the window's
frames, of their variance across the six methods (missing frames count as no difference). Every recording's
correlation is recomputed here from the traces and must give the table's per-cell correlation for every cell
and method, and Ring's and Suite2p's per-recording correlations too.

**The scoring** is the table's. ``tables_public cai1`` writes D3's two (its module docstring): as reported
(``cv_cai1.json``), ASTRID's ΔF/F0 against every other trace with only its mean removed, and with the same
running baseline removed from every trace (``--baseline equal``, ``cv_cai1_equal_baseline.json``, whose
``table["baseline"]`` is the note ``tables_public.EQUAL_BASELINE``), the scoring the paper's Figure 5 shows.
Given the equal table, every trace but ASTRID's is scored and drawn after its own running baseline is removed
under ASTRID's rule (``public_data.minus_baseline`` at the recording's frame rate), so the correlations
recomputed here, the example the rule picks and the traces drawn are those of that scoring.

    python -m astrid_repro.figures cai1 --table <tables>/cv_cai1_equal_baseline.json --export <export>
        --ours <root> [...] --peers <folder> --suite2p <root> --out <folder>
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from astrid_repro import public_data as D
from astrid_repro.figures import common as F
from astrid_repro.ring import cai1_reference_rows
from astrid_repro.tables_public import EQUAL_BASELINE

ORDER = F.PUBLIC_ORDER
NAME = "D3 CRCNS cai-1"
PANELS = ("fig4a", "fig4b")
MARKERS = {"GCaMP6f": "o", "GCaMP6s": "^"}
WINDOW_S = 30.0
UM_PER_PX = 0.47          # an estimate (zoom 50, about 30 um over 256 px, binned 4 x 4): not in the file headers
BAR_UM = 10
PREDICTED = "#A0A0A0"


# ---- the table ----

def read_table(path: str | Path) -> dict[str, Any]:
    """D3's table as ``tables_public cai1`` writes it, checked: every method's cells in ASTRID's order,
    Suite2p's entry with ASTRID's correlations, and the five methods' Holm family. Returns {"record",
    "suite2p", "cells", "values" (method -> r per cell), "tests", "settings" (method -> chosen per cell),
    "baseline" ("reported", or "equal" for a table with ``tables_public``'s equal-baseline note)}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    record, s2p = d["table"], d["suite2p"]
    note = record.get("baseline")
    if note is not None and note != EQUAL_BASELINE:
        raise ValueError(f"D3: the table's baseline note is not tables_public's equal baseline: {note!r}")
    res = record["results"]
    cells = res["ours"]["cells"]
    for m in ORDER[:5]:
        if res[m]["cells"] != cells:
            raise ValueError(f"D3: {m}'s cells are not in ASTRID's order")
    if s2p is None or s2p["cells"] != cells or not np.allclose(s2p["ours"]["r"], res["ours"]["r"], rtol=0,
                                                                atol=1e-12):
        raise ValueError("D3: Suite2p's entry is missing, or its cells or ASTRID's correlations are not the table's")
    if tuple(record.get("holm", {}).get("family", ())) != F.PUBLIC_FAMILY:
        raise ValueError(f"D3: the Holm family is not {F.PUBLIC_FAMILY}")
    tests = {m: record["stats"][F.TEST_KEYS[m]] for m in ORDER[1:5]} | {"suite2p": s2p["stats"]["Suite2p"]}
    F.check_holm(tests, "D3 r")
    return {"record": record, "suite2p": s2p, "cells": cells,
            "values": {m: res[m]["r"] for m in ORDER[:5]} | {"suite2p": s2p["suite2p"]["r"]},
            "tests": tests,
            "settings": {m: res[m]["chosen"] for m in ORDER[:5]} | {"suite2p": ["suite2p"] * len(cells)},
            "baseline": "reported" if note is None else "equal"}


def indicators(export: str | Path) -> dict[str, str]:
    """Each cell's indicator, from the export's index."""
    out: dict[str, str] = {}
    for row in D.read_cai1_index(export):
        if out.setdefault(row["cell"], row["indicator"]) != row["indicator"]:
            raise ValueError(f"D3: cell {row['cell']} has recordings with two indicators")
    return out


def draw_scores(ax: Any, table: dict[str, Any], indicator: dict[str, str]) -> dict[str, Any]:
    """Panel 4A: each method's correlation per cell, with the mark of each comparison (module docstring)."""
    cells = table["cells"]
    counts = {ind: sum(indicator[c] == ind for c in cells) for ind in MARKERS}
    marks = {m: F.stars(float(t["p_holm"])) for m, t in table["tests"].items()}
    shown = F.paired_dots(ax, table["values"], ORDER, marks,
                          "Correlation with spike-predicted\ncalcium (higher is better)",
                          f"{NAME}, {len(cells)} cells", markers=[MARKERS[indicator[c]] for c in cells],
                          marker_labels={m: f"{ind} ({counts[ind]})" for ind, m in MARKERS.items()})
    equal = table["baseline"] == "equal"
    return {"dataset": "D3", "measure": "Pearson r with the spike-predicted calcium, each cell's mean over its "
            "recordings, at the setting chosen by leave-one-cell-out cross-validation"
            + (", the same running baseline removed from every trace" if equal else ""),
            "scoring": EQUAL_BASELINE if equal else "as reported: ASTRID's ΔF/F0, every other trace with only "
            "its mean removed", "cells": cells,
            "indicators": {c: indicator[c] for c in cells}, **shown,
            "p_holm": {m: float(t["p_holm"]) for m, t in table["tests"].items()},
            "holm_family": list(F.PUBLIC_FAMILY)}


# ---- the example ----

def _z(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    return (x - x[ok].mean()) / x[ok].std()


def _trace(m: str, name: str, setting: str, export: Path, ours: Sequence[str | Path], peers: str | Path | None,
           suite2p: str | Path, baseline: str = "reported", cnmf_grid: str | Path | None = None) -> np.ndarray:
    """One method's trace of one recording at ``setting``, as the table scored it: CNMF over its grid with
    ``cnmf_grid`` (as ``tables_public cai1 --cnmf-grid``), and with ``baseline`` "equal", every trace but
    ASTRID's minus its own running baseline (module docstring)."""
    if m == "suite2p":
        x = D.npz_trace(D.suite2p_path(suite2p, "cai1", name))[0]
    elif m == "ring":                                  # D3's own ring, as the table scored it
        x = cai1_reference_rows(export / f"{name}.h5")["ring"]
    elif m == "CNMF" and cnmf_grid:
        x = np.asarray(D.cai1_cnmf_grid(cnmf_grid, name)[setting], dtype=float)
    else:
        x = np.asarray(D.cai1_traces(m, name, peers, list(ours))[setting], dtype=float)
    if baseline == "equal" and m != "ours":
        return D.minus_baseline(x, D.read_cai1_fps(export, name))
    return x


def example(table: dict[str, Any], *, export: str | Path, ours: Sequence[str | Path], peers: str | Path | None,
            suite2p: str | Path, cnmf_grid: str | Path | None = None) -> dict[str, Any]:
    """The example by the rule of the module docstring, after checking every recording's correlation against
    the table (CNMF over ``cnmf_grid`` when the table's CNMF was cross-validated over it)."""
    export = Path(export)
    cells, values, settings, baseline = table["cells"], table["values"], table["settings"], table["baseline"]
    per_recording = {"suite2p": table["suite2p"]["suite2p"]["per_recording"],
                     "ring": table["record"]["results"]["ring"]["per_recording"]}
    index = D.read_cai1_index(export)
    per: dict[str, dict[str, dict[str, float]]] = {}
    worst = 0.0
    for j, cell in enumerate(cells):
        rows = [row for row in index if row["cell"] == cell]
        for row in rows:
            pred, ok = D.read_cai1_scored(export, row["name"])
            for m in ORDER:
                x = _trace(m, row["name"], settings[m][j], export, ours, peers, suite2p, baseline, cnmf_grid)
                r = D.corr(x, pred, ok)
                per.setdefault(cell, {}).setdefault(m, {})[row["rec"]] = r
            for m, table_r in per_recording.items():
                worst = max(worst, abs(per[cell][m][row["rec"]] - table_r[row["name"]]))
        for m in ORDER:
            worst = max(worst, abs(float(np.nanmean(list(per[cell][m].values()))) - float(values[m][j])))
    if not worst <= 1e-9:
        raise ValueError(f"D3: the recomputed correlations differ from the table by {worst:g}")
    r = {m: np.asarray(values[m], dtype=float) for m in ORDER}
    margin = r["ours"] - np.max([r[m] for m in ORDER[1:]], axis=0)
    j = int(np.argmax(margin))
    cell = cells[j]
    recs = per[cell]
    rec_margin = {rec: recs["ours"][rec] - max(recs[m][rec] for m in ORDER[1:]) for rec in recs["ours"]}
    rec = max(sorted(rec_margin), key=lambda k: rec_margin[k])
    name = next(row["name"] for row in index if row["cell"] == cell and row["rec"] == rec)
    with h5py.File(export / f"{name}.h5", "r") as f:
        pred = np.asarray(f["predicted"], dtype=float)
        ok = np.asarray(f["valid"], dtype=bool)
        present = np.asarray(f["frame_present"], dtype=bool)
        t = np.asarray(f["t_frame"], dtype=float)
        spikes = np.asarray(f["spike_times"], dtype=float)
        roi = np.asarray(f["roi"], dtype=bool)
        span = (float(f.attrs["ephys_start_s"]), float(f.attrs["ephys_end_s"]))
        indicator = str(f.attrs["indicator"])
        fps = float(f.attrs["fps"])
        mov = f["mov"]
        acc = np.zeros(mov.shape[1:])
        for s in range(0, mov.shape[0], 2000):                 # the mean over the scored frames
            b = np.asarray(mov[s:s + 2000], dtype=np.float64)
            acc += b[ok[s:s + b.shape[0]]].sum(axis=0)
        image = acc / ok.sum()
    z = {m: _z(_trace(m, name, settings[m][j], export, ours, peers, suite2p, baseline, cnmf_grid), ok)
         for m in ORDER}
    spread = np.var(np.array([z[m] for m in ORDER]), axis=0)
    w = int(round(WINDOW_S * fps))
    inside = (t >= span[0]) & (t <= span[1])
    score = np.convolve(np.where(ok, spread, 0.0), np.ones(w), "valid") / w
    allowed = np.convolve(inside.astype(float), np.ones(w), "valid") == w
    if not allowed.any():
        raise ValueError(f"{name}: no {WINDOW_S:g}-s window within the electrophysiology")
    start = int(np.argmax(np.where(allowed, score, -np.inf)))
    return {"cell": cell, "rec": rec, "name": name, "indicator": indicator, "fps": fps, "t": t, "ok": ok,
            "present": present, "pred": _z(pred, ok), "z": z, "spikes": spikes, "roi": roi, "image": image,
            "window": (start, start + w), "r": {m: recs[m][rec] for m in ORDER},
            "cell_r": {m: float(r[m][j]) for m in ORDER}, "margin": float(margin[j]), "rec_margin": rec_margin,
            "settings": {m: settings[m][j] for m in ORDER},
            "best_other": max(ORDER[1:], key=lambda m: r[m][j]), "baseline": baseline}


def draw_image(ax: Any, ex: dict[str, Any]) -> dict[str, Any]:
    """The left part of panel 4B (module docstring)."""
    img, roi = ex["image"], ex["roi"]
    lo, hi = np.percentile(img, [1, 99.5])
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax.contour(roi.astype(float), levels=[0.5], colors=[F.EXAMPLE_COLOR], linewidths=0.8)
    h, w = img.shape
    bar_px = BAR_UM / UM_PER_PX
    y_bar = h + 0.04 * h                                       # below the image
    ax.plot([w - 0.5 - bar_px, w - 0.5], [y_bar, y_bar], color="black", lw=1.2, solid_capstyle="butt",
            clip_on=False)
    ax.text(w - 0.5 - bar_px / 2, y_bar + 0.02 * h, f"{BAR_UM} µm", color="black", ha="center", va="top",
            clip_on=False)
    ax.set_ylim(h + 0.16 * h, -0.5)
    ax.set_axis_off()
    return {"recording": ex["name"], "indicator": ex["indicator"], "scale bar (um)": BAR_UM,
            "um per binned pixel (estimate)": UM_PER_PX, "gray scale": [float(lo), float(hi)]}


def draw_traces(fig: Any, ex: dict[str, Any]) -> dict[str, Any]:
    """The right part of panel 4B (module docstring)."""
    from matplotlib.lines import Line2D

    axes = fig.subplots(len(ORDER) + 1, 1, sharex=True, gridspec_kw={"height_ratios": [1] * len(ORDER) + [0.35]})
    s, e = ex["window"]
    t = ex["t"][s:e] - ex["t"][s]
    hide = ~ex["present"][s:e]                                 # missing frames are left as gaps
    pred = np.where(hide, np.nan, ex["pred"][s:e])
    traces = {m: np.where(hide, np.nan, ex["z"][m][s:e]) for m in ORDER}
    lo = min(float(np.nanmin(v)) for v in (pred, *traces.values()))
    hi = max(float(np.nanmax(v)) for v in (pred, *traces.values()))
    rows = []
    for ax, m in zip(axes[:-1], ORDER, strict=True):
        ax.plot(t, pred, color=PREDICTED, lw=0.8, zorder=1)
        ax.plot(t, traces[m], color=F.color(m), lw=0.6, zorder=2)
        ax.set_ylim(lo, hi)
        F.trace_row_style(ax, F.label(m), True, False, None if m == "ours" else F.color(m))
        F.score_label(ax, f"{ex['r'][m]:.2f}")
        rows.append({"method": m, "setting": ex["settings"][m], "r_recording": float(ex["r"][m]),
                     "r_cell": ex["cell_r"][m]})
    spikes = ex["spikes"][(ex["spikes"] >= ex["t"][s]) & (ex["spikes"] < ex["t"][e - 1])] - ex["t"][s]
    rast = axes[-1]
    rast.vlines(spikes, 0, 1, color="black", lw=0.6)
    rast.set_ylim(0, 1)
    rast.set_yticks([])
    rast.spines[["left", "top", "right"]].set_visible(False)
    rast.set_ylabel("Spikes", rotation=0, ha="right", va="center", labelpad=2)
    rast.set_xlim(0, WINDOW_S)
    rast.set_xticks(np.arange(0, WINDOW_S + 1, 10))
    rast.set_xlabel("Time (s)")
    F.column_footer(axes[-2], "r", style="italic")
    key = [Line2D([], [], color=PREDICTED, lw=1.0, label="Spike-predicted calcium"),
           Line2D([], [], color="#606060", lw=1.0, label="Demixed trace, in the method's color")]
    fig.legend(handles=key, loc="outside lower center", ncols=1, frameon=False)
    return {"dataset": "D3", "cell": ex["cell"], "recording": ex["name"], "indicator": ex["indicator"],
            "best other method": ex["best_other"], "ASTRID's r above the best other method's (cell)": ex["margin"],
            "ASTRID's r above the best other method's (each recording of the cell)": ex["rec_margin"],
            "window (frames)": list(ex["window"]),
            "window (s)": [float(ex["t"][s]), float(ex["t"][e - 1])], "spikes shown": int(spikes.size),
            "traces": "as the table scored them" + (", every trace but ASTRID's minus its own running baseline "
                                                     "(public_data.minus_baseline)" if ex["baseline"] == "equal"
                                                     else "") + ", standardized over the scored frames",
            "rows": rows, "rule": "astrid_repro.figures.cai1 (module docstring)"}


# ---- the panels ----

def panels(table_path: str | Path, *, export: str | Path, ours: Sequence[str | Path], peers: str | Path | None,
           suite2p: str | Path, cnmf_grid: str | Path | None = None) -> list[F.Panel]:
    """Figure 5: (file stem, figure, record) for the scores, the example's image and the example's traces;
    ``cnmf_grid`` as given to ``tables_public cai1``."""
    import matplotlib.pyplot as plt

    table = read_table(table_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    out = [(f"{PANELS[0]}_D3_scores", fig, draw_scores(ax, table, indicators(export)))]
    ex = example(table, export=export, ours=ours, peers=peers, suite2p=suite2p, cnmf_grid=cnmf_grid)
    fig, ax = plt.subplots(figsize=(3.2, 3.4), layout="constrained")
    fig.supxlabel(NAME)
    out.append((f"{PANELS[1]}_D3_image", fig, draw_image(ax, ex)))
    fig = plt.figure(figsize=(7.0, 5.6), layout="constrained")
    out.append((f"{PANELS[1]}_D3_traces", fig, draw_traces(fig, ex)))
    return out
