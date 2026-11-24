"""The statistics and tables of D1, D6, D7 and the ablations, from the per-case scores (Methods, "Statistics";
Supplementary, "Injection and leakage tests", "Two-color test on D6").

Every table follows one rule. A unit's score is the median of a measure over its cases: on D1 and D6 the unit is the
imaging location (all its recordings together), on D7 the recording. Each column gives the mean ± SD over the units
and, for each other method, ASTRID minus that method, the units where ASTRID is better, and the p of a two-sided
paired Wilcoxon signed-rank test over the units, Holm-corrected over the column's comparisons
(``metrics.wilcoxon_p`` with SciPy's default method, in every table; ``metrics.holm``). The raw
ROI average is a reference row without a test. Methods are named as in the score tables: ``demix`` (ASTRID),
``tuncat``, ``tuncat_astro`` (TUnCaT + terr.), ``fissa``, ``cnmf``, ``cnmf_astro`` (CNMF + terr.), ``ring``,
``suite2p`` and ``roi`` (the raw ROI average).

**D1 and D7** add their signals at six amplitudes (``LEVELS``, ``cases.SCALES``). A unit's score at a level is its
median over its cases at that level; the primary test of a measure is over its curve, each unit's mean of its six
level scores, against the seven comparators (Holm over the seven), and each level is tested the same way (Holm over
the seven within the level). Each level also carries its difficulty ratio, the median and quartiles over its cases
of the signal-to-background ratio (``sbr``, injection) or the astrocyte-to-ROI ratio (``arr``, leakage). The
measures (``grid_dataset``): primary, the error below 0.5 Hz (``err_05hz``) and the leakage (``leak``);
secondary, the gain's distance from 1 (``|gain - 1|``, "gain deviation"), the kept correlation (higher is better)
and the error at astrocyte and at control sites; and per level the cases with a gain below ``LOW_GAIN``.

Commands (from ``reproduce/``: ``python -m astrid_repro.stats <command> --help``):
- ``d1``: D1's injection and leakage tests (``d1``). Input: the per-case scores pooled over D1's recordings
  (``d1 pool``, of one scoring: the paper's tables use ``--baseline without_added``),
  ``injection_scores[_<suffix>].csv`` (``location``, ``recording``, ``case``, ``site``, ``scale``, ``method``,
  ``err_05hz``, ``gain``, ``sbr``, ...) and ``leakage_scores[_<suffix>].csv`` (``location``, ``recording``,
  ``target``, ``scale``, ``method``, ``leak``, ``kept_corr``, ``arr``, ...), every method's rows. A location's
  name starts with its mouse's, before the first underscore. Output: ``d1.json`` (``primary`` and ``secondary``,
  every column per level and per curve) and ``d1.md``.
- ``ablations``: D1's ablations (``ablations``), full ASTRID against variants I, II and III, each from its own
  pooled scores (ASTRID's rows only; every variant must have exactly the full model's cases): the error and the
  leakage per level and per curve, the secondary measures per level, Holm over the three variants of each column.
  Output: ``ablation_stats.json`` and ``.md``.
- ``d6-scores``: D6's per-case scores of one recording from the traces (``d6_scores``) and the kept transient
  size of each case (``d6_sizes``); ``d6``: D6's table from every recording's scores (``d6_tables``), with the
  kept transient size (``d6_size_table``) where every recording has it. Both take ``--baseline``: ``reported``,
  every trace as the method gives it, or ``equal`` (the paper's table), every trace but ASTRID's and the ring
  subtraction's minus its own running baseline under ASTRID's rule (``D6_EQUAL_BASELINE``).
- ``d7``: D7's tests from every recording's per-case scores of one scoring (``--baseline``; the paper's tables
  use ``without_added``), with D7's failure rule (``d7_tables``), as D1's, the recording being the unit, with the
  imaging locations apart and the failures. Output: ``d7.json`` and ``d7.md``.
- ``d7-split``: D7's error split (Supplementary Equation S24, ``d7_split_table``) from every recording's traces:
  per case and method the gain and residual parts of the squared error below 0.5 Hz, and the x3 cases with a
  gain below 0.5. Output: ``d7_split_cases.csv``, ``d7_split.json`` and ``.md``.
- ``nesting``: the tests' units nested in mice (Supplementary Table S5, ``nesting``) from the records of ``d1``,
  ``d6`` and ``d7`` (D1's and D7's error and leakage curves, D6's primary column): per mouse on every dataset; on
  D1 also a mixed model (statsmodels, the extra ``nesting``) and a hierarchical bootstrap. Output:
  ``nesting.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from astrid_repro.cases import BASELINES, SCORE_DIRS
from astrid_repro.metrics import holm, leakage, lowpass_fourier, usable, wilcoxon_p
from astrid_repro.public_data import minus_baseline

OURS = "demix"                         # ASTRID's rows in the score tables
REFERENCE = ("roi",)                   # the raw ROI average: a reference row, no test
NAMES = {"demix": "ASTRID", "tuncat": "TUnCaT", "tuncat_astro": "TUnCaT + terr.", "fissa": "FISSA",
         "cnmf": "CNMF", "cnmf_astro": "CNMF + terr.", "ring": "Ring", "suite2p": "Suite2p", "roi": "raw ROI"}
COMPARED = ("tuncat", "tuncat_astro", "fissa", "cnmf", "cnmf_astro", "ring", "suite2p")   # Holm over these
METHODS = (OURS, *COMPARED, *REFERENCE)
LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)   # the six amplitudes of D1's and D7's tests (cases.SCALES)
LOW_GAIN = 0.5                         # cases with a gain below this are counted


def fmt_p(p: float) -> str:
    """Below 0.001 as such, three decimals below 0.1, then two (1.0 at the top)."""
    if p < 0.001:
        return "p < 0.001"
    if p >= 0.995:
        return "p 1.0"
    return f"p {p:.3f}" if p < 0.1 else f"p {p:.2f}"


def fmt_p4(p: float) -> str:
    """Four decimals (D6 and D7)."""
    return "p < 0.0001" if p < 0.0001 else f"p {p:.4f}"


def read_scores(folder: Path, suffix: str = "") -> dict[str, pd.DataFrame]:
    """A folder's ``injection_scores[_<suffix>].csv`` and ``leakage_scores[_<suffix>].csv``."""
    tail = f"_{suffix}" if suffix else ""
    return {k: pd.read_csv(folder / f"{k}_scores{tail}.csv") for k in ("injection", "leakage")}


def write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------- D1 and D7 on the grid

def cell2(x: pd.Series) -> str:
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def diff_cell(s: dict[str, Any], nd: int, fmt: Any = fmt_p) -> str:
    return f"{s['mean_diff']:+.{nd}f} ({s['wins']}/{s['n']}; {fmt(s['p_holm'])})".replace("-", "−")


def grid_column(units: pd.DataFrame, meta: dict[str, Any], better: str = "lower") -> dict[str, Any]:
    """One column: every unit's score per method, the mean and SD over the units, and the tests (``compare7``,
    Holm over the seven)."""
    if units[list(METHODS)].isna().any().any():
        raise SystemExit(f"{meta.get('measure')} {meta.get('scale')}: a unit lacks a method's score")
    return {**meta, "better": better, "units": units.index.tolist(),
            "scores": {m: units[m].tolist() for m in METHODS},
            "mean": {m: float(units[m].mean()) for m in METHODS},
            "sd": {m: float(units[m].std(ddof=1)) for m in METHODS},
            "stats": compare7(units, better == "lower")}


def ratio_summary(t: pd.DataFrame, col: str) -> dict[str, float]:
    """A level's difficulty ratio (``sbr`` or ``arr``, one per case): its median and quartiles over the cases."""
    x = t[t.method == OURS][col].to_numpy(float)
    return {"cases": int(x.size), "median": float(np.median(x)), "q25": float(np.percentile(x, 25)),
            "q75": float(np.percentile(x, 75))}


def grid_curves(t: pd.DataFrame, unit: str, value: str, name: str, ratio: str, info: pd.DataFrame,
                better: str = "lower") -> dict[str, Any]:
    """The columns of one measure: every level (each unit's median over its cases at that level, with the level's
    difficulty ratio ``ratio``), then the curve (each unit's mean of its six level scores). ``info``: per unit the
    fields every column carries (D1 the mouse and the recordings, D7 the location)."""
    out: dict[str, Any] = {}
    levels = sorted(t.scale.unique())
    if tuple(levels) != LEVELS:
        raise SystemExit(f"{name}: levels {levels}, not the grid's {LEVELS}")
    per = {}
    for s in levels:
        g = t[t.scale == s]
        u = g.groupby([unit, "method"])[value].median().unstack()[list(METHODS)]
        per[s] = u
        meta = {"measure": f"{value}, each unit's median over its cases", "scale": s,
                **info.reindex(u.index).to_dict(orient="list"),
                "cases": g[g.method == OURS].groupby(unit).size().reindex(u.index).tolist(),
                "ratio": {"name": ratio, **ratio_summary(g, ratio)}}
        out[f"{name} x{s:g}"] = grid_column(u, meta, better)
    idx = per[levels[0]].index
    if any(not per[s].index.equals(idx) for s in levels):
        raise SystemExit(f"{name}: the levels have different units")
    curve = sum(per[s] for s in levels) / len(levels)
    meta = {"measure": f"{value}: each unit's mean of its six level scores", "scale": None,
            **info.reindex(curve.index).to_dict(orient="list")}
    out[f"{name} curve"] = grid_column(curve, meta, better)
    return out


def grid_dataset(inj: pd.DataFrame, leak: pd.DataFrame, unit: str,
                 info: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Every column of D1 or D7 (module docstring): the primary measures (the error below 0.5 Hz, the leakage)
    per level and curve, the secondary ones (the gain's distance from 1, the kept correlation, the error at
    astrocyte and at control sites) the same way, and per level the cases with a gain below ``LOW_GAIN``."""
    inj = inj.assign(gain_dev=(inj.gain - 1).abs())
    rec: dict[str, Any] = {}
    rec.update(grid_curves(inj, unit, "err_05hz", "error", "sbr", info["injection"]))
    rec.update(grid_curves(leak, unit, "leak", "leakage", "arr", info["leakage"]))
    sec: dict[str, Any] = {}
    sec.update(grid_curves(inj, unit, "gain_dev", "gain deviation", "sbr", info["injection"]))
    sec.update(grid_curves(leak, unit, "kept_corr", "kept correlation", "arr", info["leakage"], better="higher"))
    for site, label in (("astro", "astrocyte sites"), ("control", "control sites")):
        g = inj[inj.site == site]
        if g.empty:
            continue
        sub = info["injection"].reindex(sorted(set(g[unit])))
        sec.update(grid_curves(g, unit, "err_05hz", f"error, {label}", "sbr", sub))
    sec["low gain"] = {f"x{s:g}": {m: int(((g.method == m) & (g.gain < LOW_GAIN)).sum()) for m in METHODS}
                       | {"cases": int((g.method == OURS).sum())} for s, g in inj.groupby("scale")}
    return {"primary": rec, "secondary": sec}


def grid_markdown(record: dict[str, Any], title: str) -> list[str]:
    """The record's columns, one row each: the units, the difficulty ratio, and every method's mean ± SD with,
    for the seven, the units where ASTRID is better and the Holm p."""
    lines = [f"# {title}", "",
             "Each unit's score at a level is its median over its cases; the curve is the mean of its six level "
             "scores. Mean ± SD over the units; for each other method, the units where ASTRID is better and the "
             "paired two-sided Wilcoxon p, Holm-corrected over the seven within each column.", ""]
    for group in ("primary", "secondary"):
        cols = {k: v for k, v in record[group].items() if isinstance(v, dict) and "stats" in v}
        lines += [f"## {group.capitalize()}", "", "| column | units | ratio, median (IQR) | "
                  + " | ".join(NAMES[m] for m in METHODS) + " |", "|---|---|---|" + "---|" * len(METHODS)]
        for col, c in cols.items():
            r = c.get("ratio")
            rr = f"{r['name']} {r['median']:.2f} ({r['q25']:.2f}-{r['q75']:.2f})" if r else ""
            cells = []
            for m in METHODS:
                v = f"{c['mean'][m]:.3f} ± {c['sd'][m]:.3f}"
                if m in COMPARED:
                    s = c["stats"][m]
                    v += f" ({s['wins']}/{s['n']}; {fmt_p4(s['p_holm'])})"
                cells.append(v)
            lines.append(f"| {col} | {len(c['units'])} | {rr} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"Cases with a gain below {LOW_GAIN}, per level:", ""]
    for lv, row in record["secondary"]["low gain"].items():
        lines.append(f"- {lv} ({row['cases']} cases): "
                     + ", ".join(f"{NAMES[m]} {row[m]}" for m in METHODS if m not in REFERENCE))
    lines.append("")
    return lines


def d1_info(t: pd.DataFrame) -> pd.DataFrame:
    """Per location: its mouse (the start of its name, before the first underscore) and its recordings."""
    rec = t[t.method == OURS].groupby("location").recording.nunique()
    return pd.DataFrame({"mouse": rec.index.str.split("_").str[0], "recordings": rec.to_numpy()}, index=rec.index)


def d1(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """D1's record from the pooled per-case scores of every method (``injection`` and ``leakage``, one scoring):
    the imaging location is the unit (``grid_dataset``)."""
    inj, leak = tables["injection"], tables["leakage"]
    return {"unit": "location", **grid_dataset(inj, leak, "location",
                                               {"injection": d1_info(inj), "leakage": d1_info(leak)})}


def d1_markdown(record: dict[str, Any]) -> str:
    lines = grid_markdown(record, "D1: the injection and leakage tests over the imaging locations")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ ablations

VARIANTS = ("I", "II", "III")
ABLATION_NAMES = {"full": "Full (ASTRID)", "I": "I, no astrocyte components",
                  "II": "II, no penalty on fast fluctuations", "III": "III, no evidence map"}
MEASURE = {"gain_dev": "|gain - 1|"}
ABLATION_KEYS = {"injection": ["location", "recording", "case", "scale"],
                 "leakage": ["location", "recording", "target", "scale"]}


def ablation_columns() -> tuple[tuple[str, str, str, float | None, str | None, str, bool], ...]:
    """The ablation table's columns, (name, table, score, level (None: the curve), site (None: all), better,
    primary): the error below 0.5 Hz and the leakage at each level and per curve, primary; at each level the gain's
    distance from 1, the kept correlation and the error at astrocyte and at control sites, secondary."""
    cols: list[tuple[str, str, str, float | None, str | None, str, bool]] = []
    for name, kind, value in (("error", "injection", "err_05hz"), ("leakage", "leakage", "leak")):
        cols += [(f"{name} x{s:g}", kind, value, s, None, "lower", True) for s in LEVELS]
        cols.append((f"{name} curve", kind, value, None, None, "lower", True))
    for s in LEVELS:
        cols += [(f"gain deviation x{s:g}", "injection", "gain_dev", s, None, "lower", False),
                 (f"kept correlation x{s:g}", "leakage", "kept_corr", s, None, "higher", False),
                 (f"error x{s:g}, astrocyte sites", "injection", "err_05hz", s, "astro", "lower", False),
                 (f"error x{s:g}, control sites", "injection", "err_05hz", s, "control", "lower", False)]
    return tuple(cols)


ABLATION_COLUMNS = ablation_columns()


def ours_only(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """ASTRID's rows of a pooled run, with |gain - 1| added to the injections."""
    t = {k: v[v.method == OURS].copy() for k, v in tables.items()}
    t["injection"]["gain_dev"] = (t["injection"].gain - 1).abs()
    return t


def ablation_by_location(t: pd.DataFrame, value: str, scale: float | None, site: str | None) -> pd.Series:
    """Each location's median over its cases at ``scale``; for a curve (None), the mean of its six levels'."""
    if scale is None:
        return sum(ablation_by_location(t, value, s, site) for s in LEVELS) / len(LEVELS)
    g = t[(t.scale == scale) & ((t.site == site) if site else True)]
    return g.groupby("location")[value].median()


def ablation_compare(loc: pd.DataFrame, better: str) -> dict[str, dict[str, float]]:
    """Full against each variant over the locations: Full minus the variant, the locations where Full is
    better, the Wilcoxon p (``metrics.wilcoxon_p``, SciPy's default method: zero differences dropped) and
    Holm's over the variants."""
    stats: dict[str, dict[str, float]] = {}
    raw: dict[str, float] = {}
    for v in VARIANTS:
        d = (loc["full"] - loc[v]).to_numpy()
        p = wilcoxon_p(d)
        raw[v] = p
        wins = int((d < 0).sum()) if better == "lower" else int((d > 0).sum())
        stats[v] = {"mean_diff": float(d.mean()), "wins": wins, "n": int(d.size),
                    "n_nonzero": int((d != 0).sum()), "p": p}
    for v, ph in holm(raw).items():
        stats[v]["p_holm"] = ph
    return stats


def ablations(full: Mapping[str, pd.DataFrame], variants: Mapping[str, Mapping[str, pd.DataFrame]]) -> dict[str, Any]:
    """The ablation table: ``full`` and each variant's pooled scores (``read_scores``; ASTRID's rows used, the same
    under both scorings of the injection test), every column of ``ABLATION_COLUMNS`` over the locations, and the
    injections with a gain below ``LOW_GAIN`` at x3 (``_descriptive``) and at every level (``_descriptive_levels``)."""
    if sorted(variants) != sorted(VARIANTS):
        raise ValueError(f"the variants must be exactly {', '.join(VARIANTS)}")
    runs = {"full": ours_only(full)} | {v: ours_only(variants[v]) for v in VARIANTS}
    for kind, key_cols in ABLATION_KEYS.items():               # every variant has exactly Full's cases
        ref = set(map(tuple, runs["full"][kind][key_cols].astype(str).to_numpy()))
        for v in VARIANTS:
            got = set(map(tuple, runs[v][kind][key_cols].astype(str).to_numpy()))
            if got != ref:
                raise SystemExit(f"{v}: its {kind} cases differ from Full's ({len(got ^ ref)} differ)")
    methods = ("full", *VARIANTS)
    record: dict[str, Any] = {}
    for col, kind, value, scale, site, better, primary in ABLATION_COLUMNS:
        loc = pd.DataFrame({m: ablation_by_location(runs[m][kind], value, scale, site) for m in methods})
        if loc.isna().any().any():
            raise SystemExit(f"{col}: a location lacks a variant's score")
        g = runs["full"][kind]
        g = g[((g.scale == scale) if scale is not None else g.scale.isin(LEVELS))
              & ((g.site == site) if site else True)]
        record[col] = {"measure": f"{MEASURE.get(value, value)}, each location's median over its cases"
                                  + (f" (sites: {site})" if site else ""),
                       "scale": scale, "primary": primary, "better": better,
                       "locations": loc.index.tolist(), "mouse": [s.split("_")[0] for s in loc.index],
                       "recordings": g.groupby("location").recording.nunique().reindex(loc.index).tolist(),
                       "cases": g.groupby("location").size().reindex(loc.index).tolist(),
                       "scores": {m: loc[m].tolist() for m in methods},
                       "mean": {m: float(loc[m].mean()) for m in methods},
                       "sd": {m: float(loc[m].std(ddof=1)) for m in methods},
                       "stats": ablation_compare(loc, better)}
    low = {m: int((runs[m]["injection"].query("scale == 3").gain < LOW_GAIN).sum()) for m in methods}
    n3 = int((runs["full"]["injection"].scale == 3).sum())
    record["_descriptive"] = {"measure": "x3 injections with gain below 0.5", "n_cases": n3, "counts": low}
    record["_descriptive_levels"] = {
        f"x{s:g}": {"n_cases": int((runs["full"]["injection"].scale == s).sum()),
                    "counts": {m: int((runs[m]["injection"].query(f"scale == {s}").gain < LOW_GAIN).sum())
                               for m in methods}} for s in LEVELS}
    return record


def ablation_markdown(record: dict[str, Any]) -> str:
    methods = ("full", *VARIANTS)
    lines = ["# D1 ablations: full ASTRID against each variant over the imaging locations", "",
             "Each location's score is the median over its cases at a level; on a curve, the mean of its six level "
             "scores. Mean ± SD over the locations. Gain deviation is |gain - 1|. For each variant: Full minus "
             "it, the locations where Full is better (lower error, leakage and gain deviation; higher kept "
             "correlation), and the p of a two-sided paired Wilcoxon signed-rank test over the locations (SciPy's "
             "default method), Holm-corrected over the three variants of each column.", "",
             "| column | locations | " + " | ".join(ABLATION_NAMES[m] for m in methods) + " |",
             "|---|---|" + "---|" * len(methods)]
    for col, *_ in ABLATION_COLUMNS:
        r = record[col]
        cells = []
        for m in methods:
            x = pd.Series(r["scores"][m])
            cells.append(cell2(x) if m == "full" else f"{cell2(x)}: {diff_cell(r['stats'][m], 2)}")
        lines.append(f"| {col} | {len(r['locations'])} | " + " | ".join(cells) + " |")
    d = record["_descriptive_levels"]
    lines += ["", "Descriptive: injections with gain below 0.5 per level: " + "; ".join(
        f"{lv} (of {v['n_cases']}): " + ", ".join(f"{ABLATION_NAMES[m]} {v['counts'][m]}" for m in methods)
        for lv, v in d.items()) + ".", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- D6

D6_FPS = 30.9
D6_BANDS = {"r_5hz": 5.0, "r_05hz": 0.5}
D6_MOVIES = ("M", "R")                 # M: the two channels summed (one-color); R: the neuronal channel alone
D6_PEER_KEYS = {"tuncat": "tuncat", "tuncat_astro": "tuncat_astro", "fissa": "fissa", "cnmf": "cnmf",
                "cnmf_astro": "cnmf_astro", "roi": "tuncat_raw_roi"}
D6_COLUMNS = (("leakage in territory", "in", "leak_5hz"),          # (name, cases, score); lower is better
              ("leakage in territory, below 0.5 Hz", "in", "leak_05hz"),
              ("leakage outside territory", "out", "leak_5hz"),
              ("leakage, all cases", "all", "leak_5hz"),
              ("r in territory (trace kept)", "in", "r_5hz"))           # higher is better
D6_PRIMARY = "leakage in territory"
D6_BASELINES = ("reported", "equal")
D6_AS_IS = ("demix", "ring")           # F - F0 of each movie already: never given the equal baseline
D6_EQUAL_BASELINE = ("every trace but ASTRID's and the ring subtraction's minus its own running baseline under "
                     "ASTRID's rule (the 10th percentile of 1-s means in a moving 60-s window, interpolated to the "
                     "frames), on each movie and in each case")
D6_SCORE_FILES = {"reported": "scores.csv", "equal": "scores_equal.csv"}
D6_SIZE_FILES = {"reported": "transient_size.csv", "equal": "transient_size_equal.csv"}
# the kept transient size on R (descriptive)
SIZE_BAND_HZ = 5.0                     # the scoring's low-pass
SIZE_PROMINENCE = 4.0                  # times the raw ROI average's noise s.d.
SIZE_DISTANCE_S = 2.0                  # s between peaks
SIZE_RING_MAX = 0.2                    # the ring's amplitude over the raw ROI average's, at most
SIZE_PEAK_HALF = 3                     # frames each side of the peak
SIZE_BASE_S = (2.0, 0.5)               # s before the peak: the amplitude's baseline window
SIZE_MIN_EVENTS = 3                    # neuron-only transients a case needs
RING_FACTOR = 0.7                      # the ring subtraction's factor (astrid.patch.ring_trace)
SIZE_CLASSES = ("in", "out", "all")


def _check_d6_baseline(baseline: str) -> None:
    if baseline not in D6_BASELINES:
        raise ValueError(f"baseline must be one of {D6_BASELINES}, got {baseline!r}")


def recording_arrays(z: Mapping[str, Any], key: str) -> list[np.ndarray]:
    """ASTRID's per-recording arrays saved under ``key``: stacked, or one per recording (``key__r<i>`` with
    their count ``key__n``)."""
    if key in z:
        return list(z[key])
    if f"{key}__n" not in z:
        raise KeyError(key)
    return [np.asarray(z[f"{key}__r{r}"]) for r in range(int(z[f"{key}__n"]))]


def d6_traces(run: Path, movie: str, case: str, baseline: str = "reported",
              fps: float = D6_FPS) -> dict[str, np.ndarray | None]:
    """Every method's trace of one case on one movie, None where it is missing; with ``baseline`` "equal", every
    usable trace but ASTRID's and the ring subtraction's minus its own running baseline under ASTRID's rule
    (``public_data.minus_baseline``)."""
    _check_d6_baseline(baseline)
    out: dict[str, np.ndarray | None] = dict.fromkeys(METHODS)
    p = run / movie / "ours" / f"{case}.npz"
    if p.exists():
        with np.load(p) as z:
            out["demix"] = recording_arrays(z, "delivered")[0]
            out["ring"] = recording_arrays(z, "ring_trace")[0]
    p = run / movie / "peers" / f"{case}.npz"
    if p.exists():
        with np.load(p) as z:
            out.update({m: np.asarray(z[k], dtype=np.float64) for m, k in D6_PEER_KEYS.items() if k in z.files})
    p = run / movie / "suite2p" / f"{case}.npz"
    if p.exists():
        with np.load(p) as z:
            out["suite2p"] = np.asarray(z["suite2p"], dtype=np.float64)
    if baseline == "equal":
        out = {m: minus_baseline(x, fps) if m not in D6_AS_IS and x is not None and usable(x) else x
               for m, x in out.items()}
    return out


def pearson(x: np.ndarray, y: np.ndarray, f: float, fps: float = D6_FPS) -> float:
    """Pearson r of two traces low-passed below ``f`` Hz (NaN if either is then constant)."""
    a, b = lowpass_fourier(x, fps, f), lowpass_fourier(y, fps, f)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def d6_scores(run: Path, fps: float = D6_FPS, baseline: str = "reported") -> pd.DataFrame:
    """One recording's D6 scores, one row per case and method. Each method's trace on the one-color movie M is
    compared with its own trace on the neuron movie R (``d6_traces``, under ``baseline``):
    - ``leak_5hz``, ``leak_05hz``: the leakage of the real astrocyte channel (``metrics.leakage``: the change of
      the trace between the two movies over the channel's signal in the ROI, M's ROI average minus R's),
      below 5 Hz (the primary) and below 0.5 Hz;
    - ``r_5hz``, ``r_05hz``: the Pearson r of the two traces, the check that the neuron trace is kept.
    A trace that is missing, not finite or constant on either movie is a failure, scored leakage 1 (the raw
    ROI's value) and r 0, never left out, and so is a trace that is constant after the low-pass (its r is
    undefined).

    Inputs in ``run`` (one recording): ``run.json`` (``location``, ``recording``); ``M/compare/index.json`` (the
    cases: ``spec``, ``label``, ``territory`` ("in", "between" or "out"), ``territory_share``, ``neighbours``,
    ``atoms``) and per movie X (M, R): ``X/compare/<case>.npz`` (``roi_mean``, the ROI average),
    ``X/ours/<case>.npz`` (ASTRID's ``delivered`` trace and the ring's ``ring_trace``),
    ``X/peers/<case>.npz`` (``astrid_repro.peers``; ``tuncat_raw_roi`` is the raw ROI) and
    ``X/suite2p/<case>.npz`` (``astrid_repro.suite2p_runner`` cases)."""
    _check_d6_baseline(baseline)
    info = json.loads((run / "run.json").read_text())
    index = json.loads((run / "M" / "compare" / "index.json").read_text())
    rows = []
    for c in index:
        tr = {m: d6_traces(run, m, c["spec"], baseline, fps) for m in D6_MOVIES}
        roi = {m: np.load(run / m / "compare" / f"{c['spec']}.npz")["roi_mean"] for m in D6_MOVIES}
        astro = roi["M"] - roi["R"]                                   # the astrocyte channel in the ROI
        for meth in METHODS:
            x, y = tr["M"][meth], tr["R"][meth]
            ok = usable(x) and usable(y) and x is not None and y is not None and x.shape == y.shape
            row = {"location": info["location"], "recording": info["recording"], "case": c["spec"],
                   "label": c["label"], "territory": c["territory"], "territory_share": c["territory_share"],
                   "neighbours": c["neighbours"], "atoms": c["atoms"], "method": meth,
                   "status": "ok" if ok else ("missing" if x is None or y is None else "unusable")}
            for k, f in D6_BANDS.items():
                r = pearson(x, y, f, fps) if ok and x is not None and y is not None else float("nan")
                if not np.isfinite(r):
                    row["status"] = "failure" if row["status"] == "ok" else row["status"]
                row[k] = r if np.isfinite(r) else 0.0                   # a failure scores r = 0
                good = row["status"] == "ok" and x is not None and y is not None
                lk = leakage(x, y, astro, fps, f) if good and x is not None and y is not None else float("nan")
                row["leak" + k[1:]] = lk if np.isfinite(lk) else 1.0    # a failure scores 1
            rows.append(row)
    return pd.DataFrame(rows)


def noise_sd(x: np.ndarray) -> float:
    """A trace's noise per frame: 1.4826 x the median absolute deviation of its frame-to-frame differences over
    sqrt 2."""
    d = np.diff(np.asarray(x, dtype=np.float64))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def amplitude(x: np.ndarray, p: int, fps: float = D6_FPS) -> float:
    """The mean over the peak ± ``SIZE_PEAK_HALF`` frames minus the median over ``SIZE_BASE_S`` before it."""
    b0, b1 = p - round(SIZE_BASE_S[0] * fps), p - round(SIZE_BASE_S[1] * fps)
    return float(np.mean(x[p - SIZE_PEAK_HALF:p + SIZE_PEAK_HALF + 1]) - np.median(x[b0:b1]))


def case_sizes(tr: Mapping[str, np.ndarray | None], roi_mean: np.ndarray, base: np.ndarray | None,
               fps: float = D6_FPS) -> dict[str, float]:
    """The kept transient size of every method in one case on R (``d6_sizes``), with the numbers of peaks and of
    the neuron's own transients (``n_peaks``, ``n_events``); NaN where the case has fewer than
    ``SIZE_MIN_EVENTS`` transients or the method's trace is not usable. ``tr``: the methods' traces on R,
    ``roi_mean``: the ROI's mean trace on R, ``base``: ASTRID's ROI baseline on R (``baseline_roi``)."""
    row: dict[str, float] = {"n_peaks": 0, "n_events": 0, **{f"tsize_{m}": float("nan") for m in METHODS}}
    raw, ring_tr = tr["roi"], tr["ring"]
    if raw is None or ring_tr is None or base is None or not usable(raw):
        return row
    lp = {m: lowpass_fourier(np.asarray(x, dtype=np.float64), fps, SIZE_BAND_HZ) for m, x in tr.items()
          if x is not None and usable(x)}
    ring = lowpass_fourier((np.asarray(roi_mean, dtype=np.float64) - base - ring_tr) / RING_FACTOR, fps,
                           SIZE_BAND_HZ)
    peaks, _ = find_peaks(lp["roi"], prominence=SIZE_PROMINENCE * noise_sd(raw),
                          distance=round(SIZE_DISTANCE_S * fps))
    peaks = [int(p) for p in peaks if p - round(SIZE_BASE_S[0] * fps) >= 0 and p + SIZE_PEAK_HALF < raw.size]
    kept = []
    for p in peaks:
        a = amplitude(lp["roi"], p, fps)
        if a > 0 and amplitude(ring, p, fps) <= SIZE_RING_MAX * a:
            kept.append((p, a))
    row["n_peaks"], row["n_events"] = len(peaks), len(kept)
    if len(kept) >= SIZE_MIN_EVENTS:
        for m, x in lp.items():
            row[f"tsize_{m}"] = float(np.median([amplitude(x, p, fps) / a for p, a in kept]))
    return row


def d6_sizes(run: Path, fps: float = D6_FPS, baseline: str = "reported") -> pd.DataFrame:
    """One recording's kept transient sizes on the neuron movie R, one row per case (descriptive): how large each
    method keeps the neuron's own transients. They are the peaks of the raw ROI average low-passed below 5 Hz,
    with a prominence of at least ``SIZE_PROMINENCE`` times its noise s.d. (``noise_sd``, of the unfiltered
    trace) and at least ``SIZE_DISTANCE_S`` apart, at which the ring's amplitude is at most ``SIZE_RING_MAX``
    times the raw ROI average's (the ring's mean recovered from the ROI average, ASTRID's ROI baseline and the
    ring subtraction, divided by ``RING_FACTOR``). A transient's size in a trace is its ``amplitude`` (after the
    same low-pass) over the raw ROI average's, and a method's size in a case is the median over the case's
    transients. Traces as in ``d6_scores`` (``baseline``); inputs as there, with ASTRID's ``baseline_roi`` (a case
    without it has no sizes)."""
    info = json.loads((run / "run.json").read_text())
    index = json.loads((run / "M" / "compare" / "index.json").read_text())
    rows = []
    for c in index:
        roi_mean = np.load(run / "R" / "compare" / f"{c['spec']}.npz")["roi_mean"]
        p = run / "R" / "ours" / f"{c['spec']}.npz"
        base = None
        if p.exists():
            with np.load(p) as z:
                if "baseline_roi" in z.files or "baseline_roi__n" in z.files:
                    base = recording_arrays(z, "baseline_roi")[0]
        rows.append({"location": info["location"], "recording": info["recording"], "case": c["spec"],
                     "territory": c["territory"],
                     **case_sizes(d6_traces(run, "R", c["spec"], baseline, fps), roi_mean, base, fps)})
    return pd.DataFrame(rows)


def d6_size_table(sizes: list[pd.DataFrame]) -> dict[str, Any]:
    """The kept transient size over every recording (``d6_sizes``): the cases with at least ``SIZE_MIN_EVENTS``
    transients, and per class ("in", "out", "all") and method each location's median over them, mean ± SD over
    the locations."""
    t = pd.concat(sizes, ignore_index=True)
    used = t[t.n_events >= SIZE_MIN_EVENTS]
    out: dict[str, Any] = {"cases": int(len(t)), "used": int(len(used)), "transients": int(used.n_events.sum()),
                           "peaks": int(t.n_peaks.sum())}
    for k in SIZE_CLASSES:
        s = used if k == "all" else used[used.territory == k]
        out[k] = {}
        for m in METHODS:
            loc = s.groupby("location")[f"tsize_{m}"].median()
            out[k][m] = {"mean": float(loc.mean()), "sd": float(loc.std(ddof=1)), "locations": int(loc.notna().sum())}
    return out


def lower_is_better(value: str) -> bool:
    return value.startswith("leak")


def selected(t: pd.DataFrame, cls: str) -> pd.DataFrame:
    """The cases of a territory class ("in", "out", "between"), or all of them ("all")."""
    return t if cls == "all" else t[t.territory == cls]


def d6_by_location(t: pd.DataFrame, cls: str, value: str) -> pd.DataFrame:
    """(location x method): each location's median over its cases of class ``cls``."""
    loc = selected(t, cls).groupby(["location", "method"])[value].median().unstack()
    if loc.isna().any().any():
        raise SystemExit(f"{cls} {value}: a location lacks a method")
    return loc[list(METHODS)]


def compare7(units: pd.DataFrame, lower: bool = True) -> dict[str, dict[str, float]]:
    """ASTRID minus each of the seven comparators over the units, the units where ASTRID is better (lower, or
    higher if not ``lower``: ``wins``), the Wilcoxon p and Holm's over the seven."""
    stats: dict[str, dict[str, float]] = {}
    raw: dict[str, float] = {}
    for m in COMPARED:
        d = (units[OURS] - units[m]).to_numpy()
        raw[m] = wilcoxon_p(d)
        wins = int((d < 0).sum()) if lower else int((d > 0).sum())
        stats[m] = {"mean_diff": float(d.mean()), "wins": wins, "n": int(d.size), "p": raw[m]}
    for m, ph in holm(raw).items():
        stats[m]["p_holm"] = ph
    return stats


def d6_tables(scores: list[pd.DataFrame]) -> dict[str, Any]:
    """D6's table from every recording's scores (``d6_scores``). A location's mouse is the start of its name,
    before the first underscore and any hyphen (two names of one mouse, such as A-M1 and A-M2, count once)."""
    t = pd.concat(scores, ignore_index=True)
    record: dict[str, Any] = {"recordings": len(scores), "primary": D6_PRIMARY, "columns": {}}
    for col, cls, value in D6_COLUMNS:
        loc = d6_by_location(t, cls, value)
        g = selected(t, cls)
        g = g[g.method == OURS]
        record["columns"][col] = {
            "measure": f"{value}, each location's median over its cases ({cls})",
            "cases_class": cls, "score": value, "lower_is_better": lower_is_better(value),
            "primary": col == D6_PRIMARY, "locations": loc.index.tolist(),
            "mouse": [s.split("_")[0].split("-")[0] for s in loc.index],
            "recordings": g.groupby("location").recording.nunique().reindex(loc.index).tolist(),
            "cases": g.groupby("location").size().reindex(loc.index).tolist(),
            "scores": {m: loc[m].tolist() for m in METHODS},
            "mean": {m: float(loc[m].mean()) for m in METHODS},
            "sd": {m: float(loc[m].std(ddof=1)) for m in METHODS},
            "stats": compare7(loc, lower_is_better(value))}
    fails = t[t.status != "ok"].groupby("method").size().reindex(list(METHODS), fill_value=0)
    record["failures"] = {m: int(fails[m]) for m in METHODS}
    record["cases_by_class"] = t[t.method == OURS].territory.value_counts().to_dict()
    return record


def cell3(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def d6_markdown(record: dict[str, Any], name: str = "d6") -> str:
    cols = record["columns"]
    head = " | ".join(f"{c} ({len(cols[c]['locations'])} locations)" for c, *_ in D6_COLUMNS)
    lines = [f"# {name.upper()}: how much of the real astrocyte channel reaches each method's neuron trace", "",
             "Each method runs on the one-color movie M (astrocyte channel + neuron channel) and on the neuron "
             "movie R alone. Leakage: the change of the method's trace between the two over the astrocyte "
             "channel's signal in the ROI, low-passed below 5 Hz or 0.5 Hz (0 is ideal, the raw ROI scores 1, "
             "lower is better). r: the correlation of the two traces, the check that the neuron trace is kept "
             "(higher is better). Each location's score is the median over its cases of the column's group, all "
             "its recordings together; mean ± SD over the locations. For each comparator: ASTRID minus it, the "
             "locations where ASTRID is better, and the p of a paired two-sided Wilcoxon signed-rank test over "
             "the locations, Holm-corrected over the seven comparisons of each column. Primary: "
             f"{D6_PRIMARY}. A failed trace scores leakage 1 and r 0.", "",
             f"| | {head} |", "|---|" + "---|" * len(D6_COLUMNS)]
    for m in (OURS, *COMPARED):
        cells = []
        for col, *_ in D6_COLUMNS:
            x = pd.Series(cols[col]["scores"][m])
            cells.append(cell3(x) if m == OURS else f"{cell3(x)}: {diff_cell(cols[col]['stats'][m], 3, fmt_p4)}")
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines.append(f"| {NAMES['roi']} (reference, no test) | "
                 + " | ".join(cell3(pd.Series(cols[c]["scores"]["roi"])) for c, *_ in D6_COLUMNS) + " |")
    lines += ["", "Unrounded: ASTRID minus each comparator over the locations.", "",
              "| column | comparator | mean diff | locations ASTRID better | p | p (Holm) |",
              "|---|---|---|---|---|---|"]
    for col, *_ in D6_COLUMNS:
        for m, s in cols[col]["stats"].items():
            lines.append(f"| {col} | {NAMES[m]} | {s['mean_diff']:+.4f} | {s['wins']}/{s['n']} "
                         f"| {s['p']:.4f} | {s['p_holm']:.4f} |")
    mice = {c: len(set(cols[c]["mouse"])) for c, *_ in D6_COLUMNS}
    where = [f"{c}: {len(cols[c]['locations'])} in {mice[c]} {'mouse' if mice[c] == 1 else 'mice'}, "
             f"{sum(cols[c]['recordings'])} recordings, {sum(cols[c]['cases'])} cases" for c, *_ in D6_COLUMNS]
    failed = ", ".join(f"{NAMES[m]} {n}" for m, n in record["failures"].items())
    lines += ["", "Locations: " + "; ".join(where) + ".",
              f"Cases by territory class (ASTRID's rows): {record['cases_by_class']}.",
              f"Failures (cases scored leakage 1 and r 0): {failed}.", ""]
    if "baseline" in record:
        lines += [f"Baseline: {record['baseline']}.", ""]
    ts = record.get("transient_size")
    if ts is not None:
        lines += ["## Kept transient size on R (descriptive)", "",
                  f"Each location's median over its cases with at least {SIZE_MIN_EVENTS} of the neuron's own "
                  f"transients ({ts['used']} of {ts['cases']} cases, {ts['transients']} transients), mean ± SD over "
                  "the locations; 1 keeps them at the size of the raw ROI average.", "",
                  "| method | " + " | ".join(SIZE_CLASSES) + " |", "|---|" + "---|" * len(SIZE_CLASSES)]
        for m in METHODS:
            lines.append(f"| {NAMES[m]} | " + " | ".join(f"{ts[k][m]['mean']:.3f} ± {ts[k][m]['sd']:.3f}"
                                                        for k in SIZE_CLASSES) + " |")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- D7

D7_ERRORS = ("err_05hz", "err_2hz", "err_raw")


def d7_tables(root: Path, baseline: str = "reported") -> dict[str, pd.DataFrame]:
    """D7's per-case tables over every recording's scores of ``baseline`` (``cases.SCORE_DIRS``), with D7's
    failure rule applied.

    Inputs, one folder per recording under ``root``, named ``<location>_<k>`` (the location, then the recording's
    name after the last underscore): ``compare/index.json`` (the cases, ``astrid_repro.peers``' format) and, in
    the scoring's folder, ``injection_scores.csv`` (``case``, ``method``, ``corr``, ``gain``, ``err_05hz``,
    ``err_2hz``, ``err_raw``, ``sbr``) and ``leakage_scores.csv`` (``target``, ``scale``, ``method``, ``leak``,
    ``kept_corr``, ``arr``); a recording counts once its scorer's printout ``summary.txt`` is there.

    The failure rule: a method's trace that is missing (no score row for a case of the index), not finite or
    constant (a NaN correlation) is scored as the raw ROI average: its errors, gain and correlation on that case,
    and leakage 1 with the raw ROI's kept correlation (``status`` "failure")."""
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {BASELINES}, got {baseline!r}")
    folder = SCORE_DIRS[baseline]
    inj, leak = [], []
    roi = REFERENCE[0]
    runs = sorted(p.parent.parent for p in root.glob(f"*/{folder}/summary.txt"))
    if not runs:
        raise SystemExit(f"no recording scored under {root} (*/{folder}/summary.txt)")
    for run in runs:
        location, k = run.name.rsplit("_", 1)
        rec = f"{location}/{k}"
        index = json.loads((run / "compare" / "index.json").read_text(encoding="utf-8"))
        cases = [c for c in index if c["test"] == "injection"]
        if cases:
            s = pd.read_csv(run / folder / "injection_scores.csv")
            for c in cases:
                g = s[s.case == c["spec"]].set_index("method")
                if roi not in g.index:
                    raise SystemExit(f"{run.name} {c['spec']}: no raw ROI row")
                for m in METHODS:
                    ok = (m in g.index and np.isfinite(g.loc[m, list(D7_ERRORS)].to_numpy(float)).all()
                          and np.isfinite(float(g.loc[m, "corr"])))
                    src = g.loc[m] if ok else g.loc[roi]
                    inj.append({"recording": rec, "location": location, "case": c["spec"], "site": c["site"],
                                "scale": float(c["scale"]), "method": m, "status": "ok" if ok else "failure",
                                **{e: float(src[e]) for e in (*D7_ERRORS, "gain", "corr")},
                                "sbr": float(g.loc[roi, "sbr"])})
        cases = [c for c in index if c["test"] == "leakage" and c["scale"] > 0]
        if cases:
            s = pd.read_csv(run / folder / "leakage_scores.csv")
            for c in cases:
                g = s[(s.target == c["base"]) & (s.scale == c["scale"])].set_index("method")
                for m in METHODS:
                    ok = (m in g.index and np.isfinite(float(g.loc[m, "leak"]))
                          and np.isfinite(float(g.loc[m, "kept_corr"])))
                    leak.append({"recording": rec, "location": location, "target": c["base"],
                                 "scale": float(c["scale"]), "method": m, "status": "ok" if ok else "failure",
                                 "leak": float(g.loc[m, "leak"]) if ok else 1.0,
                                 "kept_corr": float(g.loc[m if ok else roi, "kept_corr"]),
                                 "arr": float(g.loc[roi, "arr"])})
    return {"injection": pd.DataFrame(inj), "leakage": pd.DataFrame(leak)}


def d7_injection_status(run: Path, index: list[dict[str, Any]], baseline: str = "reported") -> pd.DataFrame:
    """Per injected case and method of one recording, ``status`` ok or failure under D7's failure rule
    (``d7_tables``), from the scores of ``baseline``."""
    s = pd.read_csv(run / SCORE_DIRS[baseline] / "injection_scores.csv")
    rows = []
    for c in [c for c in index if c["test"] == "injection"]:
        g = s[s.case == c["spec"]].set_index("method")
        for m in METHODS:
            ok = (m in g.index and np.isfinite(g.loc[m, list(D7_ERRORS)].to_numpy(float)).all()
                  and np.isfinite(float(g.loc[m, "corr"])))
            rows.append({"case": c["spec"], "method": m, "status": "ok" if ok else "failure"})
    return pd.DataFrame(rows)


def d7_info(t: pd.DataFrame) -> pd.DataFrame:
    return t[t.method == OURS].groupby("recording").agg(location=("location", "first"))


def d7_locations(primary: dict[str, Any]) -> dict[str, Any]:
    """The imaging locations apart: per primary column and location, the mean and SD of each method and the
    recordings where ASTRID is lower; no test."""
    out: dict[str, Any] = {}
    for col, c in primary.items():
        units = pd.DataFrame(c["scores"], index=c["units"])
        where = pd.Series(c["location"], index=c["units"])
        out[col] = {}
        for location, sub in units.groupby(where):
            out[col][str(location)] = {"n": int(len(sub)), "mean": {m: float(sub[m].mean()) for m in METHODS},
                                       "sd": {m: float(sub[m].std(ddof=1)) if len(sub) > 1 else float("nan")
                                              for m in METHODS},
                                       "wins": {m: int((sub[OURS] < sub[m]).sum()) for m in COMPARED}}
    return out


def d7(root: Path, baseline: str = "reported") -> dict[str, Any]:
    """D7's record (``d7_tables``): the recording is the unit (``grid_dataset``), with the imaging locations apart
    (``d7_locations``) and the failures per test and method."""
    t = d7_tables(root, baseline)
    d = grid_dataset(t["injection"], t["leakage"], "recording",
                     {"injection": d7_info(t["injection"]), "leakage": d7_info(t["leakage"])})
    d["locations"] = d7_locations(d["primary"])
    d["failures"] = {k: {"cases": int((v.method == OURS).sum()),
                         **{m: int(((v.method == m) & (v.status == "failure")).sum()) for m in METHODS}}
                     for k, v in t.items()}
    return {"unit": "recording", "recordings": int(t["injection"].recording.nunique()), "baseline": baseline, **d}


def d7_markdown(record: dict[str, Any]) -> str:
    lines = grid_markdown(record, f"D7: the injection and leakage tests over the recordings ({record['baseline']})")
    lines += ["## The imaging locations apart", "",
              "Mean ± SD over each location's recordings, and the recordings where ASTRID is lower; no test.", "",
              "| column | location | n | " + " | ".join(NAMES[m] for m in METHODS) + " |",
              "|---|---|---|" + "---|" * len(METHODS)]
    for col, per in record["locations"].items():
        for location, c in per.items():
            vals = [f"{c['mean'][m]:.3f} ± {c['sd'][m]:.3f}"
                    + ("" if m in (OURS, *REFERENCE) else f" ({c['wins'][m]}/{c['n']})") for m in METHODS]
            lines.append(f"| {col} | {location} | {c['n']} | " + " | ".join(vals) + " |")
    for name, f in record["failures"].items():
        lines += ["", f"Failures ({name}, {f['cases']} cases of ASTRID, scored as the raw ROI): "
                  + ", ".join(f"{NAMES[m]} {f[m]}" for m in METHODS if m not in REFERENCE) + "."]
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ D7's error split

SPLIT_METHODS = ("demix", "ring", "fissa", "tuncat", "tuncat_astro", "cnmf", "cnmf_astro", "suite2p")
SPLIT_VALUES = ("err05", "gain_part", "resid_part", "g05", "g_raw", "dev05", "dev_raw")
SPLIT_BAND_HZ = 0.5                    # the error's band


def error_split(trace: np.ndarray, truth: np.ndarray, fps: float) -> dict[str, float]:
    """Supplementary Equation S24 for one trace y and its added signal q, both low-passed below 0.5 Hz with their
    means removed (the error's own signals): ``g05`` = y·q / q·q, the gain part (1 - g05)², and the residual part
    |y - g05 q|² / |q|², whose sum is the squared error ``err05``; and ``g_raw``, the gain on the unfiltered
    traces (D1's gain). Each trace in its own precision, as ``cases.injection_scores`` scores it."""
    from astrid_repro.cases import ac, band

    yb, qb = band(trace, SPLIT_BAND_HZ, fps), band(truth, SPLIT_BAND_HZ, fps)
    g = float(yb @ qb / (qb @ qb))
    r = yb - g * qb
    return {"g05": g, "gain_part": (1 - g) ** 2, "resid_part": float(r @ r / (qb @ qb)),
            "err05": float(np.linalg.norm(yb - qb) / np.linalg.norm(qb)),
            "g_raw": float(ac(trace) @ ac(truth) / (ac(truth) @ ac(truth)))}


def d7_error_split(run: Path, results: list[Path] | None = None) -> pd.DataFrame:
    """One recording's error split (``error_split``) on the traces as the methods give them, one row per injected
    case and method of ``SPLIT_METHODS``, with D7's failure rule (``d7_injection_status``): a failed method takes
    the raw ROI average's values (``failed`` True). ``g_raw_as_is`` and ``g05_as_is`` are the method's own gains
    whatever its status (NaN without a trace), from which the gains below ``LOW_GAIN`` are counted; ``dev_raw``
    and ``dev05`` are |g - 1|. Inputs: a D7 run folder (``astrid_repro.d7``: ``run.json``, ``compare/``,
    ``injection/``, ``scores/``) and the folders of the peers' results (default ``<run>/compare/results``)."""
    from astrid_repro.cases import case_traces

    fps = float(json.loads((run / "run.json").read_text(encoding="utf-8"))["fps"])
    index = json.loads((run / "compare" / "index.json").read_text(encoding="utf-8"))
    status = d7_injection_status(run, index)
    folders = results if results is not None else [run / "compare" / "results"]
    rows = []
    for c in [c for c in index if c["test"] == "injection"]:
        tr = case_traces(run / "injection", folders, c["spec"])
        truth = np.load(run / "compare" / f"{c['spec']}.npz")["truth"]
        vals = {m: error_split(tr[m], truth, fps) for m in (*SPLIT_METHODS, *REFERENCE) if m in tr}
        st = status[status.case == c["spec"]].set_index("method").status
        for m in SPLIT_METHODS:
            ok = st[m] == "ok"
            own = vals.get(m, {"g_raw": float("nan"), "g05": float("nan")})
            rows.append({"case": c["spec"], "scale": c["scale"], "method": m, "failed": not ok,
                         **(vals[m] if ok else vals[REFERENCE[0]]),
                         "g_raw_as_is": own["g_raw"], "g05_as_is": own["g05"]})
    t = pd.DataFrame(rows)
    t["dev_raw"] = (t.g_raw - 1).abs()
    t["dev05"] = (t.g05 - 1).abs()
    return t


def d7_split_table(root: Path, results: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The error split over D7's recordings (``d7_tables``' folders under ``root``; ``results``: the peers'
    results folders, relative to each recording's folder, default ``compare/results``): the per-case rows
    (with ``location`` and ``recording``) and the record. Per recording, the median of each value of
    ``SPLIT_VALUES`` over its cases at each strength; then per method and strength the mean ± SD over the
    recordings (so the parts need not add up to the squared error) and the mean of each imaging location; and
    the x3 cases whose gain, raw or below 0.5 Hz, is below ``LOW_GAIN`` (the traces as they are). Descriptive,
    without tests."""
    runs = sorted(p.parent.parent for p in root.glob("*/scores/summary.txt"))
    if not runs:
        raise SystemExit(f"no scored recording under {root}")
    parts = []
    for run in runs:
        location, k = run.name.rsplit("_", 1)
        folders = [run / r for r in (results or ["compare/results"])]
        parts.append(d7_error_split(run, folders).assign(location=location, recording=f"{location}/{k}"))
    t = pd.concat(parts, ignore_index=True)
    rec = t.groupby(["recording", "location", "method", "scale"])[list(SPLIT_VALUES)].median().reset_index()
    record: dict[str, Any] = {"recordings": len(runs), "by_strength": {}, "by_location": {}}
    for scale, g in rec.groupby("scale"):
        key = f"x{scale:g}"
        record["by_strength"][key] = {m: {v: {"mean": float(g[g.method == m][v].mean()),
                                              "sd": float(g[g.method == m][v].std(ddof=1))}
                                          for v in SPLIT_VALUES} for m in SPLIT_METHODS}
        record["by_location"][key] = {loc: {m: {v: float(h[h.method == m][v].mean()) for v in SPLIT_VALUES}
                                            for m in SPLIT_METHODS} for loc, h in g.groupby("location")}
    x3 = t[t.scale == 3]
    record["low_gain_x3"] = {"cases": int((x3.method == OURS).sum()),
                             "raw": {m: int((x3[x3.method == m].g_raw_as_is < LOW_GAIN).sum()) for m in SPLIT_METHODS},
                             "g05": {m: int((x3[x3.method == m].g05_as_is < LOW_GAIN).sum()) for m in SPLIT_METHODS}}
    return t, record


def d7_split_markdown(record: dict[str, Any]) -> str:
    lines = ["# D7: the error split (descriptive)", "",
             "Per injected case: y and q, the trace and the added signal below 0.5 Hz with their means removed; "
             "g = y·q / q·q; error² = (1 - g)² + |y - g q|² / |q|². Each recording's median over its cases, then "
             f"mean ± SD over the {record['recordings']} recordings. A failed trace takes the raw ROI's values.", ""]
    for key, block in record["by_strength"].items():
        lines += [f"**{key}**", "", "| method | error | gain part | residual part | g (0.5 Hz) | g (raw) | "
                  "mean abs(g - 1) (raw) |", "|---|---|---|---|---|---|---|"]
        for m in SPLIT_METHODS:
            b = block[m]
            cells = [f"{b[v]['mean']:.3f} ± {b[v]['sd']:.3f}" for v in ("err05", "gain_part", "resid_part", "g05",
                                                                        "g_raw")]
            lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + f" | {b['dev_raw']['mean']:.3f} |")
        lines.append("")
    low = record["low_gain_x3"]
    lines += [f"x3 cases with a gain below {LOW_GAIN} (of {low['cases']}; the traces as they are): raw gain "
              + ", ".join(f"{NAMES[m]} {low['raw'][m]}" for m in SPLIT_METHODS) + "; 0.5-Hz gain "
              + ", ".join(f"{NAMES[m]} {low['g05'][m]}" for m in SPLIT_METHODS) + ".", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- imaging locations in mice

NEST_BOOT, NEST_SEED = 10_000, 0       # the hierarchical bootstrap's resamples and seed
NEST_ALPHA = 0.05                      # the tables' significance
NEST_OPTIMIZERS = (None, "powell", "nm", "bfgs", "cg")   # the mixed model's: statsmodels' default first
D6_NEST_COLUMN = "leakage in territory"


NEST_GRID_COLUMNS = ("error curve", "leakage curve")


def nesting_columns(d1: Path, d6: Path, d7: Path) -> list[dict[str, Any]]:
    """Every column the check covers, from the tables' own per-unit scores: D1's and D7's error and leakage curves
    (``stats d1``'s and ``stats d7``'s records, Holm over the seven) and D6's primary column (``stats d6``'s
    record). A unit's mouse is the tables' own: on D1 and D6 the start of the location's name, on D7 the location
    (one mouse each)."""
    cols: dict[str, list[dict[str, Any]]] = {}
    for ds, path, mouse_key in (("D1", d1, "mouse"), ("D7", d7, "location")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        cols[ds] = []
        for col in NEST_GRID_COLUMNS:
            c = rec["primary"][col]
            cols[ds].append({"dataset": ds, "column": col, "units": c["units"], "mouse": c[mouse_key],
                             "scores": {m: dict(zip(c["units"], c["scores"][m], strict=True))
                                        for m in (OURS, *COMPARED)},
                             "p_holm": {m: c["stats"][m]["p_holm"] for m in COMPARED}})
    c = json.loads(d6.read_text(encoding="utf-8"))["columns"][D6_NEST_COLUMN]
    six = {"dataset": "D6", "column": D6_NEST_COLUMN, "units": c["locations"], "mouse": c["mouse"],
           "scores": {m: dict(zip(c["locations"], c["scores"][m], strict=True)) for m in (OURS, *COMPARED)},
           "p_holm": {m: c["stats"][m]["p_holm"] for m in COMPARED}}
    return [*cols["D1"], six, *cols["D7"]]


def nesting_per_mouse(d: pd.Series, mouse: pd.Series, direction: float) -> dict[str, Any]:
    """Per mouse the units, the mean and median of d and the units with d < 0, and whether every mouse's mean has
    the sign of the mean over all units."""
    per = {}
    for m in sorted(set(mouse)):
        x = d[mouse == m]
        per[m] = {"n": int(x.size), "mean": float(x.mean()), "median": float(x.median()),
                  "n_negative": int((x < 0).sum())}
    return {"per_mouse": per, "agrees_in_every_mouse": bool(all(np.sign(v["mean"]) == direction
                                                                for v in per.values()))}


def nesting_mixed_model(d: pd.Series, mouse: pd.Series) -> dict[str, Any]:
    """d ~ 1 + (1 | mouse) by REML (statsmodels' MixedLM): the intercept, its SE, the Wald z and its p, a t with
    mice - 1 degrees of freedom and its p, the mouse and residual variances, and whether the fit is singular (the
    mouse variance below 1e-8 of the residual's, a boundary warning, or no SE). A fit without an SE is refitted
    with the other optimizers of ``NEST_OPTIMIZERS``; if none gives one, its p is 1."""
    import warnings

    from scipy import stats as st
    try:
        import statsmodels.formula.api as smf
    except ImportError as err:      # pragma: no cover - optional dependency
        raise ImportError("the mixed model needs statsmodels: pip install -e \".[nesting]\"") from err
    df = pd.DataFrame({"d": d.to_numpy(float), "mouse": mouse.to_numpy()})
    for method in NEST_OPTIMIZERS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm("d ~ 1", df, groups=df["mouse"])
            fit = model.fit(reml=True) if method is None else model.fit(reml=True, method=method)
        est, se = float(fit.fe_params["Intercept"]), float(fit.bse_fe["Intercept"])
        if np.isfinite(se) and se > 0:
            break
    var_mouse, var_resid = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    msgs = sorted({str(w.message).splitlines()[0][:120] for w in caught})
    k = df.mouse.nunique()
    ok = bool(np.isfinite(se) and se > 0)
    z = est / se if ok else float("nan")
    return {"intercept": est, "se": se if ok else None, "z": z if ok else None,
            "p_z": float(2 * st.norm.sf(abs(z))) if ok else 1.0, "df_t": k - 1,
            "p_t": float(2 * st.t.sf(abs(z), k - 1)) if ok else 1.0, "se_estimable": ok,
            "optimizer": method or "default", "var_mouse": var_mouse, "var_resid": var_resid,
            "singular": bool(var_mouse < 1e-8 * var_resid or any("boundary" in m.lower() for m in msgs) or not ok),
            "converged": bool(fit.converged), "warnings": msgs}


def nesting_bootstrap(d: pd.Series, mouse: pd.Series) -> dict[str, Any]:
    """The hierarchical bootstrap (Saravanan et al. 2020): the mice with replacement, then each drawn mouse's units
    with replacement, the mean of d over the drawn units, ``NEST_BOOT`` times from seed ``NEST_SEED``; the 95 %
    percentile interval and p = 2 min(P(mean <= 0), P(mean >= 0))."""
    groups = [d[mouse == m].to_numpy(float) for m in sorted(set(mouse))]
    rng = np.random.default_rng(NEST_SEED)
    means = np.empty(NEST_BOOT)
    for i in range(NEST_BOOT):
        picks = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in (groups[j] for j in picks)])
        means[i] = vals.mean()
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "p": p, "n_boot": NEST_BOOT, "seed": NEST_SEED}


def nesting(d1_folder: Path, d6: Path, d7: Path, mixed_model: bool = True) -> dict[str, Any]:
    """The check of the tests' units nested in mice (Supplementary Section "Imaging locations within mice", Table
    S5), from the tables' records (``nesting_columns``). Per column and comparator, d = ASTRID's score minus the
    other method's per unit (lower is better, so d < 0 favors ASTRID); the direction is the sign of d's mean, the
    significance the table's Holm p below ``NEST_ALPHA``.
    - ``A`` (every dataset): per mouse (``nesting_per_mouse``);
    - ``B`` (D1): the mixed model (``nesting_mixed_model``; statsmodels, left out with ``mixed_model`` False);
    - ``C`` (D1): the hierarchical bootstrap (``nesting_bootstrap``).
    Holm over the seven comparators of a column for B's two p and C's p (``p_t_holm``, ``p_z_holm``,
    ``p_holm``)."""
    record: dict[str, Any] = {"results": []}
    for col in nesting_columns(d1_folder, d6, d7):
        units, mouse = pd.Series(col["units"]), pd.Series(col["mouse"])
        rows = []
        for m in COMPARED:
            d = pd.Series([col["scores"][OURS][u] - col["scores"][m][u] for u in units])
            direction = float(np.sign(d.mean()))
            r = {"comparison": m, "n": int(d.size), "mean_d": float(d.mean()), "direction": direction,
                 "p_holm_paper": col["p_holm"][m], "significant": bool(col["p_holm"][m] < NEST_ALPHA),
                 "A": nesting_per_mouse(d, mouse, direction)}
            if col["dataset"] == "D1":
                if mixed_model:
                    r["B"] = nesting_mixed_model(d, mouse)
                r["C"] = nesting_bootstrap(d, mouse)
            rows.append(r)
        if col["dataset"] == "D1":
            for key, sub in (("p_t", "B"), ("p_z", "B"), ("p", "C")):
                if sub in rows[0]:
                    h = holm({r["comparison"]: r[sub][key] for r in rows})
                    for r in rows:
                        r[sub][key + "_holm"] = h[r["comparison"]]
        record["results"].append({"dataset": col["dataset"], "column": col["column"],
                                  "units_per_mouse": mouse.value_counts().sort_index().to_dict(), "rows": rows})
    return record


def nesting_markdown(record: dict[str, Any]) -> str:
    lines = ["# The tests' units nested in mice (a sensitivity check)", "",
             "d = ASTRID's score minus the other method's per unit; d < 0 favors ASTRID. A: per mouse, the mean d "
             "(the units with d < 0 of n). B (D1): d ~ 1 + (1 | mouse), REML; Holm p from a t with mice - 1 degrees "
             f"of freedom and from the Wald z. C (D1): hierarchical bootstrap, {NEST_BOOT:,} resamples, seed "
             f"{NEST_SEED}, 95 % interval, Holm p. * the table's Holm p below {NEST_ALPHA}.", ""]
    for res in record["results"]:
        mice = sorted(res["units_per_mouse"])
        d1 = res["dataset"] == "D1"
        head = ["comparison", "mean d, Holm p"] + [f"A: {k}" for k in mice] + ["A: same sign in every mouse"]
        if d1:
            head += ["B: intercept", "B: Holm p (t)", "B: Holm p (z)", "B: singular", "C: 95 % interval",
                     "C: Holm p"]
        lines += [f"## {res['dataset']}, {res['column']}", "", "| " + " | ".join(head) + " |",
                  "|" + "---|" * len(head)]
        for r in res["rows"]:
            cells = [NAMES[r["comparison"]], f"{r['mean_d']:+.3f}, {r['p_holm_paper']:.4f}"
                     + (" *" if r["significant"] else "")]
            cells += [f"{r['A']['per_mouse'][k]['mean']:+.3f} ({r['A']['per_mouse'][k]['n_negative']}/"
                      f"{r['A']['per_mouse'][k]['n']})" for k in mice]
            cells.append("yes" if r["A"]["agrees_in_every_mouse"] else "**no**")
            if d1:
                b, c = r.get("B"), r["C"]
                cells += ([f"{b['intercept']:+.3f}", f"{b['p_t_holm']:.4f}", f"{b['p_z_holm']:.4f}",
                           "yes" if b["singular"] else "no"] if b else ["", "", "", ""])
                cells += [f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]", f"{c['p_holm']:.4f}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("d1", help="D1's injection and leakage tests")
    p.add_argument("--scores", type=Path, required=True, help="the pooled per-case scores of every method")
    p.add_argument("--suffix", default="", help="the files' suffix: injection_scores_<suffix>.csv")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("ablations", help="D1's ablations")
    p.add_argument("--full", type=Path, required=True, help="the full model's pooled per-case scores")
    p.add_argument("--variant", nargs="+", required=True, help="I=<folder> II=<folder> III=<folder>")
    p.add_argument("--suffix", default="")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("d6-scores", help="D6's per-case scores and kept transient sizes of one recording")
    p.add_argument("--run", type=Path, required=True, help="the recording's folder (see d6_scores)")
    p.add_argument("--fps", type=float, default=D6_FPS)
    p.add_argument("--baseline", choices=D6_BASELINES, default="reported",
                   help=f"equal: {D6_EQUAL_BASELINE} (the paper's table)")
    p.add_argument("--out", type=Path, default=None,
                   help="the scores' CSV (default <run>/scores.csv, or scores_equal.csv); the sizes go to "
                        "<run>/transient_size.csv (transient_size_equal.csv)")
    p = sub.add_parser("d6", help="D6's table from every recording's scores")
    p.add_argument("--root", type=Path, required=True,
                   help="the recordings' folders, each with the scores (and the sizes) of the baseline")
    p.add_argument("--baseline", choices=D6_BASELINES, default="reported")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="d6")
    p = sub.add_parser("d7", help="D7's table from every recording's per-case scores")
    p.add_argument("--root", type=Path, required=True, help="the recordings' folders (see d7_tables)")
    p.add_argument("--baseline", choices=BASELINES, default="reported",
                   help="the scoring whose folder is read (without_added: the paper's tables)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="d7")
    p = sub.add_parser("d7-split", help="D7's error split (Supplementary Equation S24)")
    p.add_argument("--root", type=Path, required=True, help="the recordings' folders (see d7_tables)")
    p.add_argument("--results", nargs="+", default=None,
                   help="the peers' results folders, relative to each recording's folder (default compare/results)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("nesting", help="the tests' units nested in mice (Supplementary Table S5)")
    p.add_argument("--d1", type=Path, required=True, help="stats d1's record (d1.json)")
    p.add_argument("--d6", type=Path, required=True, help="stats d6's record (the d6.json of its scoring)")
    p.add_argument("--d7", type=Path, required=True, help="stats d7's record (d7.json)")
    p.add_argument("--no-mixed-model", action="store_true", help="leave out the mixed model (no statsmodels)")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")         # the tables' ± and − on a Windows console
    if a.command == "d1":
        record = d1(read_scores(a.scores, a.suffix))
        write_json(a.out / "d1.json", record)
        text = d1_markdown(record)
        (a.out / "d1.md").write_text(text, encoding="utf-8")
    elif a.command == "ablations":
        folders = dict(v.split("=", 1) for v in a.variant)
        if sorted(folders) != sorted(VARIANTS):
            ap.error(f"--variant needs exactly {', '.join(VARIANTS)}")
        record = ablations(read_scores(a.full, a.suffix),
                           {v: read_scores(Path(folders[v]), a.suffix) for v in VARIANTS})
        write_json(a.out / "ablation_stats.json", record)
        text = ablation_markdown(record)
        (a.out / "ablation_stats.md").write_text(text, encoding="utf-8")
    elif a.command == "d6-scores":
        s = d6_scores(a.run, a.fps, a.baseline)
        s.to_csv(a.run / D6_SCORE_FILES[a.baseline] if a.out is None else a.out, index=False)
        sizes = d6_sizes(a.run, a.fps, a.baseline)
        sizes.to_csv(a.run / D6_SIZE_FILES[a.baseline], index=False)
        text = (f"not ok: {s[s.status != 'ok'].groupby(['method', 'status']).size().to_dict()}; "
                f"{int((sizes.n_events >= SIZE_MIN_EVENTS).sum())} of {len(sizes)} cases with transients")
    elif a.command == "d6":
        files = sorted(a.root.glob(f"*/{D6_SCORE_FILES[a.baseline]}"))
        if not files:
            raise SystemExit(f"no {D6_SCORE_FILES[a.baseline]} under {a.root}")
        record = d6_tables([pd.read_csv(f) for f in files])
        if a.baseline == "equal":
            record["baseline"] = D6_EQUAL_BASELINE
        sizes = [f.parent / D6_SIZE_FILES[a.baseline] for f in files]
        if all(p.exists() for p in sizes):
            record["transient_size"] = d6_size_table([pd.read_csv(p, dtype={"case": str}) for p in sizes])
        write_json(a.out / f"{a.name}.json", record)
        text = d6_markdown(record, a.name)
        (a.out / f"{a.name}.md").write_text(text, encoding="utf-8")
    elif a.command == "d7-split":
        cases, record = d7_split_table(a.root, a.results)
        a.out.mkdir(parents=True, exist_ok=True)
        cases.to_csv(a.out / "d7_split_cases.csv", index=False)
        write_json(a.out / "d7_split.json", record)
        text = d7_split_markdown(record)
        (a.out / "d7_split.md").write_text(text, encoding="utf-8")
    elif a.command == "nesting":
        record = nesting(a.d1, a.d6, a.d7, mixed_model=not a.no_mixed_model)
        write_json(a.out / "nesting.json", record)
        text = nesting_markdown(record)
        (a.out / "nesting.md").write_text(text, encoding="utf-8")
    else:
        record = d7(a.root, a.baseline)
        write_json(a.out / f"{a.name}.json", record)
        text = d7_markdown(record)
        (a.out / f"{a.name}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
