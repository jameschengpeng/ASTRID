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

Commands (from ``reproduce/``: ``python -m astrid_repro.stats <command> --help``):
- ``d1``: D1's injection and leakage tests (``d1``). Input: the per-case scores pooled over D1's recordings,
  ``injection_scores[_<suffix>].csv`` (``location``, ``recording``, ``case``, ``site``, ``scale``, ``method``,
  ``err_05hz``, ``gain``, ...) and ``leakage_scores[_<suffix>].csv`` (``location``, ``recording``,
  ``target``, ``scale``, ``method``, ``leak``, ``kept_corr``, ...), every method's rows. A location's name
  starts with its mouse's, before the first underscore. Output: ``d1_stats.json`` (the four primary columns,
  the six comparisons before Suite2p, every p Holm-corrected over the seven with Suite2p, ``p_holm``, and
  over the six, ``p_holm_declared``), ``d1_suite2p.json`` (ASTRID against Suite2p on the primary and
  secondary measures), ``d1_secondary.json`` (the secondary measures' families of seven) and ``d1_stats.md``.
- ``ablations``: D1's ablations (``ablations``), full ASTRID against variants I, II and III, each from its own
  pooled scores (ASTRID's rows only; every variant must have exactly the full model's cases). Output:
  ``ablation_stats.json`` and ``.md``.
- ``d6-scores``: D6's per-case scores of one recording from the traces (``d6_scores``) and the kept transient
  size of each case (``d6_sizes``); ``d6``: D6's table from every recording's scores (``d6_tables``), with the
  kept transient size (``d6_size_table``) where every recording has it. Both take ``--baseline``: ``reported``,
  every trace as the method gives it, or ``equal`` (the paper's table), every trace but ASTRID's and the ring
  subtraction's minus its own running baseline under ASTRID's rule (``D6_EQUAL_BASELINE``).
- ``d7``: D7's table from every recording's per-case scores, with D7's failure rule (``d7_tables``).
- ``d7-split``: D7's error split (Supplementary Equation S24, ``d7_split_table``) from every recording's traces:
  per case and method the gain and residual parts of the squared error below 0.5 Hz, and the x3 cases with a
  gain below 0.5. Output: ``d7_split_cases.csv``, ``d7_split.json`` and ``.md``.
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

from astrid_repro.metrics import holm, leakage, lowpass_fourier, usable, wilcoxon_p
from astrid_repro.public_data import minus_baseline

OURS = "demix"                         # ASTRID's rows in the score tables
REFERENCE = ("roi",)                   # the raw ROI average: a reference row, no test
NAMES = {"demix": "ASTRID", "tuncat": "TUnCaT", "tuncat_astro": "TUnCaT + terr.", "fissa": "FISSA",
         "cnmf": "CNMF", "cnmf_astro": "CNMF + terr.", "ring": "Ring", "suite2p": "Suite2p", "roi": "raw ROI"}
# D1's primary columns, also D7's: (name, table, score, strength)
COLUMNS = (("error x1", "injection", "err_05hz", 1.0), ("error x3", "injection", "err_05hz", 3.0),
           ("leakage x1", "leakage", "leak", 1.0), ("leakage x3", "leakage", "leak", 3.0))
# D1's primary and secondary columns: (name, table, score, strength, site (None: all), better, primary)
D1_MEASURES: tuple[tuple[str, str, str, float, str | None, str, bool], ...] = (
    ("error x1", "injection", "err_05hz", 1.0, None, "lower", True),
    ("error x3", "injection", "err_05hz", 3.0, None, "lower", True),
    ("leakage x1", "leakage", "leak", 1.0, None, "lower", True),
    ("leakage x3", "leakage", "leak", 3.0, None, "lower", True),
    ("gain deviation x1", "injection", "gain_dev", 1.0, None, "lower", False),          # |gain - 1|
    ("gain deviation x3", "injection", "gain_dev", 3.0, None, "lower", False),
    ("kept correlation x1", "leakage", "kept_corr", 1.0, None, "higher", False),
    ("kept correlation x3", "leakage", "kept_corr", 3.0, None, "higher", False),
    ("error x1, astrocyte sites", "injection", "err_05hz", 1.0, "astro", "lower", False),
    ("error x1, control sites", "injection", "err_05hz", 1.0, "control", "lower", False),
)


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


# ----------------------------------------------------------------------------------------------------- D1

D1_DECLARED = ("tuncat", "tuncat_astro", "fissa", "cnmf", "cnmf_astro", "ring")   # the family before Suite2p
D1_SECONDARY: tuple[tuple[str, str, str, float, str | None, str], ...] = (
    *(m[:6] for m in D1_MEASURES[4:]),
    ("error x3, astrocyte sites", "injection", "err_05hz", 3.0, "astro", "lower"),
    ("error x3, control sites", "injection", "err_05hz", 3.0, "control", "lower"),
)
D1_CASE = {"injection": ["location", "recording", "case"],                     # a case's identity per table
           "leakage": ["location", "recording", "target", "scale"]}


def d1_by_location(table: pd.DataFrame, value: str, scale: float) -> pd.DataFrame:
    """(location x method): each location's median over its cases at ``scale``."""
    g = table[table.scale == scale]
    loc = g.groupby(["location", "method"])[value].median().unstack()
    if loc.isna().any().any():
        raise SystemExit(f"{value} x{scale:g}: a location lacks a method")
    return loc


def d1_compare(loc: pd.DataFrame) -> dict[str, dict[str, float]]:
    """ASTRID minus each declared comparator over the locations: the mean difference, the locations where
    ASTRID is lower, the Wilcoxon p and Holm's over the six."""
    stats: dict[str, dict[str, float]] = {}
    raw_p: dict[str, float] = {}
    for m in D1_DECLARED:
        d = (loc[OURS] - loc[m]).to_numpy()
        p = wilcoxon_p(d)
        raw_p[m] = p
        stats[m] = {"mean_diff": float(d.mean()), "wins": int((d < 0).sum()), "n": int(d.size), "p": p}
    for m, ph in holm(raw_p).items():
        stats[m]["p_holm"] = ph
    return stats


def d1_primary(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """The four primary columns against the six declared comparators (Holm over the six)."""
    record: dict[str, Any] = {}
    for col, name, value, scale in COLUMNS:
        t = tables[name]
        loc = d1_by_location(t, value, scale)
        g = t[(t.scale == scale) & (t.method == OURS)]
        shown = (OURS, *D1_DECLARED, *REFERENCE)
        record[col] = {"measure": f"{value}, each location's median over its cases", "scale": scale,
                       "locations": loc.index.tolist(),
                       "mouse": [s.split("_")[0] for s in loc.index],
                       "recordings": g.groupby("location").recording.nunique().reindex(loc.index).tolist(),
                       "cases": g.groupby("location").size().reindex(loc.index).tolist(),
                       "scores": {m: loc[m].tolist() for m in shown},
                       "mean": {m: float(loc[m].mean()) for m in shown},
                       "sd": {m: float(loc[m].std(ddof=1)) for m in shown},
                       "stats": d1_compare(loc)}
    return record


def suite2p_test(ours: np.ndarray, s2p: np.ndarray, better: str) -> dict[str, Any]:
    """ASTRID against Suite2p over paired units: the mean difference (ASTRID minus Suite2p), the units where
    ASTRID is better and the two-sided Wilcoxon p."""
    d = np.asarray(ours, float) - np.asarray(s2p, float)
    wins = int((d < 0).sum()) if better == "lower" else int((d > 0).sum())
    return {"mean_diff": float(d.mean()), "wins": wins, "n": int(d.size), "p": wilcoxon_p(d)}


def d1_suite2p(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """ASTRID against Suite2p on every measure of ``D1_MEASURES``, each location's median over its cases."""
    t = {}
    for k, key in D1_CASE.items():
        ours, s2p = tables[k][tables[k].method == OURS], tables[k][tables[k].method == "suite2p"]
        cases = [sorted(map(tuple, g[key].astype(str).to_numpy().tolist())) for g in (ours, s2p)]
        if cases[0] != cases[1] or len(set(cases[0])) != len(cases[0]):
            raise SystemExit(f"D1 {k}: Suite2p's cases are not ASTRID's, one each")
        t[k] = pd.concat([ours, s2p], ignore_index=True)
    t["injection"]["gain_dev"] = (t["injection"].gain - 1).abs()
    out: dict[str, Any] = {}
    for col, kind, value, scale, site, better, primary in D1_MEASURES:
        g = t[kind]
        g = g[(g.scale == scale) & ((g.site == site) if site else True)]
        loc = g[g.method.isin(["demix", "suite2p"])].groupby(["location", "method"])[value].median().unstack()
        if loc.isna().any().any() or set(loc.columns) != {"demix", "suite2p"}:
            raise SystemExit(f"{col}: a location lacks ASTRID's or Suite2p's score")
        own = g[g.method == "demix"]
        where = f" (sites: {site})" if site else ""
        out[col] = {"measure": f"{value}, each location's median over its cases{where}",
                    "scale": scale, "primary": primary, "better": better, "locations": loc.index.tolist(),
                    "mouse": [s.split("_")[0] for s in loc.index],
                    "recordings": own.groupby("location").recording.nunique().reindex(loc.index).tolist(),
                    "cases": own.groupby("location").size().reindex(loc.index).tolist(),
                    "scores": {m: loc[m].tolist() for m in ("demix", "suite2p")},
                    "mean": {m: float(loc[m].mean()) for m in ("demix", "suite2p")},
                    "sd": {m: float(loc[m].std(ddof=1)) for m in ("demix", "suite2p")},
                    "stats": {"suite2p": suite2p_test(loc["demix"].to_numpy(), loc["suite2p"].to_numpy(),
                                                      better)}}
    return out


def holm_with_suite2p(stats: dict[str, dict[str, Any]], s2p: dict[str, Any]) -> list[str]:
    """Holm over the declared comparisons ``stats`` with Suite2p's test ``s2p`` added (the paper's family):
    writes ``p_holm`` (over the seven) and ``p_holm_declared`` (over the six) into ``stats`` and ``p_holm``
    into ``s2p``; returns the family."""
    raw = {m: float(stats[m]["p"]) for m in stats}
    old, new = holm(raw), holm({**raw, "suite2p": float(s2p["p"])})
    for m in stats:
        stats[m]["p_holm_declared"], stats[m]["p_holm"] = old[m], new[m]
    s2p["p_holm"] = new["suite2p"]
    return [*raw, "suite2p"]


def location_values(df: pd.DataFrame, value: str,
                    methods: tuple[str, ...]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Each location's median of ``value`` over its cases per method, at the locations with a value for
    every method."""
    t = df[df.method.isin(methods)].groupby(["location", "method"])[value].median().unstack("method")
    t = t[list(methods)].dropna().sort_index()
    return list(t.index), {m: t[m].to_numpy(dtype=float) for m in methods}


def d1_secondary(tables: Mapping[str, pd.DataFrame], s2p: dict[str, Any]) -> dict[str, Any]:
    """The secondary measures' families of seven (``D1_SECONDARY``). Suite2p's test must be the one in ``s2p``
    (``d1_suite2p``) wherever that has the measure, and gets its Holm p there too."""
    inj = tables["injection"].copy()
    inj["gain_dev"] = (inj.gain - 1.0).abs()
    cases = {"injection": inj, "leakage": tables["leakage"]}
    methods = ("demix", *D1_DECLARED, "suite2p")
    out: dict[str, Any] = {}
    for name, kind, value, scale, site, better in D1_SECONDARY:
        g = cases[kind]
        g = g[(g.scale == scale) & ((g.site == site) if site else True)]
        locs, v = location_values(g, value, methods)
        tests: dict[str, dict[str, Any]] = {}
        for m in methods[1:]:
            d = v["demix"] - v[m]
            tests[m] = {"mean_diff": float(d.mean()), "n": int(d.size), "p": wilcoxon_p(d),
                        "wins": int((d < 0).sum() if better == "lower" else (d > 0).sum())}
        ref = s2p.get(name, {}).get("stats", {}).get("suite2p")
        mine = tests["suite2p"]
        if ref is not None and ((mine["wins"], mine["n"]) != (ref["wins"], ref["n"])
                                or abs(mine["p"] - ref["p"]) > 1e-12
                                or abs(mine["mean_diff"] - ref["mean_diff"]) > 1e-12):
            raise SystemExit(f"D1 {name}: Suite2p's test here is not d1_suite2p's")
        declared_holm = holm({k: tests[k]["p"] for k in D1_DECLARED})
        declared = {m: {**tests[m], "p_holm": declared_holm[m]} for m in D1_DECLARED}
        test = ref if ref is not None else mine
        family = holm_with_suite2p(declared, test)
        out[name] = {"scale": scale, "site": site, "better": better, "locations": locs, "family": family,
                     "stats": {**declared, "suite2p": {**mine, "p_holm": test["p_holm"]}}}
    return out


def d1(tables: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """D1's tables from the pooled per-case scores of every method (``injection`` and ``leakage``):
    ``d1_stats`` (the primary columns), ``suite2p`` (ASTRID against Suite2p) and ``secondary``."""
    primary = d1_primary(tables)
    s2p = d1_suite2p(tables)
    for col, *_ in COLUMNS:
        if s2p[col]["locations"] != primary[col]["locations"]:
            raise SystemExit(f"D1 {col}: Suite2p's locations are not the other methods'")
        family = holm_with_suite2p(primary[col]["stats"], s2p[col]["stats"]["suite2p"])
        primary[col]["holm"] = {"family": family, "suite2p": "d1_suite2p.json"}
    return {"d1_stats": primary, "suite2p": s2p, "secondary": d1_secondary(tables, s2p)}


def cell2(x: pd.Series) -> str:
    return f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"


def diff_cell(s: dict[str, Any], nd: int, fmt: Any = fmt_p) -> str:
    return f"{s['mean_diff']:+.{nd}f} ({s['wins']}/{s['n']}; {fmt(s['p_holm'])})".replace("-", "−")


def d1_markdown(res: dict[str, Any]) -> str:
    rec, s2p, sec = res["d1_stats"], res["suite2p"], res["secondary"]
    n = {c: len(rec[c]["locations"]) for c, *_ in COLUMNS}
    head = " | ".join(f"{c.replace('x', '×')} ({n[c]} locations)" for c, *_ in COLUMNS)
    lines = ["# D1: error and leakage over the imaging locations", "",
             "Each location's score is the median over its cases, all its recordings together: the error below "
             "0.5 Hz of the injected ROIs, or the leakage. Mean ± SD over the locations. For each comparator: "
             "ASTRID minus it (lower is better, so negative favors ASTRID), the locations where ASTRID is "
             "lower, and the p of a paired two-sided Wilcoxon signed-rank test over the locations, "
             "Holm-corrected over the seven comparisons of each column.", "",
             f"| | {head} |", "|---|" + "---|" * len(COLUMNS)]

    def row(m: str) -> list[str]:
        cells = []
        for col, *_ in COLUMNS:
            r = s2p[col] if m == "suite2p" else rec[col]
            x = pd.Series(r["scores"][m])
            cells.append(cell2(x) if m == OURS else f"{cell2(x)}: {diff_cell(r['stats'][m], 2)}")
        return cells

    for m in (OURS, *D1_DECLARED, "suite2p"):
        lines.append(f"| {NAMES[m]} | " + " | ".join(row(m)) + " |")
    lines.append(f"| {NAMES['roi']} (reference, no test) | "
                 + " | ".join(cell2(pd.Series(rec[c]["scores"]["roi"])) for c, *_ in COLUMNS) + " |")
    lines += ["", "Unrounded: ASTRID minus each comparator over the locations (p Holm: over the seven; "
              "declared: over the six before Suite2p).", "",
              "| column | comparator | mean diff | locations ASTRID lower | p | p (Holm) | p (Holm, declared) |",
              "|---|---|---|---|---|---|---|"]
    for col, *_ in COLUMNS:
        tests = {**rec[col]["stats"], "suite2p": s2p[col]["stats"]["suite2p"]}
        for m, s in tests.items():
            declared = f"{s['p_holm_declared']:.2e}" if "p_holm_declared" in s else "—"
            lines.append(f"| {col} | {NAMES[m]} | {s['mean_diff']:+.4f} | {s['wins']}/{s['n']} "
                         f"| {s['p']:.2e} | {s['p_holm']:.2e} | {declared} |")
    lines += ["", "Secondary measures (gain deviation |gain - 1|, kept correlation: higher is better, the error by "
              "injection site): ASTRID minus each comparator, the locations where ASTRID is better, Holm over "
              "the seven.", "",
              "| measure (locations) | " + " | ".join(NAMES[m] for m in (*D1_DECLARED, "suite2p")) + " |",
              "|---|" + "---|" * (len(D1_DECLARED) + 1)]
    for name, r in sec.items():
        lines.append(f"| {name} ({len(r['locations'])}) | "
                     + " | ".join(diff_cell(r["stats"][m], 3) for m in (*D1_DECLARED, "suite2p")) + " |")
    lines += ["", "Locations: " + "; ".join(
        f"{c}: {n[c]} in {len(set(rec[c]['mouse']))} mice, {sum(rec[c]['recordings'])} recordings, "
        f"{sum(rec[c]['cases'])} cases" for c, *_ in COLUMNS) + ".", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ ablations

VARIANTS = ("I", "II", "III")
ABLATION_NAMES = {"full": "Full (ASTRID)", "I": "I, no astrocyte components",
                  "II": "II, no penalty on fast fluctuations", "III": "III, no evidence map"}
MEASURE = {"gain_dev": "|gain - 1|"}
ABLATION_KEYS = {"injection": ["location", "recording", "case", "scale"],
                 "leakage": ["location", "recording", "target", "scale"]}


def ours_only(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """ASTRID's rows of a pooled run, with |gain - 1| added to the injections."""
    t = {k: v[v.method == OURS].copy() for k, v in tables.items()}
    t["injection"]["gain_dev"] = (t["injection"].gain - 1).abs()
    return t


def ablation_by_location(t: pd.DataFrame, value: str, scale: float, site: str | None) -> pd.Series:
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
    """The ablation table: ``full`` and each variant's pooled scores (``read_scores``; ASTRID's rows used)."""
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
    for col, kind, value, scale, site, better, primary in D1_MEASURES:
        loc = pd.DataFrame({m: ablation_by_location(runs[m][kind], value, scale, site) for m in methods})
        if loc.isna().any().any():
            raise SystemExit(f"{col}: a location lacks a variant's score")
        g = runs["full"][kind]
        g = g[(g.scale == scale) & ((g.site == site) if site else True)]
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
    low = {m: int((runs[m]["injection"].query("scale == 3").gain < 0.5).sum()) for m in methods}
    n3 = int((runs["full"]["injection"].scale == 3).sum())
    record["_descriptive"] = {"measure": "x3 injections with gain below 0.5", "n_cases": n3, "counts": low}
    return record


def ablation_markdown(record: dict[str, Any]) -> str:
    methods = ("full", *VARIANTS)
    lines = ["# D1 ablations: full ASTRID against each variant over the imaging locations", "",
             "Each location's score is the median over its cases; mean ± SD over the locations. Gain deviation "
             "is |gain - 1|. For each variant: Full minus it, the locations where Full is better (lower error, "
             "leakage and gain deviation; higher kept correlation), and the p of a two-sided paired Wilcoxon "
             "signed-rank test over the locations (SciPy's default method), Holm-corrected over the three variants of "
             "each column.", ""]
    for title, prim in (("Primary", True), ("Secondary", False)):
        specs = [c for c in D1_MEASURES if c[6] == prim]
        head = " | ".join(f"{c[0].replace(' x1', ' ×1').replace(' x3', ' ×3')} "
                          f"({len(record[c[0]]['locations'])})" for c in specs)
        lines += [f"{title}:", "", f"| | {head} |", "|---|" + "---|" * len(specs)]
        for m in methods:
            cells = []
            for spec in specs:
                r = record[spec[0]]
                x = pd.Series(r["scores"][m])
                cells.append(cell2(x) if m == "full" else f"{cell2(x)}: {diff_cell(r['stats'][m], 2)}")
            lines.append(f"| {ABLATION_NAMES[m]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Unrounded: Full minus each variant over the locations.", "",
              "| column | variant | mean diff | locations Full better | nonzero | p | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
    for c in D1_MEASURES:
        for v, s in record[c[0]]["stats"].items():
            lines.append(f"| {c[0]} | {v} | {s['mean_diff']:+.4f} | {s['wins']}/{s['n']} "
                         f"| {s['n_nonzero']} | {s['p']:.2e} | {s['p_holm']:.2e} |")
    d = record["_descriptive"]
    lines += ["", f"Descriptive: x3 injections with gain below 0.5, of {d['n_cases']}: "
              + ", ".join(f"{ABLATION_NAMES[m]} {d['counts'][m]}" for m in methods) + ".", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------- D6

D6_FPS = 30.9
D6_BANDS = {"r_5hz": 5.0, "r_05hz": 0.5}
D6_MOVIES = ("M", "R")                 # M: the two channels summed (one-color); R: the neuronal channel alone
COMPARED = ("tuncat", "tuncat_astro", "fissa", "cnmf", "cnmf_astro", "ring", "suite2p")   # Holm over these
# (COMPARED, METHODS and compare7 serve D6 and D7, whose families hold all seven comparators)
METHODS = (OURS, *COMPARED, *REFERENCE)
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


def d7_injection_table(run: Path, index: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per injected case and method with the failure rule applied (``status`` ok or failure)."""
    cases = [c for c in index if c["test"] == "injection"]
    if not cases:
        return pd.DataFrame()
    s = pd.read_csv(run / "scores" / "injection_scores.csv")
    rows = []
    roi = REFERENCE[0]
    for c in cases:
        g = s[s.case == c["spec"]].set_index("method")
        if roi not in g.index or not np.isfinite(g.loc[roi, list(D7_ERRORS)].to_numpy(float)).all():
            raise SystemExit(f"{run.name} {c['spec']}: no usable raw ROI row")
        for m in METHODS:
            ok = (m in g.index and np.isfinite(g.loc[m, list(D7_ERRORS)].to_numpy(float)).all()
                  and np.isfinite(float(g.loc[m, "corr"])))
            src = g.loc[m] if ok else g.loc[roi]
            rows.append({"case": c["spec"], "site": c["site"], "scale": c["scale"], "method": m,
                         "status": "ok" if ok else "failure", **{e: float(src[e]) for e in D7_ERRORS}})
    return pd.DataFrame(rows)


def d7_leakage_table(run: Path, index: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per leakage target, strength (> 0) and method, with the failure rule applied."""
    cases = [c for c in index if c["test"] == "leakage" and c["scale"] > 0]
    if not cases:
        return pd.DataFrame()
    s = pd.read_csv(run / "scores" / "leakage_scores.csv")
    rows = []
    for c in cases:
        g = s[(s.target == c["base"]) & (s.scale == c["scale"])].set_index("method")
        for m in METHODS:
            ok = (m in g.index and np.isfinite(float(g.loc[m, "leak"]))
                  and np.isfinite(float(g.loc[m, "kept_corr"])))
            rows.append({"target": c["base"], "scale": c["scale"], "method": m,
                         "status": "ok" if ok else "failure", "leak": float(g.loc[m, "leak"]) if ok else 1.0})
    return pd.DataFrame(rows)


def d7_tables(root: Path) -> dict[str, Any]:
    """D7's table: D1's four columns over the recordings, Holm over the seven comparators, and the imaging
    locations apart (no test).

    Inputs, one folder per recording under ``root``, named ``<location>_<k>`` (the location, then the
    recording's name after the last underscore): ``compare/index.json`` (the cases, ``astrid_repro.peers``'
    format), ``scores/injection_scores.csv`` (``case``, ``method``, ``corr``, ``err_05hz``, ``err_2hz``,
    ``err_raw``, ...) and ``scores/leakage_scores.csv`` (``target``, ``scale``, ``method``, ``leak``,
    ``kept_corr``, ...); a recording counts once its scorer's printout ``scores/summary.txt`` exists.

    The failure rule: a method's trace that is missing (no score row for a case of the index), not finite or
    constant (a NaN correlation) is scored as the raw ROI average: its error on that case, and leakage 1.
    A recording's score is its median over its cases."""
    inj, leak = [], []
    runs = sorted(p.parent.parent for p in root.glob("*/scores/summary.txt"))
    if not runs:
        raise SystemExit(f"no scored recording under {root}")
    for run in runs:
        location, k = run.name.rsplit("_", 1)
        index = json.loads((run / "compare" / "index.json").read_text(encoding="utf-8"))
        for parts, table in ((inj, d7_injection_table(run, index)), (leak, d7_leakage_table(run, index))):
            if len(table):
                parts.append(table.assign(location=location, recording=f"{location}/{k}"))
    tables = {"injection": pd.concat(inj, ignore_index=True), "leakage": pd.concat(leak, ignore_index=True)}
    record: dict[str, Any] = {"recordings": len(runs), "columns": {}, "failures": {}, "by_location": {}}
    for col, name, value, scale in COLUMNS:
        t = tables[name][tables[name].scale == scale]
        rec = t.groupby(["recording", "method"])[value].median().unstack()[list(METHODS)]
        where = [r.split("/")[0] for r in rec.index]
        cases = t[t.method == OURS].groupby("recording").size().reindex(rec.index)
        record["columns"][col] = {
            "measure": f"{value}, each recording's median over its cases", "test": name, "scale": scale,
            "recordings": rec.index.tolist(), "location": where, "cases": cases.tolist(),
            "scores": {m: rec[m].tolist() for m in METHODS},
            "mean": {m: float(rec[m].mean()) for m in METHODS},
            "sd": {m: float(rec[m].std(ddof=1)) for m in METHODS},
            "stats": compare7(rec)}
        if name == "injection":
            record["columns"][col]["other_bands"] = {
                e: {m: v.tolist() for m, v in
                    t.groupby(["recording", "method"])[e].median().unstack()[list(METHODS)].items()}
                for e in D7_ERRORS[1:]}
        record["by_location"][col] = {}
        for location in sorted(set(where)):
            sub = rec[[w == location for w in where]]
            if len(sub):
                record["by_location"][col][location] = {
                    "n": len(sub), "mean": {m: float(sub[m].mean()) for m in METHODS},
                    "sd": {m: float(sub[m].std(ddof=1)) if len(sub) > 1 else float("nan") for m in METHODS},
                    "wins": {m: int((sub[OURS] < sub[m]).sum()) for m in COMPARED}}
    for name, t in tables.items():
        f = t[t.status == "failure"].groupby("method").size().reindex(list(METHODS), fill_value=0)
        record["failures"][name] = {"cases": int((t.method == OURS).sum()), **{m: int(f[m]) for m in METHODS}}
    return record


def d7_markdown(record: dict[str, Any]) -> str:
    cols = record["columns"]
    head = " | ".join(f"{c.replace('x', '×')} ({len(cols[c]['recordings'])} recordings)" for c, *_ in COLUMNS)
    lines = [f"# D7: error and leakage over the recordings ({record['recordings']} recordings)", "",
             "D1's tests on D7. Error: the injected neurons' relative error below 0.5 Hz. Leakage: the share of "
             "the injected astrocyte signal that reaches the trace (0 is ideal, the raw ROI scores 1). Each "
             "recording's score is the median over its cases; mean ± SD over the recordings. For each "
             "comparator: ASTRID minus it (lower is better, so negative favors ASTRID), the recordings where "
             "ASTRID is lower, and the p of a paired two-sided Wilcoxon signed-rank test over the recordings, "
             "Holm-corrected over the seven comparisons of each column. A failed trace is scored as the raw ROI "
             "(its error, leakage 1).", "",
             f"| | {head} |", "|---|" + "---|" * len(COLUMNS)]
    for m in (OURS, *COMPARED):
        cells = []
        for col, *_ in COLUMNS:
            x = pd.Series(cols[col]["scores"][m])
            cells.append(cell3(x) if m == OURS else f"{cell3(x)}: {diff_cell(cols[col]['stats'][m], 3, fmt_p4)}")
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines.append(f"| {NAMES['roi']} (reference, no test) | "
                 + " | ".join(cell3(pd.Series(cols[c]["scores"]["roi"])) for c, *_ in COLUMNS) + " |")
    lines += ["", "Unrounded: ASTRID minus each comparator over the recordings.", "",
              "| column | comparator | mean diff | recordings ASTRID lower | p | p (Holm) |",
              "|---|---|---|---|---|---|"]
    for col, *_ in COLUMNS:
        for m, s in cols[col]["stats"].items():
            lines.append(f"| {col} | {NAMES[m]} | {s['mean_diff']:+.4f} | {s['wins']}/{s['n']} "
                         f"| {s['p']:.4f} | {s['p_holm']:.4f} |")
    lines += ["", "The imaging locations apart: mean ± SD over each location's recordings, and the recordings where "
              "ASTRID is lower; no test.", "",
              "| column | location | n | " + " | ".join(NAMES[m] for m in METHODS) + " |",
              "|---|---|---|" + "---|" * len(METHODS)]
    for col, *_ in COLUMNS:
        for location, c in record["by_location"][col].items():
            vals = [f"{c['mean'][m]:.3f} ± {c['sd'][m]:.3f}"
                    + ("" if m in (OURS, *REFERENCE) else f" ({c['wins'][m]}/{c['n']})") for m in METHODS]
            lines.append(f"| {col} | {location} | {c['n']} | " + " | ".join(vals) + " |")
    for name, f in record["failures"].items():
        lines += ["", f"Failures ({name}, {f['cases']} cases, scored as the raw ROI): "
                  + ", ".join(f"{NAMES[m]} {f[m]}" for m in METHODS if m not in REFERENCE) + "."]
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ D7's error split

SPLIT_METHODS = ("demix", "ring", "fissa", "tuncat", "tuncat_astro", "cnmf", "cnmf_astro", "suite2p")
SPLIT_VALUES = ("err05", "gain_part", "resid_part", "g05", "g_raw", "dev05", "dev_raw")
SPLIT_BAND_HZ = 0.5                    # the error's band
LOW_GAIN = 0.5                         # x3 cases with a gain below this are counted


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
    """One recording's error split (``error_split``), one row per injected case and method of
    ``SPLIT_METHODS``, with D7's failure rule (``d7_injection_table``): a failed method takes the raw ROI
    average's values (``failed`` True). ``g_raw_as_is`` and ``g05_as_is`` are the method's own gains whatever
    its status (NaN without a trace), from which the gains below ``LOW_GAIN`` are counted; ``dev_raw`` and
    ``dev05`` are |g - 1|. Inputs: a D7 run folder (``astrid_repro.d7``: ``run.json``, ``compare/``,
    ``injection/``, ``scores/``) and the folders of the peers' results (default ``<run>/compare/results``)."""
    from astrid_repro.cases import case_traces

    fps = float(json.loads((run / "run.json").read_text(encoding="utf-8"))["fps"])
    index = json.loads((run / "compare" / "index.json").read_text(encoding="utf-8"))
    status = d7_injection_table(run, index)
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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="d7")
    p = sub.add_parser("d7-split", help="D7's error split (Supplementary Equation S24)")
    p.add_argument("--root", type=Path, required=True, help="the recordings' folders (see d7_tables)")
    p.add_argument("--results", nargs="+", default=None,
                   help="the peers' results folders, relative to each recording's folder (default compare/results)")
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")         # the tables' ± and − on a Windows console
    if a.command == "d1":
        res = d1(read_scores(a.scores, a.suffix))
        write_json(a.out / "d1_stats.json", res["d1_stats"])
        write_json(a.out / "d1_suite2p.json", res["suite2p"])
        write_json(a.out / "d1_secondary.json", res["secondary"])
        text = d1_markdown(res)
        (a.out / "d1_stats.md").write_text(text, encoding="utf-8")
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
    else:
        record = d7_tables(a.root)
        write_json(a.out / f"{a.name}.json", record)
        text = d7_markdown(record)
        (a.out / f"{a.name}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
