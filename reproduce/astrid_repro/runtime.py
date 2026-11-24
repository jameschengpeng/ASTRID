"""The runtime table (Supplementary Table S13), rebuilt from each dataset's per-unit times. It does not measure
times: the times are those of the paper's runs (Supplementary Section 4.7: every method by its own timer, reading
the movie from disk not counted except within CNMF's timer, on the hardware described there), given as input files.

**Input files** (``--times``, one folder):
- ``runtime_units.csv``, one row per dataset, unit and method: ``dataset`` (D1-D7), ``unit_kind`` (``location``,
  ``video``, ``cell`` or ``recording``), ``unit``, ``method`` (``demix`` for ASTRID, ``tuncat``, ``fissa``,
  ``cnmf``, ``tuncat_astro`` and ``cnmf_astro`` (+ territory), ``suite2p``) and ``seconds``, the unit's time per
  video: a video's time on D2, D4 and D5 and a recording's on D7, the mean over its recordings for a cell (D3)
  and an imaging location (D1, D6). Each method of a dataset has a time for every unit of the dataset.
- ``runtime_videos.csv``, one row per video timed: ``dataset``, ``video`` and ``rois``, its number of ROIs.

**The table**, per dataset: the units, the videos timed and their ROIs (the median and range); per method the
mean ± SD (ddof 1) over the units; for each other method, ASTRID minus it over the units, the units where ASTRID is
faster, and the p of a two-sided paired Wilcoxon signed-rank test (``metrics.wilcoxon_p``), Holm-corrected over the
dataset's other methods (``metrics.holm``: five and Suite2p on D1, D6 and D7, three and Suite2p on D2-D5); and
the fastest method (the lowest mean). Writes ``runtime.json`` and ``runtime.md``.

    python -m astrid_repro.runtime --times <folder> --out <tables>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from astrid_repro.metrics import holm, wilcoxon_p

DATASETS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")
METHODS = ("demix", "tuncat", "fissa", "cnmf", "tuncat_astro", "cnmf_astro", "suite2p")   # the table's order
OURS = "demix"
NAMES = {"demix": "ASTRID (ours)", "tuncat": "TUnCaT", "fissa": "FISSA", "cnmf": "CNMF",
         "tuncat_astro": "TUnCaT + territory", "cnmf_astro": "CNMF + territory", "suite2p": "Suite2p"}
UNIT_NAMES = {"location": "locations", "video": "videos", "cell": "cells", "recording": "recordings"}


def read_times(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two input files (module docstring), checked: known datasets and methods, one kind of unit per
    dataset, and every method with a time for every unit of its dataset."""
    folder = Path(folder)
    units = pd.read_csv(folder / "runtime_units.csv", dtype={"unit": str}, float_precision="round_trip")
    videos = pd.read_csv(folder / "runtime_videos.csv", dtype={"video": str})
    if not set(units.dataset) <= set(DATASETS) or not set(units.method) <= set(METHODS):
        raise ValueError(f"{folder}: unknown datasets or methods "
                         f"{sorted(set(units.dataset) - set(DATASETS))} {sorted(set(units.method) - set(METHODS))}")
    for ds, g in units.groupby("dataset"):
        if g.unit_kind.nunique() != 1 or OURS not in set(g.method):
            raise ValueError(f"{ds}: one kind of unit and ASTRID's times are needed")
        if g.duplicated(["unit", "method"]).any():
            raise ValueError(f"{ds}: a unit has two times for one method")
        wide = g.pivot(index="unit", columns="method", values="seconds")
        if wide.isna().any().any():
            raise ValueError(f"{ds}: a method lacks the time of some unit")
    return units, videos


def dataset_table(units: pd.DataFrame, videos: pd.DataFrame) -> dict[str, Any]:
    """One dataset's column (module docstring)."""
    wide = units.pivot(index="unit", columns="method", values="seconds").loc[pd.unique(units.unit)]   # file order
    methods = [m for m in METHODS if m in wide.columns]
    summary = {m: {"n": int(wide[m].size), "mean": float(wide[m].mean()), "sd": float(wide[m].std(ddof=1))}
               for m in methods}
    tests: dict[str, dict[str, Any]] = {}
    raw = {}
    for m in methods:
        if m == OURS:
            continue
        d = (wide[OURS] - wide[m]).to_numpy()
        raw[m] = wilcoxon_p(d)
        tests[m] = {"mean_diff": float(d.mean()), "faster": int((d < 0).sum()), "n": int(d.size), "p": raw[m]}
    for m, ph in holm(raw).items():
        tests[m]["p_holm"] = ph
    rois = videos.rois.to_numpy(float)
    return {"unit": str(units.unit_kind.iloc[0]), "units": int(wide.shape[0]), "videos": int(len(videos)),
            "rois": {"median": float(np.median(rois)), "min": float(rois.min()), "max": float(rois.max())},
            "summary": summary, "tests": tests, "fastest": min(methods, key=lambda m: summary[m]["mean"])}


def runtime_table(folder: str | Path) -> dict[str, Any]:
    """Every dataset's column of the runtime table from the input files in ``folder``."""
    units, videos = read_times(folder)
    return {ds: dataset_table(units[units.dataset == ds], videos[videos.dataset == ds])
            for ds in DATASETS if ds in set(units.dataset)}


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.4f}"


def markdown(record: dict[str, Any]) -> str:
    cols = list(record)
    lines = ["# Runtime in seconds per video (Supplementary Table S13)", "",
             "Mean ± SD over the units of each dataset (the time of a cell or an imaging location is the mean over "
             "its recordings). Below each other method: the units where ASTRID is faster and the Holm-corrected p "
             "of a paired two-sided Wilcoxon signed-rank test against ASTRID. The fastest method is in bold. Rebuilt "
             "from the times of the paper's runs; nothing is timed here.", "",
             "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols),
             "| Units | " + " | ".join(f"{record[d]['units']} {UNIT_NAMES[record[d]['unit']]}" for d in cols) + " |",
             "| Videos timed | " + " | ".join(str(record[d]["videos"]) for d in cols) + " |"]
    rois = []
    for d in cols:
        r = record[d]["rois"]
        rng = f" ({r['min']:.0f}-{r['max']:.0f})" if r["min"] != r["max"] else ""
        rois.append(f"{r['median']:.0f}{rng}")
    lines.append("| ROIs per video, median (range) | " + " | ".join(rois) + " |")
    for m in METHODS:
        cells = []
        for d in cols:
            s = record[d]["summary"].get(m)
            if s is None:
                cells.append("-")
                continue
            c = f"{s['mean']:.1f} ± {s['sd']:.1f}"
            c = f"**{c}**" if record[d]["fastest"] == m else c
            t = record[d]["tests"].get(m)
            cells.append(c if t is None else f"{c} ({t['faster']}/{t['n']}; {fmt_p(t['p_holm'])})")
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--times", type=Path, required=True, help="the folder of the two input files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    record = runtime_table(a.times)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "runtime.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    text = markdown(record)
    (a.out / "runtime.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
