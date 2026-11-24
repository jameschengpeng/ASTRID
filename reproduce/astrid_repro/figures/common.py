"""What the data panels share: each method's name and color, their order, the marks of the tests, the plot of
the scores per unit, and saving a panel with a record of what it shows.

The panels draw the paper's data as the paper's figures do: the same values, examples, windows, marks and
colors. The paper's typography, panel sizes and page layout are not part of this code: each panel is drawn
with matplotlib's defaults and saved on its own.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from astrid_repro.metrics import holm

# key -> (name in the figures, color): one color per method in every figure (Okabe-Ito, safe for color-blind
# readers). The keys are those of the tables (tables_public).
METHODS: dict[str, tuple[str, str]] = {
    "ours": ("ASTRID", "#D55E00"),
    "TUnCaT": ("TUnCaT", "#0072B2"),
    "FISSA": ("FISSA", "#009E73"),
    "CNMF": ("CNMF", "#CC79A7"),
    "ring": ("Ring", "#7F7F7F"),
    "suite2p": ("Suite2p", "#E69F00"),
}
LONG = {"ours": "ASTRID", "TUnCaT": "TUnCaT", "FISSA": "FISSA", "CNMF": "CNMF", "ring": "the ring subtraction",
        "suite2p": "Suite2p"}
# the methods of the public datasets' figures (Fig. 5 and the supplementary figure of D2, D4 and D5), in their order
PUBLIC_ORDER = ("ours", "TUnCaT", "FISSA", "CNMF", "ring", "suite2p")
# each comparison's name in the tables' tests, and the Holm family of D2-D5 (five comparisons)
TEST_KEYS = {"TUnCaT": "TUnCaT", "FISSA": "FISSA", "CNMF": "CNMF", "ring": "Ring", "suite2p": "Suite2p"}
PUBLIC_FAMILY = ("TUnCaT", "FISSA", "CNMF", "Ring", "Suite2p")
EXAMPLE_COLOR = "#F0E442"            # the example ROI's outline, Okabe-Ito yellow on the gray image
UNIT_LINE = "#D0D0D0"                # the gray lines joining a unit's dots


def label(key: str) -> str:
    return METHODS[key][0]


def color(key: str) -> str:
    return METHODS[key][1]


def stars(p: float) -> str:
    """The mark of a comparison, from its Holm-corrected p: n.s. (p >= 0.05), * (< 0.05), ** (< 0.01) or ***
    (< 0.001)."""
    for cut, mark in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return mark
    return "n.s."


def check_holm(tests: dict[str, dict[str, Any]], where: str) -> None:
    """The Holm-corrected p of one family (``tests``: name -> {"p", "p_holm", ...}) must be Holm's correction of
    its raw p."""
    expected = holm({k: float(t["p"]) for k, t in tests.items()})
    worst = max(abs(expected[k] - float(t["p_holm"])) for k, t in tests.items())
    if worst > 1e-12:
        raise ValueError(f"{where}: the corrected p are not Holm's correction of the raw p (by {worst:g})")


def _round_ticks(lo: float, hi: float) -> list[float]:
    from matplotlib.ticker import MaxNLocator
    return [float(t) for t in MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)]


def paired_dots(ax: Any, values: dict[str, Sequence[float]], order: Iterable[str], marks: dict[str, str],
                ylabel: str, xlabel: str = "", *, markers: Sequence[str] | None = None,
                marker_labels: dict[str, str] | None = None) -> dict[str, Any]:
    """The scores of one measure: one dot per unit (video or cell) for each method of ``order``, the units
    joined by gray lines across the methods, the mean as a black bar, and above each method after the first
    (ASTRID) the mark of its comparison with ASTRID (``marks``). ``markers``: one marker per unit (D3: the
    indicator of each cell), explained by ``marker_labels`` (marker -> text). The y axis starts at a round value
    below the data. Returns the values drawn."""
    from matplotlib.lines import Line2D

    keys = list(order)
    data = np.array([np.asarray(values[k], dtype=float) for k in keys])      # (methods, units)
    xs = np.arange(len(keys), dtype=float)
    units = data.shape[1]
    kinds = list(markers) if markers else ["o"] * units
    for unit in data.T:
        ax.plot(xs, unit, color=UNIT_LINE, lw=0.6, zorder=1)
    for i, k in enumerate(keys):
        for m in dict.fromkeys(kinds):
            idx = [u for u in range(units) if kinds[u] == m]
            ax.scatter(np.full(len(idx), xs[i]), data[i, idx], s=26 if m == "^" else 20, marker=m, color=color(k),
                       linewidths=0, zorder=2)
        ax.plot([xs[i] - 0.28, xs[i] + 0.28], [np.nanmean(data[i])] * 2, color="black", lw=1.5, zorder=3)
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    ticks = _round_ticks(lo, hi)
    below = [t for t in ticks if t <= lo] or [t for t in ticks if t <= lo + 1e-9 * max(hi - lo, 1e-12)]
    bottom = max(below) if below else min(ticks)         # a tick at the lowest value can round just above it
    span = hi - bottom
    for i, k in enumerate(keys[1:], start=1):
        ax.text(xs[i], hi + 0.04 * span, marks[k], ha="center", va="bottom")
    ax.set_xticks(xs, [label(k) for k in keys])
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(bottom, hi + 0.16 * span)
    ticks = [t for t in _round_ticks(bottom, hi) if bottom - 1e-9 <= t <= hi + 1e-4 * span]
    ax.set_yticks(ticks)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(bottom, max(ticks[-1], hi))
    ax.set_ylabel(ylabel)
    if xlabel:                                   # what the plot shows goes below it
        ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", length=0)
    if marker_labels:
        handles = [Line2D([], [], ls="", marker=m, color="#808080", label=t) for m, t in marker_labels.items()]
        ax.legend(handles=handles, loc="lower right", frameon=False)
    return {"methods": keys, "values": {k: data[i].tolist() for i, k in enumerate(keys)},
            "means": {k: float(np.nanmean(data[i])) for i, k in enumerate(keys)},
            "marks": {k: marks[k] for k in keys[1:]}}


def trace_row_style(ax: Any, name: str, first_column: bool, last_row: bool, name_color: str | None) -> None:
    """One row of a stack of traces: no y axis, the method's name on the left (in ``name_color``, black if
    None), the x axis on the last row only."""
    ax.set_yticks([])
    ax.spines[["left", "top", "right"]].set_visible(False)
    if not last_row:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    if first_column:
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=2,
                      color=name_color if name_color else "black")


def score_label(ax: Any, text: str) -> None:
    """A row's score, on the right of its trace."""
    ax.text(1.02, 0.5, text, transform=ax.transAxes, ha="left", va="center")


def column_footer(ax: Any, text: str, **kw: Any) -> None:
    """The name of the score column, below the last row's score."""
    ax.text(1.02, 0.22, text, transform=ax.transAxes, ha="left", va="top", **kw)


def callable_from(spec: str | None) -> Any:
    """``module:function`` -> the function (None for None)."""
    if not spec:
        return None
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return x.name
    return x


def save(fig: Any, out: Path, stem: str, record: dict[str, Any] | None = None,
         formats: Iterable[str] = ("png", "svg"), dpi: int = 300) -> list[Path]:
    """Write the panel as ``<out>/<stem>.<format>`` for each format and, with ``record``, what it shows as
    ``<stem>.json``; closes the figure."""
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out / f"{stem}.{f}"
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    if record is not None:
        p = out / f"{stem}.json"
        p.write_text(json.dumps(jsonable(record), indent=1, ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    plt.close(fig)
    return paths


Panel = tuple[str, Any, dict[str, Any]]          # (file stem, matplotlib figure, record)
PanelMaker = Callable[..., list[Panel]]
