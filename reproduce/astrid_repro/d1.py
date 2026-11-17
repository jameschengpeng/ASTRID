"""D1's injection and leakage tests from its input files: their format, the reader, and the runner of one
recording's cases (the rules, the patches, the fits, the export and the scores are ``astrid_repro.cases``).

**Input files**, one folder per imaging location, ``<data>/<location>/``. Images are (H, W) on the movies' pixel
grid and in their orientation.
- ``location.npz``:
  - ``neurons`` (int): the ROIs as labels 1..n; ROI v is row v-1 of every recording's ``dff``;
  - ``candidates`` (int): the detection's candidate map, labels of its 4-connected pieces (0: none);
  - ``events`` (int): the number of AQuA2 events that touch each pixel, over all recordings of the location;
    the astrocyte territory is where it is at least 2;
  - ``tau`` (float64): the tagRFP evidence in [0, 1];
  - ``max_proj`` (float32): the mean over the location's recordings of their maximum projections, from which a
    source's footprint is taken (in float32);
  - ``astro_label`` (int): the astrocyte ROIs as labels 1..m (0: none); astrocyte ROI k is row k-1 of every
    recording's ``astro_dff``;
  - ``fps`` (Hz), ``um_per_px`` (um per pixel), ``border_px`` (the registration border: pixels at the edge
    that no ROI or patch uses);
  - ``recordings``: the location's recordings, over which ``events``, ``tau`` and ``max_proj`` were pooled.
- ``<recording>.h5``, one per recording:
  - ``mov`` (T, H, W) float32: the registered movie, downsampled 2x in space, that every method read;
  - ``dff`` (n, T) float64: each ROI's ΔF/F0 from the detection step (its average and its ring's low-passed
    below 5 Hz, smoothed by a Gaussian of s.d. 6 frames and reduced by their moving 6-s baselines; the ROI's
    part minus 0.7 times the ring's, over the ROI's baseline), which ranks the targets and the sources and is
    the injected trace;
  - ``astro_dff`` (m, T) float64: each astrocyte ROI's ΔF/F0 (as ``dff``, without the ring and with a 60-s
    baseline), the added astrocyte's trace.

**Settings** from ``reproduce/datasets.yaml`` (D1): the time constants, the defaults otherwise, at most 2,000
iterations in the leakage test (``leakage_test_settings``); the location's frame rate, pixel size and border
must be D1's there. ``--variant`` runs one of the paper's ablations: A without astrocyte components
(``astro_components=False``), B with kappa_a 0, C with kappa_rho 0.

**Outputs** of one recording, in ``<out>/<location>_<recording>/``:
- ``run.json``: the location, the recording, the settings and the choices (the sources, the sites, the leakage
  targets, the astrocyte and its shifts);
- ``injection/`` and ``leakage/``: ASTRID's outputs per case (``<case>.npz``, ``astrid_repro.cases``) and
  ``fits.csv`` (stop, iterations, convergence, patch size and time of each fit);
- ``compare/``: the cases in the format of ``astrid_repro.peers`` (``index.json``), which the peer methods read
  (``python -m astrid_repro.peers launch --cases <run>/compare ...``, results in ``compare/results``;
  ``suite2p_runner cases``, with results in a folder of their own);
- ``scores/``: ``injection_scores.csv`` and ``leakage_scores.csv`` (``cases.injection_scores``,
  ``cases.leakage_scores``), every method whose traces are there, and ``summary.txt``
(``cases.run_tests``).

Run from ``reproduce/`` with one BLAS thread (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1``
exported, as in the paper's runs: the last digits of the fits and the scores depend on the BLAS's summation
order):
    python -m astrid_repro.d1 run --data <inputs> --location <location> --recording <recording> --out <root>
        [--tests injection leakage] [--workers 16] [--variant A|B|C [--cases <the full model's run>/compare]]
    python -m astrid_repro.d1 score --run <root>/<location>_<recording> [--results <dir> ...]
    python -m astrid_repro.d1 pool --root <root> --out <scores>
``score`` scores again once the peer methods have run (``--results``: their results folders, e.g.
``compare/results`` and Suite2p's, a later one's trace replacing an earlier one's; default ``compare/results``).
``pool`` writes every recording's scores into ``injection_scores.csv`` and ``leakage_scores.csv`` with
``location`` and ``recording``, the input of ``python -m astrid_repro.stats d1`` (and, for a variant's root, of
``stats ablations``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml

import astrid
from astrid.api import Movie
from astrid.patch import REFERENCE_UM_PER_PX
from astrid_repro import cases as C

CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"
LOCATION_FILE = "location.npz"
VARIANTS: dict[str, tuple[dict[str, float], bool]] = {   # (settings changed, astrocyte components)
    "A": ({}, False), "B": ({"kappa_a": 0.0}, True), "C": ({"kappa_rho": 0.0}, True)}
BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# --------------------------------------------------------------------------------------------- the inputs
def read_location(folder: str | Path) -> C.Location:
    """The location in ``<folder>/location.npz`` (module docstring)."""
    folder = Path(folder)
    with np.load(folder / LOCATION_FILE) as z:
        a = {k: z[k] for k in z.files}
    loc = C.location(folder.name, a["neurons"], a["candidates"], a["events"], a["tau"],
                     np.asarray(a["max_proj"], dtype=np.float32), a["astro_label"], float(a["fps"]),
                     float(a["um_per_px"]), int(a["border_px"]))
    return loc


@dataclass
class Recording:
    """One recording's file: its movie (lazily read), the ROIs' and the astrocyte ROIs' ΔF/F0."""
    name: str
    path: Path
    frames: int
    dff: np.ndarray
    astro_dff: np.ndarray

    def movie(self) -> Movie:
        return Movie(self.name, path=self.path, key="mov")


def read_recording(folder: str | Path, name: str, loc: C.Location) -> Recording:
    """``<folder>/<name>.h5`` (module docstring), checked against the location."""
    path = Path(folder) / f"{name}.h5"
    with h5py.File(path, "r") as f:
        shape = tuple(int(x) for x in f["mov"].shape)
        dtype = f["mov"].dtype
        dff = np.asarray(f["dff"][()], dtype=np.float64)
        astro = np.asarray(f["astro_dff"][()], dtype=np.float64)
    n, m = int(loc.neurons.max()), int(loc.astro_label.max())
    if len(shape) != 3 or shape[1:] != loc.shape or dtype != np.float32:
        raise ValueError(f"{path}: mov must be (T, H, W) float32 on the location's {loc.shape} grid, got "
                         f"{shape} {dtype}")
    if dff.shape != (n, shape[0]):
        raise ValueError(f"{path}: dff is {dff.shape}, expected {(n, shape[0])} (ROIs x frames)")
    if astro.shape != (m, shape[0]):
        raise ValueError(f"{path}: astro_dff is {astro.shape}, expected {(m, shape[0])}")
    return Recording(name, path, shape[0], dff, astro)


def d1_entry(config: str | Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))["datasets"]["D1"]


def d1_settings(loc: C.Location, variant: str | None = None,
                config: str | Path = CONFIG) -> tuple[dict[str, astrid.Settings], bool]:
    """ASTRID's settings per test from ``datasets.yaml`` (with a variant's change), and whether the patches
    have astrocyte components. Refuses a location whose frame rate, pixel size or border is not D1's."""
    e = d1_entry(config)
    fld = loc.field
    if (float(e["fps"]), REFERENCE_UM_PER_PX / float(e["um_per_px"]), int(e["border_px"])) != (
            fld.fps, fld.px_scale, fld.border_px):
        raise ValueError(f"{fld.name}: fps {fld.fps}, {REFERENCE_UM_PER_PX / fld.px_scale:g} um per pixel and "
                         f"border {fld.border_px} are not D1's ({e['fps']}, {e['um_per_px']}, {e['border_px']})")
    change, astro = VARIANTS[variant] if variant else ({}, True)
    base = astrid.Settings(tau_r=float(e["tau_r"]), tau_d=float(e["tau_d"]), **{**(e.get("settings") or {}), **change})
    leak = dataclasses.replace(base, **(e.get("leakage_test_settings") or {}))
    return {"injection": base, "leakage": leak}, astro


# --------------------------------------------------------------------------------------------- one recording
def build_cases(loc: C.Location, rec: Recording,
                tests: Sequence[str] = C.TESTS) -> tuple[dict[str, list[C.Case]], dict[str, Any]]:
    """The cases of each requested test and the choices made. The leakage test's astrocyte is the location's
    largest astrocyte ROI (``cases.largest_astrocyte``); a location without one has no leakage test."""
    dff = {rec.name: rec.dff}
    out: dict[str, list[C.Case]] = {}
    choices: dict[str, Any] = {}
    if "injection" in tests:
        out["injection"], choices["injection"] = C.injection_cases(loc, dff)
    if "leakage" in tests:
        src = C.largest_astrocyte(loc.astro_label)
        if src > 0:
            out["leakage"], choices["leakage"] = C.leakage_cases(loc, dff, {rec.name: rec.astro_dff[src - 1]}, src)
        else:
            choices["leakage"] = {"targets": [], "astro_source": None, "astro_shifts": {}}
    return out, choices


def run_recording(data: str | Path, location_name: str, recording: str, out: str | Path, *,
                  tests: Sequence[str] = C.TESTS, workers: int = 0, variant: str | None = None,
                  cases_dir: str | Path | None = None, config: str | Path = CONFIG,
                  log: Callable[[str], Any] = C.say) -> Path:
    """Build, fit, export and score one recording's cases (module docstring); returns its run folder. With
    ``cases_dir`` (an earlier export of the same recording's cases), the cases are scored against it and not
    exported again."""
    folder = Path(data) / location_name
    loc = read_location(folder)
    rec = read_recording(folder, recording, loc)
    settings, astro = d1_settings(loc, variant, config)
    built, choices = build_cases(loc, rec, tests)
    log(f"{location_name} {recording}: {({t: len(c) for t, c in built.items()})} cases; {choices}")
    record = {"location": location_name, "recording": recording, "frames": rec.frames, "variant": variant,
              "choices": choices}
    return C.run_tests(loc, rec.movie(), built, settings, Path(out) / f"{location_name}_{recording}", record,
                       astro_components=astro, workers=workers,
                       cases_dir=None if cases_dir is None else Path(cases_dir), log=log)


def pool(root: str | Path, out: str | Path) -> dict[str, pd.DataFrame]:
    """Every run folder's scores under ``root`` (a folder with ``run.json`` and ``scores``), concatenated in the
    folders' name order with ``location`` and ``recording`` columns, written to ``out`` as
    ``injection_scores.csv`` and ``leakage_scores.csv`` where there are such scores (the input of
    ``stats.py d1``)."""
    parts: dict[str, list[pd.DataFrame]] = {"injection": [], "leakage": []}
    for info_path in sorted(Path(root).glob("*/run.json")):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for test in parts:
            p = info_path.parent / "scores" / f"{test}_scores.csv"
            if p.exists():
                parts[test].append(pd.read_csv(p).assign(location=info["location"], recording=info["recording"]))
    if not any(parts.values()):
        raise SystemExit(f"no scores under {root}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {test: pd.concat(ps, ignore_index=True) for test, ps in parts.items() if ps}
    for test, t in tables.items():
        t.to_csv(out / f"{test}_scores.csv", index=False)
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="build, fit, export and score one recording's cases")
    r.add_argument("--data", type=Path, required=True, help="the input files, one folder per location")
    r.add_argument("--location", required=True)
    r.add_argument("--recording", required=True)
    r.add_argument("--out", type=Path, required=True, help="the root of the run folders")
    r.add_argument("--tests", nargs="+", choices=C.TESTS, default=list(C.TESTS))
    r.add_argument("--workers", type=int, default=0, help="cases fitted at a time (0: in this process)")
    r.add_argument("--variant", choices=list(VARIANTS), default=None, help="one of the paper's ablations")
    r.add_argument("--cases", type=Path, default=None,
                   help="score against this export of the same recording (e.g. the full model's) instead of "
                        "exporting the cases again")
    s = sub.add_parser("score", help="score a run folder again (after the peer methods)")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--cases", type=Path, default=None, help="the export (default <run>/compare)")
    s.add_argument("--results", type=Path, nargs="+", default=None,
                   help="the peers' results folders (default <cases>/results)")
    p = sub.add_parser("pool", help="pool every recording's scores for stats.py d1")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == "run":
        for var in BLAS_THREADS:                       # one BLAS thread in each worker process
            os.environ[var] = "1"
        run_recording(a.data, a.location, a.recording, a.out, tests=a.tests, workers=a.workers, variant=a.variant,
                      cases_dir=a.cases)
    elif a.command == "score":
        C.score_run(a.run, a.cases, a.results)
    else:
        tables = pool(a.root, a.out)
        print({k: len(v) for k, v in tables.items()}, "rows written to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
