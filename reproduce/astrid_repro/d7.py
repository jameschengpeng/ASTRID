"""D7's injection and leakage tests from its input files: their format, the reader, the ΔF/F0 that the tests copy,
and the runner of one recording's cases (``astrid_repro.cases`` with ``cases.D7_RULES``).

**Input files**, one folder per imaging location, ``<data>/<location>/``, named as the location's entry of
``reproduce/datasets.yaml`` (``D7_location1``, ``D7_location2``). Images are (H, W) on the movies' 2.07-um grid
and in their orientation.
- ``location.npz``:
  - ``neurons`` (int): the ROIs, the neuronal nuclei marked by tagRFP, as labels 1..n;
  - ``fps`` (Hz), ``um_per_px`` (2.07) and ``border_px`` (the pixels at the edge that no ROI or patch uses);
  - ``rate_pct_per_s``: the location's bleaching rate, percent per second (negative: the movie dims);
  - ``tau_r`` and ``tau_d``: the neuronal time constants estimated from the location's recordings (s);
  - ``recordings``: the location's recordings.
  These must be the location's entry of datasets.yaml, which sets ASTRID.
- ``<recording>.h5``, one per recording:
  - ``mov`` (T, H, W) float32: the registered green movie that every method read;
  - ``events`` (int): the number of the recording's AQuA2 events (D7's settings, ``aqua2_settings``) that touch
    each pixel; the recording's territory is where it is at least 1;
  - ``largest_event`` (bool): the footprint of the recording's largest AQuA2 event, the leakage test's astrocyte;
  - ``max_proj`` (float32): the recording's maximum projection, from which a source's footprint is taken (in
    float64).

**What the builder computes from the movie** (``roi_dff``, ``astro_dff``), with D7's baseline, the location's
``astrid.patch.BaselineRule`` (the 10th percentile of a trace's 3-s bin means along the line of the bleaching
rate, times that line; ``baseline``):
- each ROI's ΔF/F0, which ranks the targets and the sources and is the injected trace: F, the ROI's mean trace
  minus 0.7 times its ring's (``ring_mask``: the pixels more than 4 and at most 16 um from the ROI that belong
  to no ROI, not cut at the border), minus its baseline, over the baseline of the ROI's mean trace; then
  low-passed below 5 Hz and smoothed by a Gaussian of ``SMOOTH_S`` seconds;
- the largest event's ΔF/F0, the added astrocyte's trace: the mean trace of its footprint over its baseline,
  minus 1, low-passed below 5 Hz.

**The tests:** ``cases.D7_RULES`` (no minimum ROI sizes; territory sites are territory pixels; no evidence map),
ASTRID's settings from the location's entry of datasets.yaml (its kernel, D7's baseline, at most 2,000 iterations
in the leakage test). A recording without a ROI near its territory has no leakage case, and one without an event
has no leakage test.

**Outputs** of one recording in ``<out>/<location>_<recording>/``, as ``astrid_repro.d1``'s (``cases.run_tests``):
``run.json``, ``injection/``, ``leakage/``, ``compare/`` (the peers' cases) and ``scores/`` (and
``scores_without_added/`` after ``score --baseline without_added``), the layout that
``python -m astrid_repro.stats d7 --root <out> [--baseline without_added]`` reads.

Run from ``reproduce/`` with one BLAS thread, as for D1:
    python -m astrid_repro.d7 run --data <inputs> --location D7_location1 --recording <recording> --out <root>
        [--tests injection leakage] [--workers 8]
    python -m astrid_repro.d7 score --run <root>/D7_location1_<recording> [--results <dir> ...]
        [--baseline reported|without_added]
With ``--baseline without_added`` (the paper's tables) the injection test is scored with every other trace's
running baseline removed by the recording's rule (D7's bleaching line), taken from the trace without the added
signal (``cases`` module docstring).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from scipy import ndimage as ndi

import astrid
from astrid.api import Movie
from astrid.patch import RING_INNER, RING_OUTER, BaselineRule, baseline, scaled
from astrid_repro import cases as C
from astrid_repro.metrics import lowpass_fourier

CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"
LOCATION_FILE = "location.npz"
TERRITORY_MIN_EVENTS = 1            # a recording's territory: at least 1 of its events
NEUROPIL_R = 0.7
LOWPASS_HZ = 5.0
SMOOTH_S = 6 / 30.9                 # the s.d. of the Gaussian smoothing of the ROIs' ΔF/F0, s
BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# --------------------------------------------------------------------------------------------- the inputs
@dataclass
class LocationInfo:
    """``location.npz`` (module docstring)."""
    name: str
    neurons: np.ndarray
    fps: float
    um_per_px: float
    border_px: int
    rate_pct_per_s: float
    tau_r: float
    tau_d: float
    recordings: list[str]


def read_location(folder: str | Path) -> LocationInfo:
    folder = Path(folder)
    with np.load(folder / LOCATION_FILE) as z:
        neurons = np.asarray(z["neurons"])
        if neurons.ndim != 2 or not np.issubdtype(neurons.dtype, np.integer) or neurons.min() < 0:
            raise ValueError(f"{folder}: neurons must be (H, W) integer labels >= 0")
        return LocationInfo(folder.name, neurons, float(z["fps"]), float(z["um_per_px"]), int(z["border_px"]),
                            float(z["rate_pct_per_s"]), float(z["tau_r"]), float(z["tau_d"]),
                            [str(r) for r in z["recordings"]])


@dataclass
class Recording:
    """One recording's file: its movie (lazily read), its events, its largest event and its projection."""
    name: str
    path: Path
    frames: int
    events: np.ndarray
    largest_event: np.ndarray
    max_proj: np.ndarray

    def movie(self) -> Movie:
        return Movie(self.name, path=self.path, key="mov")


def read_recording(folder: str | Path, name: str, info: LocationInfo) -> Recording:
    """``<folder>/<name>.h5`` (module docstring), checked against the location."""
    path = Path(folder) / f"{name}.h5"
    with h5py.File(path, "r") as f:
        shape = tuple(int(x) for x in f["mov"].shape)
        dtype = f["mov"].dtype
        events = np.asarray(f["events"][()])
        largest = np.asarray(f["largest_event"][()], dtype=bool)
        max_proj = np.asarray(f["max_proj"][()])
    grid = info.neurons.shape
    if len(shape) != 3 or shape[1:] != grid or dtype != np.float32:
        raise ValueError(f"{path}: mov must be (T, H, W) float32 on the location's {grid} grid, got {shape} {dtype}")
    for what, a in (("events", events), ("largest_event", largest), ("max_proj", max_proj)):
        if a.shape != grid:
            raise ValueError(f"{path}: {what} is {a.shape}, the location's grid {grid}")
    return Recording(name, path, shape[0], events, largest, max_proj)


def d7_entry(location: str, config: str | Path = CONFIG) -> dict[str, Any]:
    entries = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["datasets"]
    if location not in entries or not location.startswith("D7"):
        raise ValueError(f"{location} is not a D7 location of {config} "
                         f"({', '.join(k for k in entries if k.startswith('D7'))})")
    return entries[location]


def d7_settings(info: LocationInfo, config: str | Path = CONFIG) -> dict[str, astrid.Settings]:
    """ASTRID's settings per test from the location's entry of ``datasets.yaml``; refuses a location file whose
    frame rate, pixel size, border, bleaching rate or time constants are not the entry's."""
    e = d7_entry(info.name, config)
    s = astrid.Settings(tau_r=float(e["tau_r"]), tau_d=float(e["tau_d"]), **(e.get("settings") or {}))
    mine = (info.fps, info.um_per_px, info.border_px, info.rate_pct_per_s, info.tau_r, info.tau_d)
    theirs = (float(e["fps"]), float(e["um_per_px"]), int(e["border_px"]), s.trend_pct_per_s, s.tau_r, s.tau_d)
    if mine != theirs:
        raise ValueError(f"{info.name}: the location file's fps, um per pixel, border, bleaching rate and time "
                         f"constants {mine} are not datasets.yaml's {theirs}")
    leak = dataclasses.replace(s, **(e.get("leakage_test_settings") or {}))
    return {"injection": s, "leakage": leak}


def location(info: LocationInfo, rec: Recording) -> C.Location:
    """The recording's ``cases.Location``: the location's ROIs, no candidates, the territory of the recording's
    events (at least 1), no evidence map, the recording's projection in float64 and its largest event as the
    astrocyte (label 1)."""
    return C.location(f"{info.name}_{rec.name}", info.neurons, np.zeros_like(info.neurons), rec.events, None,
                      np.asarray(rec.max_proj, dtype=np.float64), rec.largest_event.astype(np.int32), info.fps,
                      info.um_per_px, info.border_px, territory_min_events=TERRITORY_MIN_EVENTS)


# --------------------------------------------------------------------------------------------- the traces
def gaussian_1d(signal: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian of s.d. ``sigma`` frames over a trace: 2 ceil(2 sigma) + 1 taps, the edges repeated."""
    radius = int(np.ceil(2 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return ndi.correlate1d(np.asarray(signal, dtype=np.float64), w / w.sum(), mode="nearest")


def ring_mask(loc: C.Location, label: int) -> tuple[np.ndarray, np.ndarray]:
    """(ROI, ring) of ROI ``label``: the ring of the ring subtraction, the pixels more than 2 and at most 8
    reference pixels (4 and 16 um) from the ROI that belong to no ROI and no candidate, not cut at the border."""
    fld = loc.field
    roi = fld.neurons == label
    d = ndi.distance_transform_edt(~roi)
    ring = ((d > scaled(fld, RING_INNER)) & (d <= scaled(fld, RING_OUTER))
            & ~((fld.neurons > 0) | (fld.candidates > 0)))
    return roi, ring


def roi_dff(loc: C.Location, movie: Movie, rule: BaselineRule) -> tuple[np.ndarray, np.ndarray]:
    """(smoothed, raw): every ROI's ΔF/F0 (module docstring), (n_rois, T), and the same before the low-pass and
    the smoothing."""
    fld = loc.field
    n = int(fld.neurons.max())
    masks = [ring_mask(loc, v) for v in range(1, n + 1)]
    flat = np.unique(np.concatenate([fld.flat(roi | ring) for roi, ring in masks]))
    y = movie.read(flat).astype(np.float64)
    f = np.empty((n, y.shape[1]))
    for v, (roi, ring) in enumerate(masks):
        f[v] = (y[np.searchsorted(flat, fld.flat(roi))].mean(axis=0)
                - NEUROPIL_R * y[np.searchsorted(flat, fld.flat(ring))].mean(axis=0))
    f0 = baseline(f, fld.fps, rule)
    roi_mean = np.stack([y[np.searchsorted(flat, fld.flat(roi))].mean(axis=0) for roi, _ in masks])
    f0_roi = baseline(roi_mean, fld.fps, rule)
    raw = (f - f0) / f0_roi
    smoothed = np.stack([gaussian_1d(lowpass_fourier(r, fld.fps, LOWPASS_HZ), SMOOTH_S * fld.fps) for r in raw])
    return smoothed, raw


def astro_dff(loc: C.Location, movie: Movie, mask: np.ndarray, rule: BaselineRule) -> np.ndarray:
    """The largest event's ΔF/F0 (module docstring), (T,)."""
    fld = loc.field
    tr = movie.read(fld.flat(mask)).astype(np.float64).mean(axis=0)
    f0 = baseline(tr[None], fld.fps, rule)[0]
    return lowpass_fourier((tr - f0) / f0, fld.fps, LOWPASS_HZ)


# --------------------------------------------------------------------------------------------- one recording
def build_cases(loc: C.Location, rec: Recording, rule: BaselineRule, tests: Sequence[str] = C.TESTS
                ) -> tuple[dict[str, list[C.Case]], dict[str, Any]]:
    """The cases of each requested test and the choices made (module docstring)."""
    movie = rec.movie()
    dff = {rec.name: roi_dff(loc, movie, rule)[0]}
    out: dict[str, list[C.Case]] = {}
    choices: dict[str, Any] = {}
    if "injection" in tests:
        out["injection"], choices["injection"] = C.injection_cases(loc, dff, C.D7_RULES)
    if "leakage" in tests:
        if rec.largest_event.any():
            astro = {rec.name: astro_dff(loc, movie, rec.largest_event, rule)}
            out["leakage"], choices["leakage"] = C.leakage_cases(loc, dff, astro, 1, C.D7_RULES)
            choices["leakage"]["astro_pixels"] = int(rec.largest_event.sum())
        else:
            choices["leakage"] = {"targets": [], "astro_source": None, "astro_shifts": {}}
    return out, choices


def run_recording(data: str | Path, location_name: str, recording: str, out: str | Path, *,
                  tests: Sequence[str] = C.TESTS, workers: int = 0, config: str | Path = CONFIG,
                  log: Callable[[str], Any] = C.say) -> Path:
    """Build, fit, export and score one recording's cases (module docstring); returns its run folder."""
    folder = Path(data) / location_name
    info = read_location(folder)
    rec = read_recording(folder, recording, info)
    settings = d7_settings(info, config)
    loc = location(info, rec)
    built, choices = build_cases(loc, rec, settings["injection"].baseline_rule(), tests)
    log(f"{location_name} {recording}: {({t: len(c) for t, c in built.items()})} cases; {choices}")
    record = {"location": location_name, "recording": recording, "frames": rec.frames, "choices": choices}
    return C.run_tests(loc, rec.movie(), built, settings, Path(out) / f"{location_name}_{recording}", record,
                       workers=workers, log=log)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="build, fit, export and score one recording's cases")
    r.add_argument("--data", type=Path, required=True, help="the input files, one folder per location")
    r.add_argument("--location", required=True, help="D7_location1 or D7_location2")
    r.add_argument("--recording", required=True)
    r.add_argument("--out", type=Path, required=True, help="the root of the run folders")
    r.add_argument("--tests", nargs="+", choices=C.TESTS, default=list(C.TESTS))
    r.add_argument("--workers", type=int, default=0, help="cases fitted at a time (0: in this process)")
    s = sub.add_parser("score", help="score a run folder again (after the peer methods)")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--results", type=Path, nargs="+", default=None,
                   help="the peers' results folders (default <run>/compare/results)")
    s.add_argument("--baseline", choices=C.BASELINES, default="reported",
                   help="the injection test's scoring (without_added: the paper's tables; cases' docstring)")
    a = ap.parse_args(argv)
    if a.command == "run":
        for var in BLAS_THREADS:                       # one BLAS thread in each worker process
            os.environ[var] = "1"
        run_recording(a.data, a.location, a.recording, a.out, tests=a.tests, workers=a.workers)
    else:
        C.score_run(a.run, None, a.results, baseline=a.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
