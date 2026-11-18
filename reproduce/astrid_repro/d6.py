"""D6's two-color test from its input files: their format, the reader, the two movies made from the channels, the
cases, ASTRID's fits on both movies and the export for the peer methods (with the patch, the fit and the export of
``astrid_repro.cases``), and the estimators of the two channels' scale. The scores, the kept transient size and
the table are in ``astrid_repro.stats`` (``d6-scores``, ``d6``).

D6's astrocytes (green) and neurons (red) were recorded in separate channels, so the leakage of a real astrocyte
signal can be measured: every method runs on the one-color movie M, the astrocyte channel plus the neuronal movie,
and on the neuronal movie R alone, with the same ROIs, neighbours and territory, and its trace on R is the answer it
should reach on M (Supplementary, "Two-color test on D6").

**Input files**, one folder per imaging location, ``<data>/<location>/``, named ``<mouse>_<location>`` (the tables
count the mice by the start of the name, before the first underscore and any hyphen). Images are (H, W) on the
movies' 2-um grid and in their orientation.
- ``location.npz``:
  - ``neurons`` (int): the ROIs as labels 1..n, detected on the neuronal movie and merged over the location's
    recordings;
  - ``candidates`` (int): the detection's candidate map, labels of its 4-connected pieces (0: none);
  - ``events`` (int): the number of AQuA2 events of the astrocyte channel (AQuA2's defaults,
    ``aqua2_settings/D6_default.csv``) that touch each pixel, over the location's recordings; the territory is
    where it is at least 2;
  - ``recordings`` and ``detected`` (bool, recordings x n): whether the detection found ROI v in recording j (row
    j, column v - 1);
  - ``fps`` (Hz), ``um_per_px`` and ``border_px`` (the registration border: pixels at the edge that no ROI or
    patch uses); these must be D6's entry of ``reproduce/datasets.yaml``.
- ``<recording>.h5``, one per recording:
  - ``ch1`` and ``ch2`` (T, H, W) float32: the registered astrocyte (green) and neuronal (red) channels,
    downsampled 2x in space as D1's movies;
  - ``g``: the ratio of the two detectors' counts per photon, ch1's over ch2's (``gain_ratio``), which puts ch2
    in ch1's units; ``b``: the bleed-through of the green indicator into the red channel, ch2's counts per ch1
    count (``bleedthrough``). These are the values the movies were made with: the estimates rounded to 4 and 6
    decimals.

**The movies** (``make_movies``): the neuronal movie R = g (ch2 - b ch1), the red channel without the green light
it records, in green-detector units, and the one-color movie M = ch1 + R. Each frame is computed in float64 and
stored in float32, R first and M from ch1 and the stored R; R may dip below zero, and nothing is clipped. Every
frame is checked: |M - R - ch1| at most ``CHECK_TOL`` of the frame's largest |M| or |R|.

**The cases** (``cases_of``): the location's ROIs that the detection found in the recording and that do not reach
into the registration margin (``border_px``), in ascending order (an ROI silent in a recording has no neuronal
signal to keep).
A case's territory class (``territory_class``): "in" when at least half of its ROI's pixels are territory, "out"
when none is, else "between".

**The fits**: ASTRID on M and on R with the same field (the ROIs, the candidates and the territory; no evidence
map, D6 has no structural marker), the patch of each case's ROI with the ROI's own label left out of its
neighbours and nothing added (``cases.build_case``), and D6's settings from datasets.yaml (the time constants
and the defaults, at most 600 iterations). Each case's outputs are ``cases.fit_case``'s.

**The export**, per movie, in the case format of ``astrid_repro.peers``: each case is its own base, with the movie
crop and the masks (``cases.export_base``), and ``<case>.npz`` holds a zero footprint and amplitude (the case's
movie is the crop as it is) and ``roi_mean``, the ROI's mean trace in the movie (float64; M's minus R's is the
astrocyte channel's signal in the ROI). The rows of ``index.json`` add ``test`` "d6", the movie, the recording, the
label, and the territory class and share.

**Outputs** of one recording, ``<out>/<location>_<recording>/``: ``run.json`` (the location, the recording, the
scale, the movies' check, the settings and the cases) and, per movie X in M and R, ``X/ours/`` (ASTRID's outputs
per case and ``fits.csv``) and ``X/compare/`` (the export). The peer methods run on each export, with their
results in ``X/peers/`` and ``X/suite2p/``, the layout that ``stats.py d6-scores`` reads:
    python -m astrid_repro.peers launch --cases <run>/M/compare --results <run>/M/peers --runners 16 ...
    python -I astrid_repro/suite2p_runner.py cases --cases <run>/M/compare --results <run>/M/suite2p ...
and the same for R. The movies are written to ``<run>/movies/`` and deleted at the end (``--keep-movies`` keeps
them).

**The scale** (``calibrate``; the input files hold g and b, so this is for checking them):
- ``gain_ratio``: per channel, over ``GAIN_FRAMES`` raw frames from the middle of the recording (at full
  resolution, the field's columns ``GAIN_COLUMNS``), each pixel's variance over its mean, the median over the
  pixels brighter than the median (a photon count's variance over its mean grows with the detector's gain); g is
  ch1's over ch2's;
- ``bleedthrough``: on the 1-s means of the registered channels (``one_second``, ``SECOND_FRAMES`` frames each,
  ``BLEED_BORDER_PX`` pixels at each edge left out), the astrocyte channel's surges: the peaks of its field mean,
  after a ``TREND_S``-s running mean is removed, with a prominence of at least ``PROMINENCE_SD`` times its noise
  s.d. (the s.d. of its frame-to-frame differences over sqrt 2), at least ``SURGE_GAP_S`` s apart and with their
  windows inside the recording. At each surge, each channel's change from its mean over ``BASE_S`` to its mean
  over ``EVENT_S`` (seconds from the peak, the end excluded), less its blur by a Gaussian of ``BLEED_SIGMA_PX``
  pixels; b is the slope through the origin of ch2's changes on ch1's over the pixels of each surge whose ch1
  change is in the top ``BLEED_TOP`` share by size, pooled over the surges.
The bleed-through needs only the input files; the gain needs the raw frames.

Run from ``reproduce/`` with one BLAS thread, as for D1:
    python -m astrid_repro.d6 run --data <inputs> --location <location> --recording <recording> --out <root>
        [--workers 8] [--keep-movies]
    python -m astrid_repro.d6 calibrate --data <inputs> --location <location> --recording <recording>
        [--raw <ch1 raw file> <ch2 raw file>]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from scipy.signal import find_peaks

import astrid
from astrid.api import Movie, load_pixels
from astrid_repro import cases as C

CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"
LOCATION_FILE = "location.npz"
MOVIES = ("M", "R")
IN_TERRITORY = 0.5                  # the share of an ROI's pixels in territory for the class "in"
CHECK_TOL = 1e-6                    # |M - R - ch1| over the frame's largest |M| or |R|
BLOCK = 1000                        # frames per block
BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# the scale's estimators
GAIN_FRAMES = 600
GAIN_COLUMNS = slice(42, 470)       # the field in the raw frames (columns 43-470 of 512); the rest is the scan's margin
SECOND_FRAMES = 31                  # frames per 1-s mean
BLEED_BORDER_PX = 8
TREND_S = 120
PROMINENCE_SD = 4.0
SURGE_GAP_S = 30
EVENT_S = (-3, 4)                   # the 7 s around the peak
BASE_S = (-25, -12)                 # from 25 to 13 s before it
BLEED_SIGMA_PX = 8.0
BLEED_TOP = 0.05


# --------------------------------------------------------------------------------------------- the inputs
@dataclass
class LocationInfo:
    """``location.npz`` (module docstring)."""
    name: str
    neurons: np.ndarray
    candidates: np.ndarray
    events: np.ndarray
    recordings: list[str]
    detected: np.ndarray
    fps: float
    um_per_px: float
    border_px: int


def read_location(folder: str | Path) -> LocationInfo:
    folder = Path(folder)
    with np.load(folder / LOCATION_FILE) as z:
        a = {k: z[k] for k in z.files}
    info = LocationInfo(folder.name, np.asarray(a["neurons"]), np.asarray(a["candidates"]), np.asarray(a["events"]),
                        [str(r) for r in a["recordings"]], np.asarray(a["detected"], dtype=bool), float(a["fps"]),
                        float(a["um_per_px"]), int(a["border_px"]))
    n = int(info.neurons.max())
    if info.detected.shape != (len(info.recordings), n):
        raise ValueError(f"{folder}: detected is {info.detected.shape}, expected {(len(info.recordings), n)} "
                         "(recordings x ROIs)")
    return info


@dataclass
class Recording:
    """One recording's file: its two channels (read lazily) and its scale."""
    name: str
    path: Path
    frames: int
    g: float
    b: float


def read_recording(folder: str | Path, name: str, info: LocationInfo) -> Recording:
    """``<folder>/<name>.h5`` (module docstring), checked against the location."""
    path = Path(folder) / f"{name}.h5"
    if name not in info.recordings:
        raise ValueError(f"{name} is not a recording of {info.name} ({', '.join(info.recordings)})")
    with h5py.File(path, "r") as f:
        shapes = {ch: (tuple(int(x) for x in f[ch].shape), f[ch].dtype) for ch in ("ch1", "ch2")}
        g, b = float(f["g"][()]), float(f["b"][()])
    grid = info.neurons.shape
    for ch, (shape, dtype) in shapes.items():
        if len(shape) != 3 or shape[1:] != grid or dtype != np.float32:
            raise ValueError(f"{path}: {ch} must be (T, H, W) float32 on the location's {grid} grid, got {shape} "
                             f"{dtype}")
    if shapes["ch1"][0] != shapes["ch2"][0]:
        raise ValueError(f"{path}: ch1 and ch2 have different numbers of frames")
    if not (np.isfinite(g) and g > 0 and np.isfinite(b)):
        raise ValueError(f"{path}: g must be positive and b finite, got {g}, {b}")
    return Recording(name, path, shapes["ch1"][0][0], g, b)


def d6_entry(config: str | Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))["datasets"]["D6"]


def d6_settings(info: LocationInfo, config: str | Path = CONFIG) -> astrid.Settings:
    """ASTRID's settings from D6's entry of ``datasets.yaml``; refuses a location whose frame rate, pixel size or
    border is not the entry's."""
    e = d6_entry(config)
    if (info.fps, info.um_per_px, info.border_px) != (float(e["fps"]), float(e["um_per_px"]), int(e["border_px"])):
        raise ValueError(f"{info.name}: fps {info.fps}, {info.um_per_px} um per pixel and border {info.border_px} "
                         f"are not D6's ({e['fps']}, {e['um_per_px']}, {e['border_px']})")
    return astrid.Settings(tau_r=float(e["tau_r"]), tau_d=float(e["tau_d"]), **(e.get("settings") or {}))


def location(info: LocationInfo) -> C.Location:
    """The location's ``cases.Location``: its ROIs, candidates and territory (events at least 2), no evidence map;
    no projection or astrocyte ROIs (D6 adds nothing)."""
    shape = info.neurons.shape
    return C.location(info.name, info.neurons, info.candidates, info.events, None, np.zeros(shape),
                      np.zeros(shape, dtype=np.int32), info.fps, info.um_per_px, info.border_px)


def cases_of(loc: C.Location, info: LocationInfo, recording: str) -> list[int]:
    """The labels of the recording's cases (module docstring), ascending."""
    found = info.detected[info.recordings.index(recording)]
    out = []
    for v in range(1, int(loc.neurons.max()) + 1):
        roi = loc.neurons == v
        if found[v - 1] and roi.any() and not (roi & ~loc.inner).any():
            out.append(v)
    return out


def territory_class(loc: C.Location, label: int) -> tuple[str, float]:
    """("in", "out" or "between", the share of the ROI's pixels in territory)."""
    roi = loc.neurons == label
    share = float((roi & loc.field.territory).sum() / roi.sum())
    return ("in" if share >= IN_TERRITORY else "out" if share == 0 else "between"), share


# --------------------------------------------------------------------------------------------- the movies
def make_movies(rec: Recording, out: Path, block: int = BLOCK) -> dict[str, Any]:
    """Write ``out/M.h5`` and ``out/R.h5`` (``mov``, (T, H, W) float32) from the recording's channels (module
    docstring); returns the frames and the check's largest relative error. A ValueError if the check fails."""
    out.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    with h5py.File(rec.path, "r") as f, h5py.File(out / "M.h5", "w") as fm, h5py.File(out / "R.h5", "w") as fr:
        d1, d2 = f["ch1"], f["ch2"]
        dm = fm.create_dataset("mov", shape=d1.shape, dtype=np.float32)
        dr = fr.create_dataset("mov", shape=d1.shape, dtype=np.float32)
        for s in range(0, d1.shape[0], block):
            c1 = np.asarray(d1[s:s + block], dtype=np.float64)
            r = (rec.g * (np.asarray(d2[s:s + block], dtype=np.float64) - rec.b * c1)).astype(np.float32)
            m = (c1 + r).astype(np.float32)
            dr[s:s + c1.shape[0]] = r
            dm[s:s + c1.shape[0]] = m
            err = np.abs(m.astype(np.float64) - r - c1).max(axis=(1, 2))
            scale = np.maximum(np.abs(m).max(axis=(1, 2)), np.abs(r).max(axis=(1, 2))).astype(np.float64)
            worst = max(worst, float((err / np.maximum(scale, 1e-30)).max()))
    if worst > CHECK_TOL:
        raise ValueError(f"{rec.name}: M - R differs from ch1 by {worst:.2e} of the frame's range")
    return {"frames": rec.frames, "check_max_rel": worst, "check_tol": CHECK_TOL}


# --------------------------------------------------------------------------------------------- the export
def export_movie(loc: C.Location, crops: dict[str, np.ndarray], cases: list[C.Case], movie: str, recording: str,
                 out: Path) -> list[dict[str, Any]]:
    """One movie's cases in the case format of ``astrid_repro.peers`` (module docstring); writes and returns the
    rows of ``index.json``."""
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for c in cases:
        assert c.label is not None
        roi = c.roi(loc)
        crop = crops[c.name]
        sl, nc, na = C.export_base(loc, crop, out, c.name, roi, c.excluded())
        # the ROI's pixels in the field's column-major order, as every trace of the patch reads them
        roi_mean = crop.transpose(0, 2, 1)[:, roi[sl].T].mean(axis=1, dtype=np.float64)
        np.savez_compressed(out / f"{c.name}.npz", footprint=np.zeros(crop.shape[1:], dtype=np.float32),
                            amplitude=np.zeros(crop.shape[0]), roi_mean=roi_mean)
        cls, share = territory_class(loc, c.label)
        index.append({"spec": c.name, "base": c.name, "test": "d6", "movie": movie, "recording": recording,
                      "label": c.label, "territory": cls, "territory_share": share, "neighbours": nc, "atoms": na})
    C.write_index(out, index)
    return index


# --------------------------------------------------------------------------------------------- one recording
def run_recording(data: str | Path, location_name: str, recording: str, out: str | Path, *, workers: int = 0,
                  keep_movies: bool = False, config: str | Path = CONFIG, log: Callable[[str], Any] = C.say) -> Path:
    """Make the movies of one recording, fit and export its cases on each (module docstring); returns its run
    folder."""
    t0 = time.time()
    folder = Path(data) / location_name
    info = read_location(folder)
    rec = read_recording(folder, recording, info)
    settings = d6_settings(info, config)
    loc = location(info)
    cases = [C.Case(f"M-{v:03d}", label=v) for v in cases_of(loc, info, recording)]
    run = Path(out) / f"{location_name}_{recording}"
    run.mkdir(parents=True, exist_ok=True)
    made = make_movies(rec, run / "movies")
    log(f"{location_name} {recording}: {len(cases)} cases of {int(loc.neurons.max())} ROIs; movies made "
        f"({made['check_max_rel']:.2e}, {time.time() - t0:.0f} s)")
    record = {"location": location_name, "recording": recording, "frames": rec.frames, "g": rec.g, "b": rec.b,
              "movies": made, "fps": loc.field.fps, "astrid": astrid.__version__, "settings": settings.as_dict(),
              "territory_pixels": int(loc.field.territory.sum()), "cases": [c.name for c in cases]}
    (run / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    params = settings.params(loc.field.fps)
    flat = np.unique(np.concatenate([C.case_flat(loc, c) for c in cases])) if cases else np.zeros(0, dtype=int)
    for movie in MOVIES:
        path = run / "movies" / f"{movie}.h5"
        if cases:
            pixels = load_pixels([Movie(recording, path=path, key="mov")], loc.field.fps, flat, workers,
                                 settings.baseline_rule(), high_pass=False)
            log(f"{movie}: {flat.size} pixels read ({time.time() - t0:.0f} s); {len(cases)} fits, at most "
                f"{params.n_outer} iterations")
            rows = C.fit_cases(loc, pixels, cases, params, run / movie / "ours", workers=workers, log=log)
            pd.DataFrame(rows).to_csv(run / movie / "ours" / "fits.csv", index=False)
            del pixels
        with h5py.File(path, "r") as f:
            crops = C.read_crops(f["mov"], {c.name: C.crop_box(loc, c.roi(loc)) for c in cases})
        index = export_movie(loc, crops, cases, movie, recording, run / movie / "compare")
        log(f"{movie}: {len(index)} cases exported, {sum(r['territory'] == 'in' for r in index)} in territory")
    if not keep_movies:
        shutil.rmtree(run / "movies")
    log(f"done in {time.time() - t0:.0f} s -> {run}")
    return run


# --------------------------------------------------------------------------------------------- the scale
def counts_per_photon(frames: np.ndarray) -> float:
    """One channel's raw frames (T, H, W): each pixel's variance over its mean, the median over the pixels brighter
    than the median pixel."""
    x = np.asarray(frames, dtype=np.float64)
    m, v = x.mean(0), x.var(0)
    bright = m > np.median(m)
    return float(np.median(v[bright] / m[bright]))


def middle_frames(raw: Any, n: int = GAIN_FRAMES) -> np.ndarray:
    """``n`` frames from the middle of a (T, H, W) array-like, the field's columns (``GAIN_COLUMNS``)."""
    s = max(0, int(raw.shape[0]) // 2 - n // 2)
    return np.asarray(raw[s:s + n])[:, :, GAIN_COLUMNS]


def gain_ratio(raw1: Any, raw2: Any) -> float:
    """g from the two channels' raw frames, (T, H, W) array-likes at full resolution (module docstring)."""
    return counts_per_photon(middle_frames(raw1)) / counts_per_photon(middle_frames(raw2))


def one_second(movie: Any, block: int = SECOND_FRAMES * 64) -> np.ndarray:
    """The 1-s means of a (T, H, W) array-like, ``SECOND_FRAMES`` frames each (float32; the frames after the last
    whole second are left out)."""
    t, h, w = (int(x) for x in movie.shape)
    n = t // SECOND_FRAMES
    out = np.empty((n, h, w), np.float32)
    for s in range(0, n * SECOND_FRAMES, block):
        x = np.asarray(movie[s:min(s + block, n * SECOND_FRAMES)])
        k = x.shape[0] // SECOND_FRAMES
        out[s // SECOND_FRAMES:s // SECOND_FRAMES + k] = x.reshape(k, SECOND_FRAMES, h, w).mean(1)
    return out


def surges(trace: np.ndarray) -> np.ndarray:
    """The surges of a 1-s field mean (module docstring) whose windows lie inside the recording."""
    x = trace - ndi.uniform_filter1d(trace, TREND_S, mode="nearest")
    noise = np.std(np.diff(x)) / np.sqrt(2)
    peaks, _ = find_peaks(x, prominence=PROMINENCE_SD * noise, distance=SURGE_GAP_S)
    return peaks[(peaks + BASE_S[0] >= 0) & (peaks + EVENT_S[1] <= trace.size)]


def _highpass(img: np.ndarray) -> np.ndarray:
    return img - ndi.gaussian_filter(img, BLEED_SIGMA_PX, mode="nearest")


def bleedthrough_slope(src: np.ndarray, dst: np.ndarray, peaks: np.ndarray) -> float:
    """The pooled regression through the origin of ``dst``'s changes on ``src``'s at ``peaks`` (module docstring);
    NaN without a surge."""
    num, den = 0.0, 0.0
    (e0, e1), (b0, b1) = EVENT_S, BASE_S
    for p in peaks:
        ds = _highpass(src[p + e0:p + e1].mean(0) - src[p + b0:p + b1].mean(0))
        dd = _highpass(dst[p + e0:p + e1].mean(0) - dst[p + b0:p + b1].mean(0))
        sel = np.abs(ds) >= np.quantile(np.abs(ds), 1 - BLEED_TOP)
        num += float(np.sum(ds[sel] * dd[sel]))
        den += float(np.sum(ds[sel] ** 2))
    return num / den if den > 0 else float("nan")


def bleedthrough(sec1: np.ndarray, sec2: np.ndarray) -> tuple[float, int]:
    """(b, the number of surges) from the two channels' 1-s means (``one_second``; seconds, H, W)."""
    e = BLEED_BORDER_PX
    g = np.asarray(sec1, dtype=np.float32)[:, e:-e, e:-e].astype(np.float64)
    r = np.asarray(sec2, dtype=np.float32)[:, e:-e, e:-e].astype(np.float64)
    peaks = surges(g.mean((1, 2)))
    return bleedthrough_slope(g, r, peaks), int(peaks.size)


class _MatlabFrames:
    """An h5py dataset of frames as MATLAB stores them (h5py reads (T, W, H)), sliced along time as (T, H, W)."""

    def __init__(self, ds: Any) -> None:
        self.ds = ds
        self.shape = (ds.shape[0], ds.shape[2], ds.shape[1])

    def __getitem__(self, key: slice) -> np.ndarray:
        return np.asarray(self.ds[key]).transpose(0, 2, 1)


def calibrate(data: str | Path, location_name: str, recording: str, raw: tuple[str, str] | None = None,
              raw_key: str = "data") -> dict[str, Any]:
    """b from the recording's input file and, given its two raw frame files (HDF5 or MATLAB v7.3, the frames under
    ``raw_key`` as MATLAB stores them), g; each beside the input file's value."""
    folder = Path(data) / location_name
    rec = read_recording(folder, recording, read_location(folder))
    with h5py.File(rec.path, "r") as f:
        sec = [one_second(f[ch]) for ch in ("ch1", "ch2")]
    b, n = bleedthrough(*sec)
    out: dict[str, Any] = {"recording": recording, "b": b, "surges": n, "b_file": rec.b}
    if raw is not None:
        with h5py.File(raw[0], "r") as f1, h5py.File(raw[1], "r") as f2:
            out["g"] = gain_ratio(_MatlabFrames(f1[raw_key]), _MatlabFrames(f2[raw_key]))
        out["g_file"] = rec.g
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="make the movies, fit and export one recording's cases")
    c = sub.add_parser("calibrate", help="estimate b (and g from the raw frames) again")
    for p in (r, c):
        p.add_argument("--data", type=Path, required=True, help="the input files, one folder per location")
        p.add_argument("--location", required=True)
        p.add_argument("--recording", required=True)
    r.add_argument("--out", type=Path, required=True, help="the root of the run folders")
    r.add_argument("--workers", type=int, default=0, help="cases fitted at a time (0: in this process)")
    r.add_argument("--keep-movies", action="store_true", help="keep M and R in <run>/movies")
    c.add_argument("--raw", nargs=2, default=None, metavar=("CH1", "CH2"), help="the two channels' raw frame files")
    c.add_argument("--raw-key", default="data", help="the frames' dataset in the raw files")
    a = ap.parse_args(argv)
    if a.command == "run":
        for var in BLAS_THREADS:                       # one BLAS thread in each worker process
            os.environ[var] = "1"
        run_recording(a.data, a.location, a.recording, a.out, workers=a.workers, keep_movies=a.keep_movies)
    else:
        print(json.dumps(calibrate(a.data, a.location, a.recording, a.raw, a.raw_key), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
