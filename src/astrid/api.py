"""``demix``: ASTRID on a movie and ROI masks, traces out (the paper's Methods).

For every ROI, ASTRID builds the patch around it (``astrid.patch``), fits the patch model (``astrid.solver``)
and delivers the ROI's demixed trace: the ROI average of the baseline-subtracted movie after subtracting the
fitted contributions of every other component (Methods, "The demixed trace"), and its dF/F0.

Inputs (arrays are (T, H, W) movies and (n, H, W) boolean masks on the same frame):
- ``movies``: one recording, or a list of recordings of the same field registered to one pixel grid, fitted
  jointly with shared footprints (Supplementary, "Joint fitting of recordings"). Each is an array, anything
  array-like that slices along time (an ``h5py.Dataset``, a ``numpy.memmap``), or a path:
  ``file.h5[:dataset]``, ``file.npy`` or ``file.tif`` (needs ``tifffile``).
- ``rois``: the ROI masks. Masks that overlap are refused unless ``overlaps="nearest"``, which gives each
  shared pixel to the mask with the nearest centroid.
- ``fps`` (frames per second) and ``um_per_px`` (micrometres per pixel) are required: the baseline, the noise
  band, the kernel and the astrocyte prior are set in seconds, and the patch geometry in micrometres.
- ``tau_r``, ``tau_d``: the indicator's time constants (s), required here or in ``settings``.
- optional ``territory``: astrocyte territory, an (H, W) or (m, H, W) boolean array (any source; masks are
  combined into one map; ``astrid.aqua2.territory`` makes it from AQuA2's results). Without it the patches
  have no astrocyte components (neuron and neuropil only).
- optional ``evidence``: the structural evidence map rho in [0, 1], (H, W) (``astrid.evidence``). Without it
  the evidence penalty is off.
- optional ``candidates``: an (H, W) map of candidate cells that are not ROIs (a detection's rejected
  proposals); its 4-connected pieces become unclassified components (Supplementary, "Spatial supports").
- ``border_px``: pixels at the field's edge that no patch uses (a registration margin); ROIs that touch it
  are not fitted. Default 0.

A fit that raises is recorded (``Result.error``) and the call continues. ``export_components=True`` adds the
other components' contributions and every component's own trace (``Result`` and the README).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import mmap
import multiprocessing
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from astrid import __version__, kernels
from astrid.model import Params
from astrid.noise import high_passed
from astrid.outputs import demixed_trace, other_contributions, support_trace
from astrid.patch import (
    DEFAULT_BASELINE,
    FOUR,
    NOISE_F_CUT,
    REFERENCE_UM_PER_PX,
    BaselineRule,
    Field,
    PixelData,
    Spec,
    build,
    patch_components,
    patch_flat,
    patch_region,
    pixel_statistics,
    ring_trace,
)
from astrid.settings import Settings
from astrid.solver import fit

OVERLAP_RULES = ("refuse", "nearest")
CONTRIBUTION_KINDS = ("neighbours", "unclassified", "astro", "neuropil")


# --------------------------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Movie:
    """One recording: an in-memory or array-like (T, H, W) movie, or a file opened lazily."""
    name: str
    data: Any = None
    path: Path | None = None
    key: str | None = None

    def _open(self) -> Any:
        """A context manager yielding the (T, H, W) array-like."""
        if self.data is not None:
            return _Held(self.data)
        assert self.path is not None
        suffix = self.path.suffix.lower()
        if suffix in (".h5", ".hdf5", ".mat"):
            import h5py

            class _H5:
                def __enter__(s) -> Any:
                    s.f = h5py.File(self.path, "r")
                    return s.f[self.key]

                def __exit__(s, *exc: object) -> None:
                    s.f.close()
            return _H5()
        if suffix == ".npy":
            return _Held(np.load(self.path, mmap_mode="r"))
        if suffix in (".tif", ".tiff"):
            try:
                import tifffile
            except ImportError as err:      # pragma: no cover - optional dependency
                raise ImportError("reading TIFF movies needs tifffile: pip install tifffile") from err
            return _Held(tifffile.memmap(self.path) if _tiff_memmappable(self.path) else
                         tifffile.imread(self.path))
        raise ValueError(f"{self.path}: unknown movie format (use .h5/.hdf5, .npy or .tif)")

    def shape(self) -> tuple[int, int, int]:
        with self._open() as d:
            if len(d.shape) != 3:
                raise ValueError(f"{self.name}: a movie must be (T, H, W), got {tuple(d.shape)}")
            return tuple(int(x) for x in d.shape)  # type: ignore[return-value]

    def read(self, flat: np.ndarray, block: int = 1000) -> np.ndarray:
        """(len(flat), T) float32 traces of the column-major pixels ``flat``, in one pass over the movie."""
        with self._open() as d:
            out = np.empty((flat.size, d.shape[0]), dtype=np.float32)
            for s in range(0, d.shape[0], block):
                b = np.asarray(d[s:s + block])
                out[:, s:s + b.shape[0]] = b.reshape(b.shape[0], -1, order="F")[:, flat].T
        return out

    def sha256(self) -> str | None:
        if self.path is None:
            return None
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        return h.hexdigest()


class _Held:
    def __init__(self, data: Any) -> None:
        self.data = data

    def __enter__(self) -> Any:
        return self.data

    def __exit__(self, *exc: object) -> None:
        return None


def _tiff_memmappable(path: Path) -> bool:
    import tifffile
    try:
        tifffile.memmap(path)
        return True
    except (ValueError, OSError):
        return False


def as_movie(x: Any, name: str) -> Movie:
    """A Movie from an array-like, an ``h5py.Dataset``, a path ("file.h5:dataset", "file.npy", "file.tif"),
    or a Movie."""
    if isinstance(x, Movie):
        return x
    if type(x).__module__.startswith("h5py") and hasattr(x, "file"):
        return Movie(name, path=Path(x.file.filename), key=x.name)
    if isinstance(x, (str, Path)):
        s = str(x)
        drive = len(s) > 1 and s[1] == ":"                 # a Windows drive letter is not a dataset
        path, key = (s.rsplit(":", 1) if s.count(":") > (1 if drive else 0) else (s, None))
        p = Path(path)
        if p.suffix.lower() in (".h5", ".hdf5", ".mat") and key is None:
            import h5py
            with h5py.File(p, "r") as f:
                keys = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            if len(keys) != 1:
                raise ValueError(f"{p}: give the dataset as path:dataset (found {keys})")
            key = keys[0]
        return Movie(name, path=p, key=key)
    if not hasattr(x, "shape") or not hasattr(x, "__getitem__"):
        x = np.asarray(x)
    if len(x.shape) != 3:
        raise ValueError(f"{name}: a movie must be (T, H, W), got {tuple(x.shape)}")
    return Movie(name, data=x)


def resolve_overlaps(masks: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Each pixel of several masks goes to the mask with the nearest centroid; the report counts the masks
    involved and the pixels moved."""
    m = np.asarray(masks, dtype=bool)
    counts = m.sum(axis=0)
    multi = counts > 1
    if not multi.any():
        return m, {"overlapping_masks": 0, "pixels_reassigned": 0}
    cents = np.array([np.argwhere(mm).mean(axis=0) for mm in m])
    out = m.copy()
    moved = 0
    for y, x in np.argwhere(multi):
        owners = np.flatnonzero(m[:, y, x])
        keep = owners[np.argmin(np.hypot(cents[owners, 0] - y, cents[owners, 1] - x))]
        for o in owners:
            if o != keep:
                out[o, y, x] = False
                moved += 1
    involved = {int(o) for y, x in np.argwhere(multi) for o in np.flatnonzero(m[:, y, x])}
    return out, {"overlapping_masks": len(involved), "pixels_reassigned": int(moved)}


def label_image(masks: np.ndarray, what: str) -> np.ndarray:
    """(n, H, W) boolean masks -> (H, W) int32 labels 1..n; overlapping masks are refused."""
    m = np.asarray(masks, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"{what}: masks must be (n, H, W), got {m.shape}")
    counts = m.sum(axis=0)
    if (counts > 1).any():
        raise ValueError(f"{what}: {int((counts > 1).sum())} pixels belong to more than one mask "
                         "(pass overlaps='nearest' to give each to the mask with the nearest centroid)")
    lab = np.zeros(m.shape[1:], dtype=np.int32)
    for i in range(m.shape[0]):
        lab[m[i]] = i + 1
    return lab


def as_map(x: np.ndarray | None, shape: tuple[int, int], what: str) -> np.ndarray | None:
    """An (H, W) boolean map from an (H, W) map or (m, H, W) masks (their union); None stays None."""
    if x is None:
        return None
    a = np.asarray(x, dtype=bool)
    if a.ndim == 3:
        a = a.any(axis=0) if a.shape[0] else np.zeros(a.shape[1:], dtype=bool)
    if a.shape != shape:
        raise ValueError(f"{what} must be (H, W) or (m, H, W) like the masks {shape}, got {np.shape(x)}")
    return a


def make_field(rois: np.ndarray, fps: float, um_per_px: float, *, territory: np.ndarray | None = None,
               evidence: np.ndarray | None = None, candidates: np.ndarray | None = None,
               border_px: int = 0, name: str = "field") -> Field:
    """A ``patch.Field`` from arrays: the ROIs as labels, the territory, the evidence (zero without a map,
    which then carries no evidence penalty), the candidates' 4-connected pieces, the pixel scale."""
    if isinstance(border_px, bool) or int(border_px) != border_px or border_px < 0:
        raise ValueError(f"border_px must be a non-negative integer, got {border_px!r}")
    if not um_per_px > 0:
        raise ValueError(f"um_per_px must be positive, got {um_per_px!r}")
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    neurons = label_image(rois, "rois")
    shape = (int(neurons.shape[0]), int(neurons.shape[1]))
    terr = as_map(territory, shape, "territory")
    cand = as_map(candidates, shape, "candidates")
    if evidence is None:
        tau = np.zeros(shape, dtype=float)
    else:
        tau = np.asarray(evidence, dtype=float)
        if tau.shape != shape or not np.all(np.isfinite(tau)) or tau.min() < 0 or tau.max() > 1:
            raise ValueError("evidence must be (H, W) like the masks, with values in [0, 1]")
    labels = (ndi.label(cand, structure=FOUR)[0] if cand is not None
              else np.zeros(shape, dtype=np.int32))
    return Field(name, shape, neurons, labels, terr if terr is not None else np.zeros(shape, dtype=bool),
                 tau, float(fps), px_scale=REFERENCE_UM_PER_PX / float(um_per_px),
                 tagrfp_channel=evidence is not None, border_px=int(border_px))


# --------------------------------------------------------------------------------------------- parallel
FORK = "fork" in multiprocessing.get_all_start_methods()   # Linux and macOS; not Windows
_SHARED: dict[str, Any] = {}     # big inputs that forked workers inherit instead of a pickled copy
WORKER_POLL_S = 10.0             # how often the workers' exit codes are checked while fits run


class WorkerDied(RuntimeError):
    """A worker process ended while fits were pending (usually killed for lack of memory)."""


@contextmanager
def shared(**objects: Any) -> Iterator[None]:
    """``objects`` in ``_SHARED`` for the processes forked meanwhile (speed only: the workers read the very
    arrays the parent holds)."""
    _SHARED.update(objects)
    try:
        yield
    finally:
        _SHARED.clear()


def _stop(ex: ProcessPoolExecutor) -> None:
    for proc in list((getattr(ex, "_processes", None) or {}).values()):
        if proc.is_alive():
            proc.terminate()
    ex.shutdown(wait=False, cancel_futures=True)


def run_parallel(fn: Any, jobs: Sequence[Any], workers: int, poll_s: float = WORKER_POLL_S,
                 context: str | None = None) -> list[Any]:
    """``fn`` over ``jobs`` in ``workers`` processes, results in job order. A worker that dies stops the run
    with ``WorkerDied`` instead of a wait without end (the workers' exit codes are checked every ``poll_s``
    seconds as well as by the pool)."""
    ex = ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context(context) if context else None)
    try:
        futures = [ex.submit(fn, j) for j in jobs]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=poll_s, return_when=FIRST_EXCEPTION)
            for f in done:
                if f.exception() is not None:
                    raise f.exception()  # type: ignore[misc]
            ended = [proc.exitcode for proc in list((getattr(ex, "_processes", None) or {}).values())
                     if proc.exitcode is not None]
            if ended and pending:
                raise WorkerDied(f"a worker process ended (exit code {ended[0]}) with {len(pending)} fits "
                                 "pending: give the job more memory or fewer workers")
        out = [f.result() for f in futures]
    except BrokenProcessPool as err:
        _stop(ex)
        raise WorkerDied(f"a worker process died ({err}): give the job more memory or fewer workers") from err
    except BaseException:
        _stop(ex)
        raise
    ex.shutdown(wait=True)
    return out


def high_passed_residual(y: np.ndarray, m: np.ndarray, fps: float) -> np.ndarray:
    """The (n, T) pixels' data minus baseline in float64, high-passed as ``patch.build``'s noise covariance
    computes it for a patch; each row from its own trace."""
    return high_passed(y.astype(np.float64) - m.astype(np.float64), fps, NOISE_F_CUT)


def _shared_array(shape: tuple[int, int]) -> np.ndarray:
    """A float64 array in anonymous shared memory: processes forked afterwards write into the parent's array."""
    count = int(shape[0]) * int(shape[1])
    return np.frombuffer(mmap.mmap(-1, max(8 * count, 8)), dtype=np.float64, count=count).reshape(shape)


def _shared_pixel_statistics(args: tuple[str, int, int, float, BaselineRule]) -> tuple[np.ndarray, np.ndarray]:
    name, a, b, fps, rule = args
    y = _SHARED["y"][name][a:b]
    m, var = pixel_statistics((y, fps, rule))
    hp = _SHARED.get("hp")
    if hp is not None:
        hp[name][a:b] = high_passed_residual(y, m, fps)
    return m, var


def load_pixels(movies: Sequence[Movie], fps: float, flat: np.ndarray, workers: int = 0,
                rule: BaselineRule | None = None, high_pass: bool | None = None) -> PixelData:
    """The pixels ``flat`` of every recording, with their baselines (``rule``, default the paper's) and noise
    variances (the variance is the mean over the recordings). ``workers`` > 1 computes the statistics of blocks
    of pixels in parallel (speed only: each pixel's values come from its own trace).

    ``high_pass``: also keep each pixel's ``high_passed_residual`` (``PixelData.hp``), from which
    ``patch.build`` takes a patch's noise covariance instead of high-passing the patch's pixels again. Each
    pixel is then high-passed once rather than once per patch it lies in; the covariance is the same bitwise.
    It doubles the memory of the pixel data. None (the default): on where the statistics are computed in
    forked processes (``workers`` > 1 on Linux or macOS), which write the rows into shared memory; off
    otherwise."""
    rule = DEFAULT_BASELINE if rule is None else rule
    flat = np.unique(flat)
    keep = (FORK and workers > 1 and flat.size > 1) if high_pass is None else bool(high_pass)
    y, m, var, hp = {}, {}, [], {}
    for mv in movies:
        y[mv.name] = mv.read(flat)
        if workers > 1 and flat.size > 1:
            ends = [int(e) for e in np.linspace(0, flat.size, min(workers, flat.size) + 1)]
            if FORK:
                if keep:
                    hp[mv.name] = _shared_array(y[mv.name].shape)
                with shared(y=y, hp=hp if keep else None):
                    parts = run_parallel(_shared_pixel_statistics, [(mv.name, a, b, fps, rule) for a, b in
                                                                   zip(ends[:-1], ends[1:], strict=True)],
                                         workers, context="fork")
            else:
                blocks = [(y[mv.name][a:b], fps, rule) for a, b in zip(ends[:-1], ends[1:], strict=True)]
                parts = run_parallel(pixel_statistics, blocks, workers)
            m[mv.name] = np.concatenate([p[0] for p in parts])
            var.append(np.concatenate([p[1] for p in parts]))
            if keep and not FORK:
                hp[mv.name] = high_passed_residual(y[mv.name], m[mv.name], fps)
        else:
            m[mv.name], v = pixel_statistics((y[mv.name], fps, rule))
            var.append(v)
            if keep:
                hp[mv.name] = high_passed_residual(y[mv.name], m[mv.name], fps)
    return PixelData(flat, y, m, np.mean(var, axis=0), hp=hp if keep else None,
                     hp_fs=float(fps) if keep else None)


# --------------------------------------------------------------------------------------------- outputs
@dataclass
class RoiResult:
    index: int
    fitted: bool
    error: str = ""
    stop: str = ""
    iterations: int = -1
    converged: bool = False
    delta_f: list[np.ndarray] = field(default_factory=list)     # per recording, fluorescence units
    dff: list[np.ndarray] = field(default_factory=list)         # delta_f / baseline
    baseline: list[np.ndarray] = field(default_factory=list)    # the ROI average of F0
    ring: list[np.ndarray] = field(default_factory=list)        # the ring subtraction, fluorescence units
    components: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    pixels: int = 0
    seconds: float = 0.0
    # export_components only: kind -> per recording the ROI-average contribution as dF/F0; per component
    # its support-averaged dF/F0 per recording, and its footprint (field pixels, column-major, and weights)
    contributions: dict[str, list[np.ndarray]] = field(default_factory=dict)
    component_dff: list[list[np.ndarray]] = field(default_factory=list)
    component_pixels: list[np.ndarray] = field(default_factory=list)
    component_weights: list[np.ndarray] = field(default_factory=list)


@dataclass
class Result:
    """What ``demix`` returns. ``arrays()`` gives everything as named arrays (``save`` writes them to an .npz
    file and the provenance to a .json file):

    - per ROI (n rows; NaN or empty for ROIs not fitted): ``fitted``, ``converged``, ``iterations``,
      ``stop``, ``error``, ``n_neighbours``, ``n_unclassified``, ``n_astro``, ``pixels`` (in the patch),
      ``seconds``;
    - per recording ``<rec>``, (n, T) traces: ``dff__<rec>`` (the demixed dF/F0),
      ``delta_f__<rec>`` (the demixed trace in fluorescence units), ``baseline__<rec>`` (the ROI
      average of F0) and ``ring__<rec>`` (the ring subtraction, ROI mean - 0.7 x ring mean, fluorescence
      units);
    - with ``export_components``: ``contrib_<kind>__<rec>`` (n, T), the ROI-average contribution of each
      other kind of component ("neighbours", "unclassified", "astro", "neuropil") as dF/F0 on the ROI's
      baseline (0 where the patch has none), so that ``dff`` plus these is the ROI average of the movie's
      dF/F0; and one row per component of every fitted patch (m rows, the ROI's own first):
      ``component_roi`` (m,) the ROI's index, ``component_name``, ``component_kind``,
      ``component_dff__<rec>`` (m, T) its trace as dF/F0 averaged over its spatial support (as in the
      paper's Fig. 1E), and its footprint as a sparse list, ``component_fp_offset`` (m + 1,) into
      ``component_fp_row``,
      ``component_fp_col`` and ``component_fp_weight`` (field coordinates, 0-based; weights >= 0 summing
      to 1); ``component_footprint`` rebuilds one as an (H, W) image.
    """
    rois: list[RoiResult]
    names: list[str]                     # recording names, in order
    frames: list[int]
    provenance: dict[str, Any]
    shape: tuple[int, int] = (0, 0)      # the field's (H, W)
    export_components: bool = False

    @property
    def dff(self) -> np.ndarray:
        """The demixed dF/F0 of the first recording, (n, T); ``trace(key, recording)`` for the others."""
        return self.trace("dff", self.names[0])

    def trace(self, key: str, recording: str | None = None) -> np.ndarray:
        """(n, T) ``key`` ("dff", "delta_f", "baseline" or "ring") of ``recording`` (default the first)."""
        return self.arrays_for(key, self.names[0] if recording is None else recording)

    def arrays_for(self, key: str, recording: str) -> np.ndarray:
        j = self.names.index(recording)
        a = np.full((len(self.rois), self.frames[j]), np.nan)
        for i, r in enumerate(self.rois):
            if r.fitted:
                a[i] = getattr(r, key)[j]
        return a

    def arrays(self) -> dict[str, np.ndarray]:
        fitted = [r for r in self.rois if r.fitted]
        out: dict[str, np.ndarray] = {
            "fitted": np.array([r.fitted for r in self.rois]),
            "converged": np.array([r.converged for r in self.rois]),
            "iterations": np.array([r.iterations for r in self.rois]),
            "stop": np.array([r.stop for r in self.rois], dtype=str),
            "error": np.array([r.error for r in self.rois], dtype=str),
            "n_neighbours": np.array([sum(k == "neuron" for k in r.kinds[1:]) for r in self.rois]),
            "n_unclassified": np.array([sum(k == "unclassified" for k in r.kinds) for r in self.rois]),
            "n_astro": np.array([sum(k == "astro" for k in r.kinds) for r in self.rois]),
            "pixels": np.array([r.pixels for r in self.rois]),
            "seconds": np.array([r.seconds for r in self.rois]),
            "recordings": np.array(self.names, dtype=str)}
        for rec in self.names:
            for key in ("dff", "delta_f", "baseline", "ring"):
                out[f"{key}__{rec}"] = self.arrays_for(key, rec)
        if self.export_components:
            n = len(self.rois)
            h = self.shape[0]
            for j, rec in enumerate(self.names):
                for kind in CONTRIBUTION_KINDS:
                    a = np.full((n, self.frames[j]), np.nan)
                    for r in fitted:
                        a[r.index] = r.contributions[kind][j] if kind in r.contributions else 0.0
                    out[f"contrib_{kind}__{rec}"] = a
            rows = [(r, i) for r in fitted for i in range(len(r.components))]
            out["component_roi"] = np.array([r.index for r, _ in rows], dtype=np.int64)
            out["component_name"] = np.array([r.components[i] for r, i in rows], dtype=str)
            out["component_kind"] = np.array([r.kinds[i] for r, i in rows], dtype=str)
            for j, rec in enumerate(self.names):
                out[f"component_dff__{rec}"] = (np.stack([r.component_dff[i][j] for r, i in rows]) if rows
                                                else np.zeros((0, self.frames[j])))
            pix = [r.component_pixels[i] for r, i in rows]
            out["component_fp_offset"] = np.concatenate([[0], np.cumsum([p.size for p in pix])]).astype(np.int64)
            flat = np.concatenate(pix) if pix else np.zeros(0, dtype=np.int64)
            out["component_fp_row"] = (flat % h).astype(np.int64)
            out["component_fp_col"] = (flat // h).astype(np.int64)
            out["component_fp_weight"] = (np.concatenate([r.component_weights[i] for r, i in rows]) if rows
                                          else np.zeros(0))
        return out

    def save(self, out_dir: str | Path) -> Path:
        """``out_dir/result.npz`` (``arrays()``) and ``out_dir/result.json`` (the provenance)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "result.npz", **self.arrays())  # type: ignore[arg-type]
        (d / "result.json").write_text(json.dumps(self.provenance, indent=1, default=str), encoding="utf-8")
        return d / "result.npz"


def component_footprint(arrays: dict[str, np.ndarray], j: int, shape: tuple[int, int]) -> np.ndarray:
    """Component row ``j``'s footprint as an (H, W) image, from ``Result.arrays()`` (``export_components``)."""
    a, b = int(arrays["component_fp_offset"][j]), int(arrays["component_fp_offset"][j + 1])
    img = np.zeros(shape)
    img[arrays["component_fp_row"][a:b], arrays["component_fp_col"][a:b]] = arrays["component_fp_weight"][a:b]
    return img


# --------------------------------------------------------------------------------------------- the fit
def fit_roi(fld: Field, data: PixelData, spec: Spec, params: Params, index: int,
            export: bool = False, astro_components: bool = True) -> RoiResult:
    """One ROI: its patch, the fit, the demixed trace per recording; ``export``: also the other components'
    contributions and every component's trace and footprint."""
    t0 = time.time()
    try:
        b = build(fld, data, spec, astro_components=astro_components)
        state = fit(b.patch, params, sigma=b.sigma)
    except Exception as exc:  # noqa: BLE001 - a failed fit is a recorded outcome
        return RoiResult(index, False, error=f"{type(exc).__name__}: {exc}", seconds=time.time() - t0)
    k = 0
    n_rec = len(b.patch.y)
    roi = b.patch.components[k].roi_mask
    delta_f = [demixed_trace(b.patch, state, k, roi, r)[0] for r in range(n_rec)]
    base = [np.broadcast_to(b.patch.m[r], b.patch.y[r].shape)[roi].mean(axis=0) for r in range(n_rec)]
    res = RoiResult(index, True, stop=state.stop_reason, iterations=int(state.iterations),
                    converged=bool(state.target_converged(k, params.tol_target)),
                    delta_f=delta_f, dff=[d / m for d, m in zip(delta_f, base, strict=True)],
                    baseline=base, ring=[ring_trace(b, r) for r in range(n_rec)],
                    components=list(b.names), kinds=[c.kind for c in b.patch.components],
                    pixels=b.patch.n_pixels)
    if export:
        parts = [other_contributions(b.patch, state, k, roi, r) for r in range(n_rec)]
        res.contributions = {kind: [p[kind] / base[r] for r, p in enumerate(parts)] for kind in parts[0]}
        for j in range(b.patch.n_components):
            on = state.h[:, j] > 0
            res.component_dff.append([support_trace(b.patch, state, j, r) for r in range(n_rec)])
            res.component_pixels.append(np.asarray(b.pixels)[on].astype(np.int64))
            res.component_weights.append(state.h[on, j].copy())
    res.seconds = time.time() - t0
    return res


def _job(args: tuple[Any, ...]) -> RoiResult:
    return fit_roi(*args)


def _shared_job(args: tuple[Any, ...]) -> RoiResult:
    """``_job`` on the field and pixel data in ``_SHARED``; ``args`` = (spec, params, index, export,
    astro_components)."""
    spec, params, index, export, astro = args
    fld, data = _SHARED["fld"], _SHARED["data"]
    return fit_roi(fld, data.subset(patch_flat(fld, spec)), spec, params, index, export, astro)


def largest_first(fld: Field, specs: Sequence[Spec], astro_components: bool = True) -> list[int]:
    """The indices of ``specs``, the largest patch problem first: the patch's pixels times its components
    (``patch.build``'s region and components), ties in the given order. ``demix`` hands the fits to its
    workers in this order, so that a slow fit does not start last; each fit is independent of the others, so
    the order changes no result."""
    sizes = []
    for sp in specs:
        roi = fld.neurons == sp.label
        region = patch_region(fld, roi)
        comps, _ = patch_components(fld, region, roi, sp.name, sp.label, astro_components=astro_components)
        sizes.append(int(region.sum()) * len(comps))
    return sorted(range(len(specs)), key=lambda i: -sizes[i])


def demix(movies: Any, rois: np.ndarray, *, fps: float, um_per_px: float, tau_r: float | None = None,
          tau_d: float | None = None, territory: np.ndarray | None = None, evidence: np.ndarray | None = None,
          candidates: np.ndarray | None = None, settings: Settings | None = None, border_px: int = 0,
          overlaps: str = "refuse", workers: int = 0, recording_names: Sequence[str] | None = None,
          export_components: bool = False, astro_components: bool = True, name: str = "field") -> Result:
    """ASTRID on ``movies`` and ``rois`` (module docstring). Returns a ``Result``.

    ``settings``: ``Settings()`` (the paper's defaults) unless given; ``tau_r``/``tau_d`` set its time
    constants. ``workers`` > 0 fits the ROIs in that many processes (with one BLAS thread each, e.g.
    OMP_NUM_THREADS=1, for speed). ``astro_components=False`` keeps the territory out of the patches (the
    paper's ablation A); ``export_components`` adds the components' contributions and traces."""
    t_call = time.perf_counter()
    mlist = movies if isinstance(movies, (list, tuple)) else [movies]
    rnames = list(recording_names) if recording_names else [f"rec{i}" for i in range(len(mlist))]
    if len(rnames) != len(mlist) or len(set(rnames)) != len(rnames):
        raise ValueError("give one distinct recording name per movie")
    mv = [as_movie(x, nm) for x, nm in zip(mlist, rnames, strict=True)]
    s = (settings or Settings()).with_time_constants(tau_r, tau_d)
    p = s.params(float(fps))
    if overlaps not in OVERLAP_RULES:
        raise ValueError(f"overlaps must be one of {OVERLAP_RULES}")
    overlap_report = {"overlapping_masks": 0, "pixels_reassigned": 0}
    if overlaps == "nearest":
        rois, overlap_report = resolve_overlaps(rois)
    fld = make_field(rois, fps, um_per_px, territory=territory, evidence=evidence, candidates=candidates,
                     border_px=border_px, name=name)
    shapes = [m.shape() for m in mv]
    for m, shp in zip(mv, shapes, strict=True):
        if shp[1:] != fld.shape:
            raise ValueError(f"{m.name}: movie frames are {shp[1:]}, the masks {fld.shape}")
    n = int(fld.neurons.max()) if fld.neurons.size else 0
    specs, skipped = [], {}
    for v in range(1, n + 1):
        roi = fld.neurons == v
        if not roi.any():
            skipped[v] = "no pixels in the mask"
            continue
        if (roi & ~fld.inner).any():
            skipped[v] = f"ROI within {int(border_px)} px of the field edge (border_px)"
            continue
        specs.append(Spec(f"R-{v:04d}", label=v))
    t0 = time.time()
    flat = np.unique(np.concatenate([patch_flat(fld, sp) for sp in specs])) if specs else np.zeros(0, int)
    data = load_pixels(mv, float(fps), flat, workers, s.baseline_rule())
    load_s = time.time() - t0
    kernels.warm_up()              # the compiled drive update, loaded before any worker forks
    # in parallel, the largest patch problems first (each fit is independent of the order)
    run_specs = ([specs[i] for i in largest_first(fld, specs, astro_components)]
                 if workers and len(specs) > 1 else specs)
    if workers and len(specs) > 1 and FORK:
        with shared(fld=fld, data=data):
            fitted = run_parallel(_shared_job, [(sp, p, sp.label - 1, export_components, astro_components)
                                                for sp in run_specs], workers, context="fork")
    else:
        jobs = [(fld, data.subset(patch_flat(fld, sp)), sp, p, sp.label - 1, export_components,
                 astro_components) for sp in run_specs]
        fitted = run_parallel(_job, jobs, workers) if workers and len(jobs) > 1 else [fit_roi(*j) for j in jobs]
    by_index = {r.index: r for r in fitted}
    out = [by_index.get(v - 1) or RoiResult(v - 1, False, error=skipped.get(v, "")) for v in range(1, n + 1)]
    frames = [shp[0] for shp in shapes]
    prov = {"astrid": __version__, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "settings": s.as_dict(), "params": dataclasses.asdict(p),
            "movies": [{"name": m.name, "path": None if m.path is None else str(m.path), "key": m.key,
                        "frames": f} for m, f in zip(mv, frames, strict=True)],
            "shape": list(fld.shape), "n_rois": n, "n_fitted": int(sum(r.fitted for r in out)),
            "skipped": skipped, "territory": territory is not None, "evidence": evidence is not None,
            "candidates": candidates is not None, "astro_components": astro_components,
            "fps": float(fps), "um_per_px": float(um_per_px), "px_scale": float(fld.px_scale),
            "border_px": int(border_px), "overlaps": overlaps, **overlap_report, "workers": int(workers),
            "load_seconds": load_s, "fit_seconds": time.time() - t0 - load_s,
            "total_seconds": time.perf_counter() - t_call}
    return Result(out, rnames, frames, prov, shape=fld.shape, export_components=export_components)
