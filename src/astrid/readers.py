"""Inputs from other pipelines: the registered movie, the ROI masks and the frame rate from a Suite2p or a
CaImAn run, so that ASTRID can demix their ROIs (``astrid.from_suite2p``, ``astrid.from_caiman``).

Neither package is needed: the files are read directly.

- **Suite2p** (Pachitariu et al. 2016): a plane folder (``suite2p/plane0``) with ``ops.npy``, ``stat.npy``,
  ``iscell.npy`` and the registered movie ``data.bin`` (int16, frames x Ly x Lx). Each ROI's mask is its
  pixels outside other ROIs (``stat['overlap']`` False), the pixels Suite2p itself averages for F, so the
  masks do not overlap.
- **CaImAn** (Giovannucci et al. 2019): the HDF5 file of ``cnm.save``. The spatial footprints
  (``estimates/A``) become masks by CaImAn's own contour rule: the brightest pixels of each footprint that
  hold ``threshold`` (0.9) of its energy. CaImAn's footprints overlap, so the masks usually do too (use
  ``overlaps="nearest"``, the default of ``from_caiman``). The movie is the motion-corrected memmap the run
  saved (``mmap_file``), or one you give.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Inputs:
    """What a reader found: the movie (T, H, W) array-like, the ROI masks (n, H, W), the frame rate, the pixel
    size if the run recorded one, and each mask's index in the run's own numbering."""
    movie: Any
    rois: np.ndarray
    fps: float
    um_per_px: float | None
    index: np.ndarray
    source: str


# --------------------------------------------------------------------------------------------- Suite2p
def read_suite2p(folder: str | Path, *, cells_only: bool = True, include_overlap: bool = False,
                 movie: Any = None) -> Inputs:
    """A Suite2p plane folder. ``cells_only``: the ROIs that ``iscell.npy`` marks as cells (all if the file is
    absent). ``include_overlap``: every pixel of each ROI (the masks may then overlap). ``movie``: use this
    registered movie instead of ``data.bin``."""
    d = Path(folder)
    ops = np.load(d / "ops.npy", allow_pickle=True).item()
    stat = np.load(d / "stat.npy", allow_pickle=True)
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    keep = np.arange(len(stat))
    if cells_only and (d / "iscell.npy").exists():
        keep = np.flatnonzero(np.load(d / "iscell.npy")[:, 0] > 0.5)
    rois = np.zeros((keep.size, ly, lx), dtype=bool)
    for k, i in enumerate(keep):
        s = stat[i]
        yp, xp = np.asarray(s["ypix"]), np.asarray(s["xpix"])
        if not include_overlap and "overlap" in s:
            own = ~np.asarray(s["overlap"], dtype=bool)
            yp, xp = yp[own], xp[own]
        rois[k, yp, xp] = True
    if movie is None:
        nframes = int(ops.get("nframes", 0))
        binary = d / "data.bin"
        if not binary.exists():
            raise FileNotFoundError(f"{binary}: no registered movie; pass movie=")
        size = binary.stat().st_size // 2
        if nframes == 0 or nframes * ly * lx != size:
            nframes = size // (ly * lx)
        movie = np.memmap(binary, dtype=np.int16, mode="r", shape=(nframes, ly, lx))
    return Inputs(movie, rois, float(ops["fs"]), None, keep, f"suite2p:{d}")


# --------------------------------------------------------------------------------------------- CaImAn
_MMAP = re.compile(r"_d1_(\d+)_d2_(\d+)_d3_(\d+)_order_([CF])_frames_(\d+)")


class CaimanMemmap:
    """CaImAn's memmap of a movie (pixels in Fortran order by frames) as a (T, d1, d2) array-like."""

    def __init__(self, path: str | Path) -> None:
        m = _MMAP.search(Path(path).name)
        if m is None:
            raise ValueError(f"{path}: not a CaImAn memmap name (..._d1_.._d2_.._d3_.._order_._frames_...)")
        d1, d2, d3, order, t = int(m[1]), int(m[2]), int(m[3]), m[4], int(m[5])
        if d3 != 1:
            raise ValueError(f"{path}: a 3-D (d3 = {d3}) movie; ASTRID demixes 2-D fields")
        self.dims = (d1, d2)
        self.yr = np.memmap(path, mode="r", dtype=np.float32, shape=(d1 * d2, t), order=order)
        self.shape = (t, d1, d2)

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("index a CaimanMemmap with a slice of frames")
        frames = np.asarray(self.yr[:, key])
        return frames.T.reshape((frames.shape[1], *self.dims), order="F")


def footprint_masks(a: Any, dims: tuple[int, int], threshold: float = 0.9) -> np.ndarray:
    """(K, d1, d2) masks from CaImAn's (d1 d2, K) footprints (a scipy sparse matrix or an array, pixels in
    Fortran order): the brightest pixels of each footprint whose cumulative squared weight stays within
    ``threshold`` of its total (CaImAn's ``get_contours``, method "nrg"), the brightest pixel always kept."""
    from scipy import sparse
    a = sparse.csc_matrix(a)
    out = np.zeros((a.shape[1], dims[0] * dims[1]), dtype=bool)
    for k in range(a.shape[1]):
        idx = a.indices[a.indptr[k]:a.indptr[k + 1]]
        val = a.data[a.indptr[k]:a.indptr[k + 1]]
        if val.size == 0 or not np.any(val > 0):
            continue
        order = np.argsort(val)[::-1]
        energy = np.cumsum(val[order] ** 2)
        energy /= energy[-1]
        inside = energy <= threshold
        inside[0] = True
        out[k, idx[order][inside]] = True
    return out.reshape((a.shape[1], dims[1], dims[0])).transpose(0, 2, 1)


def _h5_value(f: Any, path: str) -> Any:
    return f[path][()] if path in f else None


def read_caiman(path: str | Path, *, movie: Any = None, accepted_only: bool = True,
                threshold: float = 0.9) -> Inputs:
    """The HDF5 file of ``cnm.save``. ``accepted_only``: the components in ``estimates.idx_components`` when
    the run evaluated them. ``movie``: the motion-corrected movie (default: the run's ``mmap_file``)."""
    import h5py
    from scipy import sparse
    with h5py.File(path, "r") as f:
        g = f["estimates/A"]
        shape = tuple(int(x) for x in np.asarray(g["shape"][()]).ravel())
        a = sparse.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)
        dims_raw = _h5_value(f, "dims")
        if dims_raw is None:
            dims_raw = _h5_value(f, "estimates/dims")
        if dims_raw is None:
            raise ValueError(f"{path}: no 'dims' in the file")
        dims = tuple(int(x) for x in np.asarray(dims_raw).ravel())[:2]
        fr = _h5_value(f, "params/data/fr")
        dxy = _h5_value(f, "params/data/dxy")
        accepted = _h5_value(f, "estimates/idx_components")
        mmap = _h5_value(f, "mmap_file")
    index = np.arange(a.shape[1])
    if accepted_only and accepted is not None and np.asarray(accepted).dtype != object and np.size(accepted):
        index = np.asarray(accepted, dtype=np.int64).ravel()
    rois = footprint_masks(a[:, index], (dims[0], dims[1]), threshold)
    if movie is None:
        if mmap is None:
            raise ValueError(f"{path}: no mmap_file recorded; pass movie=")
        name = mmap.decode() if isinstance(mmap, bytes) else str(mmap)
        movie = CaimanMemmap(name)
    um = None
    if dxy is not None:
        v = np.asarray(dxy, dtype=float).ravel()
        if v.size and np.allclose(v, v[0]) and v[0] > 0:
            um = float(v[0])
    if fr is None:
        raise ValueError(f"{path}: no frame rate (params/data/fr) recorded")
    return Inputs(movie, rois, float(np.asarray(fr).ravel()[0]), um, index, f"caiman:{path}")
