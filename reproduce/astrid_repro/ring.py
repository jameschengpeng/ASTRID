"""The ring subtraction on the public datasets, and D3's reference rows.

- **D2, D4, D5** (the paper's Ring, D1's rule in micrometres): per ROI, its mean minus 0.7 x the mean of its
  ring, frame by frame, on the movie every method read (``mov``). The ring is every pixel more than 4 um and at
  most 16 um from the ROI (Euclidean; ``RING_INNER`` and ``RING_OUTER`` reference pixels of
  ``REFERENCE_UM_PER_PX`` um, converted to the dataset's pixel size), without the pixels of any ROI of the
  video, clipped at the field's edge.
- **D3** (one ROI per recording in a field about 30 um across; the ring is not rescaled): the ROI mean ("raw"),
  the ROI mean minus 0.7 x the mean of the ring 2 to 8 px out by 4-connected dilation ("ring"), and the dataset
  authors' own trace, ``fmean_roi`` minus 0.7 x ``fmean_neuropil`` ("authors_ring").
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

from astrid.patch import REFERENCE_UM_PER_PX, RING_FACTOR, RING_INNER, RING_OUTER

BLOCK = 2000                         # frames read at a time
CAI1_RING_IN, CAI1_RING_OUT, CAI1_RING_K = 2, 8, 0.7
CAI1_BLOCK = 1000


def ring_masks(masks: np.ndarray, um_per_px: float) -> np.ndarray:
    """(n, H, W) rings of the (n, H, W) ROI masks at ``um_per_px``."""
    scale = REFERENCE_UM_PER_PX / um_per_px
    others = masks.any(axis=0)
    rings = np.zeros_like(masks)
    for i, roi in enumerate(masks):
        d = ndi.distance_transform_edt(~roi)
        rings[i] = (d > RING_INNER * scale) & (d <= RING_OUTER * scale) & ~others
    return rings


def check_rings(masks: np.ndarray, rings: np.ndarray, where: str = "") -> None:
    """Refuse a ring that holds an ROI pixel or is empty."""
    if (rings & masks.any(axis=0)).any() or not rings.reshape(len(rings), -1).any(axis=1).all():
        raise ValueError(f"{where}: a ring holds an ROI pixel or is empty")


def ring_traces(movie: str | Path, masks: np.ndarray, rings: np.ndarray, key: str = "mov",
                block: int = BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(trace, ROI means, ring means), each (n, T) float64, from the HDF5 movie ``key`` (T, H, W), read in
    blocks of frames; trace = ROI mean - 0.7 x ring mean."""
    n = len(masks)
    w_roi = masks.reshape(n, -1).astype(np.float64)
    w_ring = rings.reshape(n, -1).astype(np.float64)
    w_roi /= w_roi.sum(axis=1, keepdims=True)
    w_ring /= w_ring.sum(axis=1, keepdims=True)
    with h5py.File(movie, "r") as f:
        mov = f[key]
        if mov.shape[1:] != masks.shape[1:]:
            raise ValueError(f"{Path(movie).name}: movie {mov.shape} and masks {masks.shape} differ")
        t = mov.shape[0]
        roi_mean, ring_mean = np.empty((n, t)), np.empty((n, t))
        for s in range(0, t, block):
            b = np.asarray(mov[s:s + block], dtype=np.float64)
            b = b.reshape(b.shape[0], -1).T                                          # (pixels, frames)
            roi_mean[:, s:s + b.shape[1]] = w_roi @ b
            ring_mean[:, s:s + b.shape[1]] = w_ring @ b
    return roi_mean - RING_FACTOR * ring_mean, roi_mean, ring_mean


def video_ring(movie: str | Path, masks: np.ndarray, um_per_px: float) -> dict[str, np.ndarray]:
    """One video's ring subtraction, checked: ``trace``, ``roi_mean``, ``ring_mean`` (n, T) and ``ring_px``
    (n,)."""
    rings = ring_masks(masks, um_per_px)
    check_rings(masks, rings, str(movie))
    trace, roi_mean, ring_mean = ring_traces(movie, masks, rings)
    if not np.isfinite(trace).all():
        raise ValueError(f"{movie}: non-finite ring trace")
    return {"trace": trace, "roi_mean": roi_mean, "ring_mean": ring_mean,
            "ring_px": rings.reshape(len(rings), -1).sum(axis=1)}


def cai1_reference_rows(path: str | Path) -> dict[str, np.ndarray]:
    """D3's reference rows of one exported recording (module docstring): "raw", "ring", "authors_ring"."""
    with h5py.File(path, "r") as f:
        roi = np.asarray(f["roi"], dtype=bool)
        mov = f["mov"]
        inner = ndi.binary_dilation(roi, iterations=CAI1_RING_IN)
        ring = ndi.binary_dilation(roi, iterations=CAI1_RING_OUT) & ~inner
        raw = np.empty(mov.shape[0])
        ring_tr = np.empty(mov.shape[0])
        for s in range(0, mov.shape[0], CAI1_BLOCK):
            b = np.asarray(mov[s:s + CAI1_BLOCK], dtype=np.float64)
            raw[s:s + b.shape[0]] = b[:, roi].mean(axis=1)
            ring_tr[s:s + b.shape[0]] = b[:, ring].mean(axis=1)
        fr, fn = np.asarray(f["fmean_roi"], dtype=float), np.asarray(f["fmean_neuropil"], dtype=float)
    return {"raw": raw, "ring": raw - CAI1_RING_K * ring_tr, "authors_ring": fr - CAI1_RING_K * fn}
