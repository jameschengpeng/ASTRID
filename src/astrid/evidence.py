"""The structural evidence map rho from an image of a structural astrocyte marker (Supplementary, "Astrocyte inputs").

The marker's temporal mean image (averaged over the recordings of the imaging location) gives each pixel a
local contrast R = G_1 * I / G_15 * I, the image smoothed with Gaussian kernels of s.d. ``sigma`` (1 px) and
``background`` (15 px) at the reference resolution of 2 um per pixel. With R_50 and R_95 the median and 95th
percentile of R within the field without a margin of ``margin_px`` pixels,

    rho = clip((R - R_50) / (R_95 - R_50), 0, 1).

In D1 the marker was tagRFP, expressed by the astrocytes and imaged in a second channel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage as ndi

SIGMA_PX = 1.0
BACKGROUND_PX = 15.0
MARGIN_PX = 5


def local_contrast(image: np.ndarray, sigma: float = SIGMA_PX, background: float = BACKGROUND_PX) -> np.ndarray:
    """R: the image smoothed at ``sigma`` over the image smoothed at ``background`` (pixels)."""
    img = np.asarray(image, dtype=np.float64)
    return ndi.gaussian_filter(img, sigma) / ndi.gaussian_filter(img, background)


def evidence_map(images: np.ndarray | Sequence[np.ndarray], *, margin_px: int = MARGIN_PX,
                 sigma: float = SIGMA_PX, background: float = BACKGROUND_PX) -> np.ndarray:
    """rho in [0, 1], (H, W), from the marker's mean image, or from several (a list, or an (n, H, W) array of
    the recordings' mean images, averaged first)."""
    arr = [np.asarray(i, dtype=np.float64) for i in images] if isinstance(images, (list, tuple)) else None
    img = np.mean(arr, axis=0) if arr is not None else np.asarray(images, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=0)
    if img.ndim != 2:
        raise ValueError(f"the marker image must be (H, W), got {img.shape}")
    contrast = local_contrast(img, sigma, background)
    inner = np.zeros(img.shape, dtype=bool)
    m = int(margin_px)
    inner[m:img.shape[0] - m, m:img.shape[1] - m] = True
    lo, hi = np.percentile(contrast[inner], [50, 95])
    return np.clip((contrast - lo) / (hi - lo), 0, 1)


def mean_image(movie: np.ndarray, block: int = 300) -> np.ndarray:
    """The temporal mean of a (T, H, W) movie (array-like), read in blocks, as float64."""
    acc = np.zeros(movie.shape[1:])
    for i in range(0, movie.shape[0], block):
        acc += np.asarray(movie[i:i + block]).sum(axis=0)
    return acc / movie.shape[0]
