"""Noise of the pixels, from frequencies above a cutoff where calcium signals are small.

``high_passed`` removes each pixel's DCT frequencies up to the cutoff (5 Hz by default); the pixel weights W
are 1 / the variance of what remains, scaled up by the fraction of a white spectrum the high-pass removes
(``patch.pixel_statistics``). ``noise_covariance`` estimates the spatial noise covariance Sigma of a patch the
same way, shrunk towards a scaled identity (Ledoit-Wolf) so that it is well conditioned; the fit uses it only
to judge convergence (the uncertainty of the demixed trace, Supplementary, "Optimization").
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def high_passed(x: np.ndarray, fs: float, f_cut: float) -> np.ndarray:
    """(P, T) traces with DCT frequencies at or below ``f_cut`` removed."""
    coef = dct(x - x.mean(axis=1, keepdims=True), type=2, norm="ortho", axis=1)
    n = x.shape[1]
    coef[:, : int(np.floor(f_cut * 2 * n / fs)) + 1] = 0
    return idct(coef, type=2, norm="ortho", axis=1)


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """(shrunk covariance, shrinkage) of the (n, P) mean-zero samples ``x``."""
    n, p = x.shape
    s = x.T @ x / n
    mu = float(np.trace(s)) / p
    d2 = float(np.sum((s - mu * np.eye(p)) ** 2))
    b2_bar = (float(np.sum(np.sum(x ** 2, axis=1) ** 2)) - n * float(np.sum(s ** 2))) / n ** 2
    shrink = min(b2_bar, d2) / d2 if d2 > 0 else 1.0
    return shrink * mu * np.eye(p) + (1 - shrink) * s, shrink


def noise_covariance(residuals: list[np.ndarray], fs: float, f_cut: float = 5.0) -> tuple[np.ndarray, float]:
    """(Sigma, shrinkage) from per-recording (P, T_r) traces (data minus baseline)."""
    parts = [high_passed(np.asarray(x, dtype=np.float64), fs, f_cut) for x in residuals]
    return covariance_of_high_passed(parts, fs, f_cut)


def covariance_of_high_passed(parts: list[np.ndarray], fs: float,
                              f_cut: float = 5.0) -> tuple[np.ndarray, float]:
    """``noise_covariance`` from the recordings' ``high_passed`` (P, T_r) traces, already computed. A trace's
    high-pass depends on that trace alone, so a patch's rows taken from the whole field's high-passed pixels
    are the patch's own, bitwise (``api.load_pixels``)."""
    x = np.concatenate(parts, axis=1).T
    kept = 1 - f_cut / (fs / 2)                  # fraction of a white spectrum above f_cut
    sigma, shrink = ledoit_wolf(x)
    return sigma / kept, shrink
