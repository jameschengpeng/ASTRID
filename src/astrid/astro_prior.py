"""Soft low-pass prior on astrocyte traces, in the orthonormal DCT-II basis (Supplementary, "Temporal models").

    Omega(C) = sum_k w(f_k) DCT(C)_k^2,   f_k = k fs / (2T),

with w = 0 up to f_a and a raised-cosine ramp to 1 at 2 f_a. The DCT-II basis means symmetric boundaries at
both ends of a recording and no wrap-around, and it makes the row update an exact shrinkage.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct


def dct_frequencies(n: int, fs: float) -> np.ndarray:
    return np.arange(n) * fs / (2 * n)


def highpass_weights(n: int, fs: float, f_a: float) -> np.ndarray:
    f = dct_frequencies(n, fs)
    ramp = np.clip((f - f_a) / f_a, 0, 1)
    return np.sin(0.5 * np.pi * ramp) ** 2


def penalty(c: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * dct(np.asarray(c, dtype=np.float64), type=2, norm="ortho") ** 2))


def shrink(y: np.ndarray, weight: float, lam: float, w: np.ndarray) -> np.ndarray:
    """Exact minimizer of (weight/2) ||c - y||^2 + lam * penalty(c, w)."""
    coef = dct(np.asarray(y, dtype=np.float64), type=2, norm="ortho")
    return idct(coef * (weight / (weight + 2 * lam * w)), type=2, norm="ortho")
