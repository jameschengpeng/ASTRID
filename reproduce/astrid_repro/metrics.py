"""The paper's measures and statistics (Methods, "Benchmark protocol"; Supplementary, "Injection and leakage
tests", "Two-color test on D6", "Statistics and runtime").

- ``error``: injection test, ||y - q|| / ||q|| with y the method's trace and q the added signal, both low-passed
  below 0.5 Hz (an FFT low-pass) and their means removed; 0 is exact recovery, 1 a flat trace.
- ``leakage``: leakage test, ||dy|| / ||q|| with dy the change of the trace when the astrocyte signal is added
  and q that signal averaged over the ROI, means removed (unfiltered on D1 and D7; below 5 Hz on D6, where q is
  the real astrocyte channel); 0 is ideal, the raw ROI average scores 1.
- ``gain`` (secondary): the regression slope of the trace on the added signal, means removed.
- ``usable`` and the failure rule: a trace that is missing, not finite or constant is scored as the raw ROI
  average (its error, a leakage of 1).
- ``wilcoxon_p`` and ``holm``: the paired two-sided Wilcoxon signed-rank test over the units (imaging
  locations, recordings, videos or cells), the one test of every table, and Holm's correction over a measure's
  comparisons.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

ERROR_BAND_HZ = 0.5
D6_BAND_HZ = 5.0


def lowpass_fourier(signal: np.ndarray, fps: float, f_thr: float) -> np.ndarray:
    """Every FFT component above ``f_thr`` Hz set to zero (MATLAB's ``ifft(..., 'symmetric')`` of the
    filtered spectrum is ``irfft``)."""
    x = np.asarray(signal, dtype=np.float64)
    spec = np.fft.rfft(x)
    freqs = fps * np.arange(spec.size) / x.size
    spec[freqs > f_thr] = 0
    return np.fft.irfft(spec, n=x.size)


def ac(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


def band(x: np.ndarray, f: float | None, fps: float) -> np.ndarray:
    """``x`` low-passed below ``f`` Hz (None: unfiltered), then its mean removed."""
    return ac(np.asarray(x, dtype=np.float64)) if f is None else ac(lowpass_fourier(x, fps, f))


def error(trace: np.ndarray, truth: np.ndarray, fps: float, f: float | None = ERROR_BAND_HZ) -> float:
    """The injection test's error (module docstring); ``f`` None gives the unfiltered error."""
    q = band(truth, f, fps)
    return float(np.linalg.norm(band(trace, f, fps) - q) / np.linalg.norm(q))


def gain(trace: np.ndarray, truth: np.ndarray) -> float:
    a, q = ac(np.asarray(trace, dtype=np.float64)), ac(np.asarray(truth, dtype=np.float64))
    return float(a @ q / (q @ q))


def leakage(with_astro: np.ndarray, without: np.ndarray, astro_in_roi: np.ndarray, fps: float,
            f: float | None = None) -> float:
    """The leakage test's measure (module docstring): unfiltered (``f`` None) on D1 and D7, ``f`` = 5 Hz on D6."""
    d = band(np.asarray(with_astro, dtype=np.float64) - np.asarray(without, dtype=np.float64), f, fps)
    return float(np.linalg.norm(d) / np.linalg.norm(band(astro_in_roi, f, fps)))


def usable(x: np.ndarray | None) -> bool:
    """A trace that can be scored: present, 1-D, finite and not constant."""
    return x is not None and np.ndim(x) == 1 and np.size(x) > 1 and bool(np.isfinite(x).all()) \
        and float(np.std(x)) > 0


WILCOXON_METHODS = ("auto", "exact")


def wilcoxon_p(d: np.ndarray, method: str = "auto") -> float:
    """Two-sided Wilcoxon signed-rank p of the paired differences ``d``, zero differences dropped (SciPy's
    ``zero_method="wilcox"``); 1 when all are zero. ``method`` is SciPy's (1.15):
    - "auto", SciPy's default and the paper's, in every table: the exact null distribution for up to 50
      differences without zeros or ties; with zeros or ties, exhaustive permutations for up to 13 differences
      and the normal approximation above;
    - "exact": the exact null distribution of the nonzero differences, ties not taken into account."""
    if method not in WILCOXON_METHODS:
        raise ValueError(f"method must be one of {WILCOXON_METHODS}, got {method!r}")
    return float(wilcoxon(d, method=method).pvalue) if np.any(np.asarray(d) != 0) else 1.0


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm's step-down correction of a measure's comparisons."""
    names = sorted(p, key=lambda k: p[k])
    out, m, running = {}, len(names), 0.0
    for i, k in enumerate(names):
        running = max(running, (m - i) * p[k])
        out[k] = min(1.0, running)
    return out
