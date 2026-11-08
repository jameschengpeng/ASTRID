"""TUnCaT's transient scoring in Python (GPL-2: see this folder's ``__init__.py`` and ``LICENSE``). It follows
``evaluation/GetPerformance_SpikeDetection_split.m``, ``SNR_normalization.m`` and ``noise_PSD.m`` of TUnCaT
(Bao et al. 2022) line by line.

Per neuron trace ``x`` (the method's output; on a raw video first filtered with the indicator's template):
- baseline ``mu``: the mode of MATLAB ``ksdensity``'s kernel-density estimate of the trace (``ksd_mode``: a normal
  kernel, the robust Silverman bandwidth, 100 points), or the median;
- noise ``sigma``: ``noise_psd``, the square root of the mean power spectral density over the top third of the
  frequencies (above 2/3 of Nyquist), the trace's mean removed and the DC term zeroed;
- detected transients: the frames where ``x > mu + thred_ratio * sigma``, split into runs. A run holding several
  peaks (``find_peaks`` with prominence ``sigma * thred_ratio / 3`` and height ``thred``) is split at the valleys
  between consecutive peaks, and a run with no peak is dropped;
- matching: the ground-truth transients are the labelled ``[start, end]`` frames with ``valid`` = 1. The overlap
  of each (ground truth, detected) pair is its number of shared frames, and a pair with no overlap cannot match.
  The Hungarian assignment maximizing the overlap gives the matches;
- recall = matched ground truth / all ground truth, precision = matched detected / all detected, pooled over the
  video's labelled neurons; F1 their harmonic mean (0 if undefined).
``thred_ratio`` is the detector's threshold, chosen per method by leave-one-video-out cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

THRED_RATIOS = tuple(range(30, 85, 5))         # eval_ABO_*.m: 30:5:80 with the PSD noise (raw and SNR videos)


def ksd_mode(x: np.ndarray, n_points: int = 100) -> float:
    """The mode of MATLAB ``ksdensity(x)`` with its defaults: a normal kernel, bandwidth
    ``medAD / 0.6745 * (4 / (3 n))^(1/5)`` (the robust Silverman rule, medAD the median absolute deviation from
    the median; when medAD is 0 MATLAB takes the range in its place, and a unit bandwidth if that is 0 too),
    100 points from ``min - 3 bw`` to ``max + 3 bw``; the mean of the arg-max points."""
    x = np.asarray(x, dtype=float)
    n = x.size
    sig = float(np.median(np.abs(x - np.median(x)))) / 0.6745
    if sig <= 0:                                   # MATLAB's fallback: the range, then a unit bandwidth
        sig = float(x.max() - x.min())
    bw = sig * (4 / (3 * n)) ** 0.2 if sig > 0 else 1.0
    if x.max() == x.min():
        return float(x[0])                         # a constant trace: the density peaks there
    grid = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, n_points)
    f = np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2).sum(axis=1)
    return float(grid[f == f.max()].mean())


def noise_psd(traces: np.ndarray, pct: float = 2 / 3) -> np.ndarray:
    """``noise_PSD.m``: per row, the square root of the mean PSD over the frequency bins ``round(pct T/2)`` to
    ``T - that`` (MATLAB 1-based ``above:(T-above)``), the DC term zeroed."""
    x = np.atleast_2d(np.asarray(traces, dtype=float))
    x = x - x.mean(axis=1, keepdims=True)
    t = x.shape[1]
    spec = np.fft.fft(x, axis=1)
    spec[:, 0] = 0
    psd = np.abs(spec) ** 2 / t
    above = int(round(pct * t / 2))
    rng = slice(above - 1, t - above)          # MATLAB above:(T-above), 1-based inclusive
    return np.sqrt(psd[:, rng].mean(axis=1))


def detect(x: np.ndarray, mu: float, sigma: float, thred_ratio: float) -> np.ndarray:
    """Detected transients as (m, 2) [start, end] 0-based inclusive frames."""
    x = np.asarray(x, dtype=float)
    thred = mu + sigma * thred_ratio
    above = x > thred
    d = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    locs, _ = find_peaks(x, prominence=sigma * thred_ratio / 3, height=thred)
    locs = np.concatenate([[0], locs, [x.size - 1]])            # MATLAB adds frames 1 and T
    peaks = np.zeros(x.size, dtype=bool)
    peaks[locs] = True
    out = []
    for s, e in zip(starts, ends, strict=True):
        p = np.flatnonzero(peaks[s:e + 1]) + s
        if p.size == 0:
            continue
        if p.size == 1:
            out.append((s, e))
            continue
        bounds = [s]
        for a, b in zip(p[:-1], p[1:], strict=True):
            valley = a + int(np.argmin(x[a:b + 1]))
            bounds += [valley - 1, valley + 1]
        bounds.append(e)
        out += [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]
    return np.array(out, dtype=int).reshape(-1, 2)


def match(gt: np.ndarray, det: np.ndarray) -> tuple[int, int]:
    """(matched ground truth, matched detected) by the Hungarian assignment on frame overlap; zero overlap never
    matches. Both arrays are (m, 2) inclusive frame ranges on the same base."""
    if gt.size == 0 or det.size == 0:
        return 0, 0
    lo = np.maximum(gt[:, None, 0], det[None, :, 0])
    hi = np.minimum(gt[:, None, 1], det[None, :, 1])
    overlap = np.clip(hi - lo + 1, 0, None)
    cost = np.where(overlap > 0, -overlap.astype(float), 1e9)
    r, c = linear_sum_assignment(cost)
    ok = overlap[r, c] > 0
    return int(ok.sum()), int(ok.sum())


def matched(gt: np.ndarray, det: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which ground-truth and which detected transients ``match`` pairs, as two boolean arrays: the same
    Hungarian assignment on frame overlap (zero overlap never matches)."""
    g, d = np.zeros(len(gt), dtype=bool), np.zeros(len(det), dtype=bool)
    if gt.size and det.size:
        lo = np.maximum(gt[:, None, 0], det[None, :, 0])
        hi = np.minimum(gt[:, None, 1], det[None, :, 1])
        overlap = np.clip(hi - lo + 1, 0, None)
        r, c = linear_sum_assignment(np.where(overlap > 0, -overlap.astype(float), 1e9))
        ok = overlap[r, c] > 0
        g[r[ok]], d[c[ok]] = True, True
    return g, d


@dataclass
class Score:
    recall: float
    precision: float
    f1: float
    n_gt: int
    n_det: int
    n_neurons: int


def baselines(traces: np.ndarray, baseline: str = "ksd") -> np.ndarray:
    """Per-row baseline ``mu`` (the ksd mode, or the median); it does not depend on the threshold."""
    x = np.atleast_2d(np.asarray(traces, dtype=float))
    return np.array([ksd_mode(row) if baseline == "ksd" else float(np.median(row)) for row in x])


def score(traces: np.ndarray, labels: list[np.ndarray | None], thred_ratio: float, *,
          sigma_from: np.ndarray | None = None, baseline: str = "ksd", mu: np.ndarray | None = None,
          sigma: np.ndarray | None = None) -> Score:
    """F1 of ``traces`` (n, T) against ``labels`` (per neuron: (m, 3) [start, end, valid], 1-based, or None for
    unlabelled neurons, left out). ``sigma_from``: traces to take the noise from instead (TUnCaT's "Raw" option).
    ``mu``/``sigma``: precomputed per-row baseline and noise (``baselines``, ``noise_psd``)."""
    x = np.atleast_2d(np.asarray(traces, dtype=float))
    sig = noise_psd(x if sigma_from is None else sigma_from) if sigma is None else np.asarray(sigma, float)
    mus = baselines(x, baseline) if mu is None else np.asarray(mu, dtype=float)
    n_gt = n_det = m_gt = m_det = n_used = 0
    for i, lab in enumerate(labels):
        if lab is None or lab.size == 0:
            continue
        n_used += 1
        gt = np.asarray(lab)
        gt = gt[gt[:, 2].astype(bool), :2].astype(int) - 1          # valid only; to 0-based
        det = detect(x[i], float(mus[i]), float(sig[i]), thred_ratio)
        a, b = match(gt, det)
        n_gt += len(gt)
        n_det += len(det)
        m_gt += a
        m_det += b
    recall = m_gt / n_gt if n_gt else 0.0
    precision = m_det / n_det if n_det else 0.0
    f1 = 2 / (1 / recall + 1 / precision) if recall > 0 and precision > 0 else 0.0
    return Score(recall, precision, f1, n_gt, n_det, n_used)


def template_from_archive(archive: str | Path, name: str = "GCaMP6f_spike_tempolate_mean.mat") -> np.ndarray | None:
    """TUnCaT's ``template/<name>`` kernel (``filter_tempolate``; its values above e^-1, normalized to sum 1,
    time-reversed for ``conv2(..., 'valid')``), used on the raw video only: ``GCaMP6f_spike_tempolate_mean`` for
    ABO, ``1P_spike_tempolate`` for the one-photon set. None if the file is missing."""
    import scipy.io

    p = Path(archive) / "template" / name
    if not p.exists():
        return None
    k = np.asarray(scipy.io.loadmat(p)["filter_tempolate"], dtype=float).ravel()
    k = k[k > np.exp(-1)]
    return (k / k.sum())[::-1]


def filtered(traces: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    """``conv2(traces, kernel, 'valid')`` along time; the identity without a kernel."""
    if kernel is None:
        return np.atleast_2d(np.asarray(traces, dtype=float))
    x = np.atleast_2d(np.asarray(traces, dtype=float))
    return np.stack([np.convolve(row, kernel, mode="valid") for row in x])


def gt_transients_from_clean(calcium: np.ndarray) -> list[np.ndarray | None]:
    """``GT_transient_NAOMi_split.m``: the ground-truth transients of simulated neurons from their clean calcium
    traces. Per neuron: baseline ``mu`` = the median, ``sigma`` = the std; the typical peak height is the mode of
    a kernel density of the trace's peaks (padded with its minimum at both ends) of prominence ``sigma``; the
    threshold is ``mu + peaks_avg / 2``; transients are the runs above it, split at the valleys between peaks of
    prominence ``peaks_avg / 2`` and height ``thred`` (as ``detect``), returned as ``[start, end, 1]`` 1-based
    rows."""
    x = np.atleast_2d(np.asarray(calcium, dtype=float))
    out: list[np.ndarray | None] = []
    for row in x:
        mu = float(np.median(row))
        sigma = float(row.std(ddof=0))
        padded = np.concatenate([[row.min()], row, [row.min()]])
        locs, _ = find_peaks(padded, prominence=sigma)
        peaks = padded[locs]
        if peaks.size == 0:
            out.append(np.zeros((0, 3), dtype=int))
            continue
        peaks_avg = ksd_mode(peaks) - mu
        thred = mu + peaks_avg / 2
        above = row > thred
        d = np.diff(np.concatenate([[0], above.astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        plocs, _ = find_peaks(row, prominence=peaks_avg / 2, height=thred)
        plocs = np.concatenate([[0], plocs, [row.size - 1]])
        pk = np.zeros(row.size, dtype=bool)
        pk[plocs] = True
        rows = []
        for s, e in zip(starts, ends, strict=True):
            p_ = np.flatnonzero(pk[s:e + 1]) + s
            if p_.size == 0:
                continue
            if p_.size == 1:
                rows.append((s, e))
                continue
            bounds = [s]
            for a, b in zip(p_[:-1], p_[1:], strict=True):
                valley = a + int(np.argmin(row[a:b + 1]))
                bounds += [valley - 1, valley + 1]
            bounds.append(e)
            rows += [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2)]
        arr = np.array(rows, dtype=int).reshape(-1, 2) + 1          # 1-based, as the labelled files
        valid = np.ones(len(arr), dtype=int)
        out.append(np.column_stack([arr, valid]) if len(arr) else np.zeros((0, 3), int))
    return out
