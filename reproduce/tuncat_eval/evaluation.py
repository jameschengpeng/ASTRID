"""The F1 scorer of D2 (ABO), D4 (NAOMi) and D5 (one-photon) as the paper ran it (GPL-2: see this folder's
``__init__.py`` and ``LICENSE``). It follows TUnCaT's ``eval_*.m`` scripts (Bao et al. 2022).

- ``scorer(dataset, video, archive)``: the scorer of one video, the ``ScorerFactory`` that
  ``astrid_repro.tables_public`` takes (``--scorer tuncat_eval.evaluation:scorer``). ``archive`` is the root of
  TUnCaT's figshare archive. The traces are filtered with the dataset's template (ABO: the GCaMP6f spike
  template; one-photon: its own; NAOMi: the 100-Hz GCaMP6f template resampled to the frame rate) and scored at
  every threshold (``transients.score``: the ksd baseline, the PSD noise). The labels are TUnCaT's manual ones
  (ABO, one-photon) or, on NAOMi, the transients of the clean traces filtered with the same template
  (``transients.gt_transients_from_clean``).
- ``cnmf_fill(mat, masks)``: ``eval_NAOMi_CNMF.m``'s rule for a CNMF result in which CaImAn removed components
  (``--cnmf-fill tuncat_eval.evaluation:cnmf_fill``). The missing neurons are found by walking the Jaccard
  distance between the masks and the thresholded footprints down the diagonal, and get all-zero traces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy.interpolate import interp1d

from tuncat_eval import transients as tr

NAOMI_TEMPLATE = "filter_template 100Hz GCaMP6f_ind_con=10.mat"
NAOMI_FPS, TEMPLATE_FS = 30.0, 100.0
RAW_TEMPLATES = {"ABO": "GCaMP6f_spike_tempolate_mean.mat", "1p": "1P_spike_tempolate.mat"}


def _matlab_round(x: np.ndarray) -> np.ndarray:
    """MATLAB ``round``: halves away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def naomi_kernel(path: str | Path, fs: float = NAOMI_FPS) -> np.ndarray:
    """The evaluation kernel at ``fs`` (``eval_NAOMi_*.m``): the 100-Hz template placed with its peak at time 0,
    linearly interpolated (extrapolated at the ends) on the ``fs`` grid from round(-peak fs/100) to
    round((len - peak) fs/100) samples, kept where >= max e^-1, normalized to sum 1, time-reversed."""
    t = np.asarray(scipy.io.loadmat(path)["template"], dtype=float).ravel()
    peak = int(t.argmax())
    xp = (np.arange(t.size) - peak) / TEMPLATE_FS
    lo = _matlab_round(np.array(-peak * fs / TEMPLATE_FS))
    hi = _matlab_round(np.array((t.size - peak) * fs / TEMPLATE_FS))
    x = np.arange(lo, hi + 1) / fs
    f = interp1d(xp, t, kind="linear", fill_value="extrapolate")(x)
    f = f[f >= f.max() * np.exp(-1)]
    return (f / f.sum())[::-1]


def naomi_labels(archive: str | Path, video: str, kernel: np.ndarray) -> list[np.ndarray | None]:
    """A NAOMi video's ground-truth transients: its clean traces filtered with ``kernel``, then
    ``transients.gt_transients_from_clean``."""
    d = scipy.io.loadmat(Path(archive) / "data" / "NAOMi" / "GT Masks" / f"Traces_etc_{video}.mat")
    clean = np.asarray(d["clean_traces"], dtype=float)
    return tr.gt_transients_from_clean(tr.filtered(clean, kernel))


class Scorer:
    """One video's transient scoring at every threshold: ``(traces, thresholds) -> (recall, precision, f1,
    n_neurons, n_gt)``, the ``Scorer`` of ``astrid_repro.cv``."""

    def __init__(self, labels: list[np.ndarray | None], kernel: np.ndarray | None) -> None:
        self.labels, self.kernel = labels, kernel

    def __call__(self, x: np.ndarray, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        y = tr.filtered(x, self.kernel)
        mu, sigma = tr.baselines(y), tr.noise_psd(y)
        rec, pre, f1 = (np.zeros(thr.size) for _ in range(3))
        n_neurons = n_gt = 0
        for j, t in enumerate(thr):
            s = tr.score(y, self.labels, float(t), mu=mu, sigma=sigma)
            rec[j], pre[j], f1[j] = s.recall, s.precision, s.f1
            n_neurons, n_gt = s.n_neurons, s.n_gt
        return rec, pre, f1, n_neurons, n_gt


def scorer(dataset: str, video: str, archive: str | Path) -> Scorer:
    """The scorer of one video of ``dataset`` ("ABO", "NAOMi" or "1p"; module docstring)."""
    a = Path(archive)
    if dataset in RAW_TEMPLATES:
        from astrid_repro import public_data as pd_
        kernel = tr.template_from_archive(a, RAW_TEMPLATES[dataset])
        return Scorer(pd_.read_labels(pd_.labels_path(a, dataset, video)), kernel)
    if dataset == "NAOMi":
        kernel = naomi_kernel(a / "template" / NAOMI_TEMPLATE)
        return Scorer(naomi_labels(a, video, kernel), kernel)
    raise ValueError(f"no transient scorer for {dataset!r} (D3 is scored by correlation)")


def cnmf_fill_missing(m: dict[str, Any], masks: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """``eval_NAOMi_CNMF.m``: when CaImAn kept fewer components than seeded masks, the missing neurons are found
    by walking the Jaccard distance between the masks and the thresholded footprints (``A_thr`` if it has one
    column per component, else ``A_gt`` above 0.2 of each pixel's largest loading) down the diagonal, and get
    all-zero traces (C and YrA both). Returns (C + YrA, missing 1-based indices)."""
    import scipy.sparse as sp

    C, YrA = np.asarray(m["C_gt"], dtype=float), np.asarray(m["YrA_gt"], dtype=float)
    n_cells, n_remain = masks.shape[0], C.shape[0]
    if n_remain >= n_cells:
        return C + YrA, []
    if n_remain == 0:                       # nothing kept: every neuron is missing (the MATLAB walk has no start)
        return np.zeros((n_cells, C.shape[1] if C.ndim == 2 else 0)), list(range(1, n_cells + 1))
    A_thr = m["A_thr"]
    A_thr = A_thr.toarray() if sp.issparse(A_thr) else np.asarray(A_thr)
    if A_thr.ndim != 2 or A_thr.shape[1] != n_remain:
        A_gt = m["A_gt"]
        A_gt = A_gt.toarray() if sp.issparse(A_gt) else np.asarray(A_gt)
        A_thr = A_gt > A_gt.max(axis=1, keepdims=True) * 0.2
    A_thr = A_thr.astype(float)
    gt = masks.transpose(1, 2, 0).reshape(-1, n_cells, order="F").astype(float)   # (pixels, n), F-order
    inter = gt.T @ A_thr
    union = gt.sum(axis=0)[:, None] + A_thr.sum(axis=0)[None, :] - inter
    d = 1 - inter / union                                       # JaccardDist_2, (n_cells, n_remain)
    removed: list[int] = []
    di = 0
    iy = 0
    ix = 1
    for ix in range(1, n_remain + 1):                           # the MATLAB walk, 1-based
        iy = ix + di
        while iy < n_cells and d[iy - 1, ix - 1] >= d[iy, ix - 1]:
            di += 1
            removed.append(ix)
            iy += 1
    while iy < n_cells:
        di += 1
        removed.append(ix)
        iy += 1
    out = C + YrA
    for ind in reversed(removed):
        out = np.insert(out, ind - 1, 0.0, axis=0)
    return out, removed


def cnmf_fill(m: dict[str, Any], masks: np.ndarray) -> np.ndarray:
    """``cnmf_fill_missing``'s traces: the ``CnmfFill`` of ``astrid_repro.public_data``."""
    return cnmf_fill_missing(m, masks)[0]
