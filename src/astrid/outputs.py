"""What a fit delivers for its target ROI (Methods and Supplementary, "The demixed trace").

- ``demixed_trace``: the ROI average of the baseline-subtracted movie minus every other component's fitted
  contribution (the residual is kept), with its dF/F0 relative to the ROI average of the fixed baseline;
- ``roi_noise_sd``: the s.d. of the noise in an ROI average (the scale below which changes of a trace are
  noise; used by the convergence check);
- ``other_contributions``: the ROI-average contribution of each other kind of component; with the demixed
  trace they add up to the ROI average of the baseline-subtracted movie;
- ``support_trace``: a component's contribution averaged over its own spatial support (as in the paper's Fig. 1E).
"""

from __future__ import annotations

import numpy as np

from astrid.model import Patch, State


def demixed_trace(patch: Patch, state: State, k: int, roi: np.ndarray,
                  r: int) -> tuple[np.ndarray, np.ndarray]:
    """(numerator, dF/F0) of target ``k`` in recording ``r``; ``roi`` is a (P,) mask."""
    others = [i for i in range(patch.n_components) if i != k]
    rest = patch.centred(r) - state.h[:, others] @ state.c[r][others]
    numerator = rest[roi].mean(axis=0)
    baseline = np.broadcast_to(patch.m[r], patch.y[r].shape)[roi].mean(axis=0)
    return numerator, numerator / baseline


def roi_noise_sd(patch: Patch, roi: np.ndarray, sigma: np.ndarray | None = None) -> float:
    """Per-frame s.d. of the noise in an ROI average: under ``sigma`` if given, else assuming Sigma = W^-1."""
    m = roi / roi.sum()
    if sigma is not None:
        var = float(m @ sigma @ m)
    elif patch.w.ndim == 1:
        var = float(np.sum(m ** 2 / patch.w))
    else:
        var = float(m @ np.linalg.solve(patch.w, m))
    return float(np.sqrt(max(var, 0.0)))


OTHER_KINDS = {"neighbours": ("neuron",), "unclassified": ("unclassified",), "astro": ("astro",),
               "neuropil": ("neuropil",)}


def other_contributions(patch: Patch, state: State, k: int, roi: np.ndarray, r: int) -> dict[str, np.ndarray]:
    """ROI-average contribution in recording ``r`` of each kind of component other than the target ``k``:
    "neighbours" (the other neurons), "unclassified", "astro" and "neuropil", for the kinds the patch has. By
    ``demixed_trace``, the demixed trace plus these contributions is the ROI average of the
    baseline-subtracted movie."""
    out = {}
    for name, kinds in OTHER_KINDS.items():
        idx = [j for j, c in enumerate(patch.components) if j != k and c.kind in kinds]
        if idx:
            out[name] = state.h[roi][:, idx].mean(axis=0) @ state.c[r][idx]
    return out


def support_trace(patch: Patch, state: State, j: int, r: int) -> np.ndarray:
    """Component ``j``'s dF/F0 averaged over its spatial support in recording ``r`` (as in the paper's
    Fig. 1E): its mean dF there, mean(h_j) c_j = c_j / |P_j| (the footprint sums to 1 on its support), over the
    support's mean baseline."""
    p = patch.components[j].support
    base = np.broadcast_to(patch.m[r], patch.y[r].shape)[p].mean(axis=0)
    return state.h[p, j].mean() * state.c[r][j] / base
