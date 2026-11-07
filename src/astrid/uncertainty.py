"""The uncertainty of a target's coefficient with fixed footprints, one of the outputs whose stability ends a
fit (``solver.target_outputs``; Supplementary, "Optimization").

With fixed footprints and working weights W, whiten the target's footprint and the others', and project the
others out:

    a = W^1/2 H_k,  B = W^1/2 H_-k,  u = (I - B B^+) a,  VIF_k = ||a||^2 / ||u||^2   (geometric, under W).

The fixed-footprint linear estimator of the target's coefficient is c_k = l^T y with l = W^1/2 u / ||u||^2
(unbiased: l^T H = e_k). Its variance under the noise covariance Sigma is the sandwich l^T Sigma l; without
Sigma it is reported assuming Sigma = W^-1. If u = 0 (within a relative rank tolerance), the target cannot be
told apart from the other components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def whitener(w: np.ndarray) -> np.ndarray:
    """W^1/2 as a (P, P) matrix, from a (P,) diagonal or a symmetric positive definite (P, P) W."""
    if w.ndim == 1:
        return np.diag(np.sqrt(w))
    vals, vecs = np.linalg.eigh(w)
    return (vecs * np.sqrt(np.maximum(vals, 0))) @ vecs.T


@dataclass(frozen=True)
class Uncertainty:
    variance: float          # per-frame variance of the target's coefficient; inf if indistinguishable
    vif: float               # geometric, under the working weights; >= 1; inf if indistinguishable
    distinguishable: bool
    valid: bool = True
    reason: str = ""         # why the result is invalid or indistinguishable
    covariance: str = ""     # "given" (sandwich with Sigma) or "assumed W^-1"
    estimator: np.ndarray | None = None   # (P,) l, with c_k = l^T (y - m)


def invalid_weights(w: np.ndarray, p: int) -> str:
    if w.shape not in ((p,), (p, p)) or not np.all(np.isfinite(w)):
        return "weights not finite or of the wrong shape"
    if w.ndim == 1:
        return "" if np.all(w > 0) else "weights not positive"
    if not np.allclose(w, w.T) or np.linalg.eigvalsh(w)[0] <= 0:
        return "weights not symmetric positive definite"
    return ""


def invalid_covariance(sigma: np.ndarray, p: int) -> str:
    """Why the noise covariance ``sigma`` cannot be used for P = ``p`` pixels, or "" if it can."""
    if sigma.shape != (p, p) or not np.all(np.isfinite(sigma)):
        return "noise covariance not finite or of the wrong shape"
    if not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(0.5 * (sigma + sigma.T))[0] < \
            -1e-10 * max(1.0, float(np.abs(sigma).max())):
        return "noise covariance not symmetric positive semi-definite"
    return ""


def target_uncertainty(h: np.ndarray, w: np.ndarray, k: int, sigma: np.ndarray | None = None,
                       rank_tol: float = 1e-10, *, sigma_problem: str | None = None) -> Uncertainty:
    """``sigma_problem``: ``invalid_covariance(sigma, P)`` if the caller has computed it already (a fit
    checks its fixed ``sigma`` once, not at every block)."""
    p = h.shape[0]
    problem = "footprints not finite" if not np.all(np.isfinite(h)) else invalid_weights(w, p)
    if not problem and sigma is not None:
        problem = invalid_covariance(sigma, p) if sigma_problem is None else sigma_problem
    if problem:
        return Uncertainty(float("inf"), float("inf"), False, valid=False, reason=problem)
    wh = whitener(w)
    a = wh @ h[:, k]
    others = np.delete(h, k, axis=1)
    if others.shape[1]:
        b = wh @ others
        u = a - b @ np.linalg.lstsq(b, a, rcond=None)[0]
    else:
        u = a
    nu2, na2 = float(u @ u), float(a @ a)
    if not (np.isfinite(nu2) and np.isfinite(na2)):
        return Uncertainty(float("inf"), float("inf"), False, valid=False, reason="non-finite projection")
    if na2 == 0 or nu2 <= rank_tol * na2:
        return Uncertainty(float("inf"), float("inf"), False,
                           reason="indistinguishable from the other components")
    estimator = wh @ u / nu2
    if sigma is None:
        return Uncertainty(1 / nu2, na2 / nu2, True, covariance="assumed W^-1", estimator=estimator)
    variance = float(estimator @ sigma @ estimator)
    if not np.isfinite(variance) or variance < 0:
        return Uncertainty(float("inf"), float("inf"), False, valid=False, reason="invalid sandwich variance")
    return Uncertainty(variance, na2 / nu2, True, covariance="given", estimator=estimator)
