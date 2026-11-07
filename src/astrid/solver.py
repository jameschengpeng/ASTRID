"""Fitting the patch model by alternating updates (Supplementary, "Optimization").

- Initialization: footprints uniform on their supports, traces from weighted least squares (Supplementary,
  "Weights of the objective"). The penalty scales are computed from this start and then held fixed
  (``model.Penalties``).
- Footprint update (H-step): the spatial quadratic program at fixed traces, with the unit-sum,
  non-negativity and support constraints and the neuropil and evidence penalties, by projected FISTA with
  monotone restart. The neuropil cap's level is read at the step's start (``neuropil_cap_levels``).
- Trace update (C-step): one HALS sweep per recording. Row i sees y_i = C_i + H_i^T W E / q_i with
  q_i = H_i^T W H_i, and minimizes (q_i/2)||C_i - y_i||^2 plus its own penalty:
  - neuron-like rows: deconvolution, never worse than the current row;
  - astrocytes: an exact DCT shrinkage;
  - neuropil: y_i itself.
- Stopping: on the stability of what each target delivers (``fit``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astrid import astro_prior
from astrid.kernels import Kernel, calcium, deconvolution_objective, deconvolve
from astrid.model import (
    ASTRO_LIKE,
    NEURON_LIKE,
    SMOOTH,
    Assessment,
    Params,
    Patch,
    Penalties,
    State,
    objective,
    roughness,
    roughness_gradient,
)
from astrid.outputs import demixed_trace, roi_noise_sd
from astrid.uncertainty import invalid_covariance, target_uncertainty

TINY = np.finfo(np.float64).tiny      # the smallest normal float64


def nonsubnormal(x: np.ndarray) -> np.ndarray:
    """``x`` with its subnormal entries (0 < |x| < ``TINY``, the decayed tails of the calcium traces) set to
    zero, as an operand of the large products only; the state keeps its values. Speed alone: on Intel CPUs a
    product with subnormal operands ran 12 times slower, and each such term, below 2.3e-308 times its
    partner, is lost against the other terms of its sum, so the products are the same."""
    return np.where(np.abs(x) < TINY, 0.0, x)


def simplex_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.arange(1, v.size + 1)
    rho = int(np.nonzero(u - css / k > 0)[0][-1])
    return np.maximum(v - css[rho] / (rho + 1), 0)


def capped_simplex_projection(v: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {0 <= x <= cap, sum(x) = 1}, exact: x = clip(v - theta, 0, cap), with theta
    found by sweeping the sorted breakpoints of g(theta) = sum(clip(v - theta, 0, cap)), a continuous,
    non-increasing, piecewise-linear function (breakpoints v and v - cap). ``cap`` may be infinite (no
    bound); with no finite cap this is ``simplex_projection`` itself.

    Raises ValueError for an infeasible set: a negative or NaN cap, or finite caps summing to less than 1
    (only possible when every entry is capped)."""
    v = np.asarray(v, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    if cap.shape != v.shape:
        raise ValueError("cap must have one entry per value")
    if np.any(np.isnan(cap)) or np.any(cap < 0):
        raise ValueError("caps must be >= 0 (NaN is not a cap)")
    fin = np.isfinite(cap)
    if not fin.any():
        return simplex_projection(v)
    if fin.all() and float(np.sum(cap)) < 1:
        raise ValueError(f"infeasible caps: they sum to {float(np.sum(cap)):.6g} < 1")
    # events as theta decreases: at v_i an entry starts to rise from 0 (joins the free set); at v_i - cap_i
    # it reaches its cap (leaves the free set, its cap joins the constant part)
    t = np.concatenate([v, v[fin] - cap[fin]])
    dn = np.concatenate([np.ones(v.size), -np.ones(int(fin.sum()))])
    dv = np.concatenate([v, -v[fin]])
    dc = np.concatenate([np.zeros(v.size), cap[fin]])
    order = np.argsort(-t, kind="stable")
    t, n, s, c = t[order], np.cumsum(dn[order]), np.cumsum(dv[order]), np.cumsum(dc[order])
    lower = np.append(t[1:], -np.inf)                 # each interval [lower_k, t_k] has fixed sets
    with np.errstate(invalid="ignore"):
        g_lower = np.where(n > 0, s + c - n * lower, s + c)   # g at the interval's lower end
    reach = np.flatnonzero(g_lower >= 1)
    if reach.size == 0:                               # rounding at the boundary of feasibility
        raise ValueError("infeasible caps: g(theta) never reaches 1")
    k = int(reach[0])
    theta = (s[k] + c[k] - 1) / n[k] if n[k] > 0 else t[k]
    return np.minimum(np.maximum(v - theta, 0), cap)


def project_columns(h: np.ndarray, supports: np.ndarray, *,
                    ncaps: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Each column projected onto the simplex over its support. ``ncaps`` (the neuropil cap,
    ``neuropil_cap_levels``): per neuropil column, a cap (P,). Where the column's projection already meets
    it, that projection stands; otherwise the column is projected onto the capped simplex, which is then the
    exact projection."""
    out = np.zeros_like(h)
    for i in range(h.shape[1]):
        on = supports[:, i]
        out[on, i] = simplex_projection(h[on, i])
        if ncaps is not None and i in ncaps:
            cap = ncaps[i][on]
            if np.any(out[on, i] > cap):
                out[on, i] = capped_simplex_projection(h[on, i], cap)
    return out


def neuropil_cap_levels(patch: Patch, h: np.ndarray) -> dict[int, np.ndarray]:
    """For each neuropil column of ``h``, its cap (P,) (the paper's neuropil cap): on each target's ROI the
    column's largest value over the patch pixels outside that target's support (on the column's support), inf
    elsewhere; where two targets' ROIs meet, the lower. A column gets no entry when no target has a pixel
    outside its support."""
    targets = [c for c in patch.components if c.target]
    out: dict[int, np.ndarray] = {}
    for b in patch.kind_index("neuropil"):
        on = patch.components[b].support
        cap = np.full(patch.n_pixels, np.inf)
        for t in targets:
            outside = on & ~t.support
            if outside.any():
                roi = t.roi_mask & on
                cap[roi] = np.minimum(cap[roi], float(h[outside, b].max()))
        if np.isfinite(cap).any():
            out[b] = cap
    return out


def cap_levels(patch: Patch, h: np.ndarray, params: Params) -> dict[int, np.ndarray] | None:
    """The neuropil caps of an H-step (``neuropil_cap_levels``); None when the cap is off."""
    return neuropil_cap_levels(patch, h) if params.neuropil_cap else None


def high_pass_weights(patch: Patch, params: Params) -> list[np.ndarray]:
    return [astro_prior.highpass_weights(y.shape[1], params.fs, params.f_a) for y in patch.y]


def kernel_norm(kernel: Kernel) -> float:
    n = int(np.ceil(20 * max(1 / (1 - kernel.a_d), 1 / (1 - kernel.a_r))))
    impulse = np.zeros(n)
    impulse[0] = 1
    return float(np.linalg.norm(calcium(impulse, kernel)))


def penalties(patch: Patch, h: np.ndarray, c: list[np.ndarray], params: Params, kernel: Kernel) -> Penalties:
    """Absolute weights from the dimensionless kappas and scales of the starting point (``model``'s
    docstring; Supplementary, "Weights of the objective")."""
    k_all = patch.n_components
    q = np.einsum("pk,pk->k", h, patch.apply_w(h))
    wbar = float(np.mean(patch.w if patch.w.ndim == 1 else np.diag(patch.w)))
    cnorm2 = sum(np.sum(cr ** 2, axis=1) for cr in c)
    lam = {name: np.zeros(k_all) for name in ("s", "a", "b", "t")}
    hnorm = kernel_norm(kernel)
    for i, comp in enumerate(patch.components):
        if comp.kind in NEURON_LIKE:
            lam["s"][i] = params.kappa_s * np.sqrt(q[i]) * hnorm
        elif comp.kind == "astro":
            lam["a"][i] = params.kappa_a * q[i] / 2
            if patch.tagrfp:                         # no evidence map: no evidence penalty
                lam["t"][i] = params.kappa_t * wbar * cnorm2[i] / comp.support.sum()
        else:
            lam["b"][i] = params.kappa_b * wbar * cnorm2[i] / 16
    return Penalties(lam["s"], lam["a"], lam["b"], lam["t"])


def least_squares_traces(patch: Patch, h: np.ndarray) -> list[np.ndarray]:
    wh = patch.apply_w(h)
    gram_inv = np.linalg.pinv(h.T @ wh)
    return [gram_inv @ (wh.T @ patch.centred(r)) for r in range(len(patch.y))]


def uniform_footprints(patch: Patch) -> np.ndarray:
    sup = patch.supports.astype(np.float64)
    return sup / sup.sum(axis=0)


def canonical_penalties(patch: Patch, params: Params) -> Penalties:
    """The one set of absolute weights for a patch and a setting of the kappas: scales from the uniform
    footprints and their least-squares traces (the paper's fixed reference starting point)."""
    kernel = Kernel.from_taus(params.tau_r, params.tau_d, params.fs)
    h = uniform_footprints(patch)
    return penalties(patch, h, least_squares_traces(patch, h), params, kernel)


def initial_state(patch: Patch, params: Params, kernel: Kernel, weights: list[np.ndarray],
                  pen: Penalties) -> State:
    """Footprints uniform on their supports, projected onto the simplex, traces from weighted least squares,
    each row then brought into its model class under the given penalties."""
    h = project_columns(uniform_footprints(patch), patch.supports)
    c = least_squares_traces(patch, h)
    q = np.einsum("pk,pk->k", h, patch.apply_w(h))
    s, z = [], []
    for r, cr in enumerate(c):
        sr, zr = np.zeros_like(cr), np.zeros((patch.n_components, 2))
        for i, comp in enumerate(patch.components):
            if comp.kind in NEURON_LIKE:
                d = deconvolve(cr[i], q[i], pen.lam_s[i], kernel, n_iter=params.n_deconv)
                cr[i], sr[i], zr[i] = d.c, d.s, d.z
            elif comp.kind in ASTRO_LIKE:
                cr[i] = astro_prior.shrink(cr[i], q[i], pen.lam_a[i], weights[r])
        s.append(sr)
        z.append(zr)
    return State(h, c, s, z, pen)


def spatial_objective(h: np.ndarray, gram: np.ndarray, cross: np.ndarray, patch: Patch,
                      pen: Penalties) -> float:
    """The H-dependent part of the objective, from the sufficient statistics gram = sum_r C C^T and
    cross = sum_r (Y - M) C^T."""
    f = 0.5 * float(np.sum(patch.apply_w(h) * (h @ gram))) - float(np.sum(patch.apply_w(cross) * h))
    for j in patch.kind_index("astro"):
        f += pen.lam_t[j] * float(np.sum((1 - patch.tau) * h[:, j]))
    for b in patch.kind_index(*SMOOTH):
        f += pen.lam_b[b] * roughness(h[:, b], patch.edges)
    return f


def sufficient_statistics(patch: Patch, state: State) -> tuple[np.ndarray, np.ndarray]:
    """gram = sum_r C_r C_r^T and cross = sum_r (Y_r - M_r) C_r^T."""
    gram = sum(c @ c.T for c in state.c)
    cross = sum(patch.centred(r) @ nonsubnormal(state.c[r]).T for r in range(len(patch.y)))
    return gram, cross


def data_term_from_statistics(patch: Patch, h: np.ndarray, gram: np.ndarray, cross: np.ndarray) -> float:
    """The data term for footprints ``h`` and the traces behind ``gram`` and ``cross``, without forming the
    residual: 1/2 E - <W cross, H> + 1/2 <W H, H gram>, E = ``patch.weighted_energy``."""
    return float(0.5 * patch.weighted_energy - np.sum(patch.apply_w(cross) * h)
                 + 0.5 * np.sum(patch.apply_w(h) * (h @ gram)))


def h_step(patch: Patch, state: State, params: Params, gram: np.ndarray | None = None,
           cross: np.ndarray | None = None) -> np.ndarray:
    """Projected FISTA on the spatial problem. The neuropil cap's levels are read from the footprints at the
    step's start (``neuropil_cap_levels``) and held for the step; a starting neuropil column that breaks its
    cap is first projected onto it, and every step projects under it (``project_columns``' ``ncaps``)."""
    pen = state.pen
    if gram is None or cross is None:
        gram, cross = sufficient_statistics(patch, state)
    astro, neuropil = patch.kind_index("astro"), patch.kind_index(*SMOOTH)
    lip = patch.w_max * float(np.linalg.eigvalsh(gram)[-1]) + 16 * float(pen.lam_b.max(initial=0)) + 1e-12
    wcross = patch.apply_w(cross)

    def gradient(h: np.ndarray) -> np.ndarray:
        g = patch.apply_w(h @ gram) - wcross
        for j in astro:
            g[:, j] += pen.lam_t[j] * (1 - patch.tau)
        for b in neuropil:
            g[:, b] += pen.lam_b[b] * roughness_gradient(h[:, b], patch.edges)
        return g

    h = state.h.copy()
    ncaps = cap_levels(patch, h, params)
    for b, cap in (ncaps or {}).items():
        on = patch.supports[:, b]
        bound = cap[on]
        if np.any(h[on, b] > bound):
            h[on, b] = capped_simplex_projection(h[on, b], bound)
    f = spatial_objective(h, gram, cross, patch, pen)
    best_h, best_f = h, f
    v, t = h.copy(), 1.0
    for _ in range(params.n_h):
        h_new = project_columns(v - gradient(v) / lip, patch.supports, ncaps=ncaps)
        f_new = spatial_objective(h_new, gram, cross, patch, pen)
        if f_new > f:                            # restart momentum from the last accepted point
            v, t = h.copy(), 1.0
            continue
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        v = h_new + ((t - 1) / t_new) * (h_new - h)
        decrease = f - f_new
        h, f, t = h_new, f_new, t_new
        if f < best_f:
            best_h, best_f = h, f
        if decrease <= params.tol * max(1.0, abs(f)):
            break
    return best_h


def c_step(patch: Patch, state: State, params: Params, kernel: Kernel, weights: list[np.ndarray]) -> float:
    """One HALS sweep per recording; returns the data term of the updated state.

    Row i's projected residual H_i^T W E, with E = (Y - M) - H C at the sweep's current C, equals
    A_i - B_i C for A = (W H)^T (Y - M) (K, T) and B = (W H)^T H (K, K); the data term 1/2 <E, W E> equals
    1/2 E_0 - <A, C> + 1/2 <C, B C>, E_0 = ``patch.weighted_energy``, so the (P, T) residual is never formed."""
    h, pen = state.h, state.pen
    wh = patch.apply_w(h)
    q = np.einsum("pk,pk->k", h, wh)
    b = wh.T @ h
    data = 0.5 * patch.weighted_energy
    for r in range(len(patch.y)):
        a = wh.T @ patch.centred(r)
        c = state.c[r]
        for i, comp in enumerate(patch.components):
            old = c[i].copy()
            y_i = old + (a[i] - b[i] @ c) / q[i]
            if comp.kind in NEURON_LIKE:
                d = deconvolve(y_i, q[i], pen.lam_s[i], kernel, s0=state.s[r][i], z0=state.z[r][i],
                               n_iter=params.n_deconv)
                if d.objective > deconvolution_objective(y_i, old, state.s[r][i], q[i], pen.lam_s[i]):
                    continue                     # guard; deconvolve never returns a worse point
                new = d.c
                state.s[r][i], state.z[r][i] = d.s, d.z
            elif comp.kind in ASTRO_LIKE:
                new = astro_prior.shrink(y_i, q[i], pen.lam_a[i], weights[r])
            else:
                new = y_i
            c[i] = new
        data -= float(np.sum(a * c)) - 0.5 * float(np.sum(c * (b @ c)))
    return data


def relative_change(a: np.ndarray, ref: np.ndarray, floor: float = 0.0) -> float:
    """||a - ref|| / max(||ref||, floor): 0 for identical arrays (zeros included), infinite against a zero
    scale. ``floor`` is an absolute scale (a noise norm), so a weak output is judged against its noise."""
    diff = float(np.linalg.norm(np.asarray(a) - np.asarray(ref)))
    if diff == 0:
        return 0.0
    scale = max(float(np.linalg.norm(ref)), floor)
    return diff / scale if scale > 0 else float("inf")


def scalar_change(a: float, ref: float, floor: float = 0.0) -> float:
    """|a - ref| / max(|ref|, floor) for scalars that may be infinite: 0 when equal, infinite when only one is
    finite."""
    if a == ref:
        return 0.0
    if not (np.isfinite(a) and np.isfinite(ref)):
        return float("inf")
    scale = max(abs(ref), floor)
    return abs(a - ref) / scale if scale > 0 else float("inf")


@dataclass(frozen=True)
class TargetOutputs:
    """What the convergence check monitors, per target:
    - ``delivered``: per recording, the ROI average of the data minus every other component (residual kept);
    - ``trace`` and ``footprint``: its own C_k per recording and H_k;
    - ``vif`` and ``variance``: the fixed-footprint reference (variance under ``sigma``, else W^-1);
    - ``snr``: per recording, the s.d. of C_k over the coefficient's s.d.;
    - the floors: per recording, the norm of the noise alone in the delivered trace and in C_k."""
    delivered: tuple[np.ndarray, ...]
    trace: tuple[np.ndarray, ...]
    footprint: np.ndarray
    vif: float
    variance: float
    snr: tuple[float, ...]
    delivered_floor: tuple[float, ...]
    trace_floor: tuple[float, ...]


def target_outputs(patch: Patch, state: State, k: int, sigma: np.ndarray | None = None, *,
                   sigma_problem: str | None = None) -> TargetOutputs:
    """``sigma_problem``: as ``target_uncertainty``'s (``fit`` checks ``sigma`` once)."""
    roi = patch.components[k].roi_mask
    u = target_uncertainty(state.h, patch.w, k, sigma=sigma, sigma_problem=sigma_problem)
    sd = np.sqrt(u.variance) if u.distinguishable else float("inf")
    roi_sd = roi_noise_sd(patch, roi, sigma)
    lengths = [y.shape[1] for y in patch.y]
    trace = tuple(state.c[r][k].copy() for r in range(len(patch.y)))
    return TargetOutputs(
        delivered=tuple(demixed_trace(patch, state, k, roi, r)[0] for r in range(len(patch.y))),
        trace=trace, footprint=state.h[:, k].copy(), vif=u.vif, variance=u.variance,
        snr=tuple(float(np.std(c)) / sd if sd > 0 else float("inf") for c in trace),
        delivered_floor=tuple(np.sqrt(n) * roi_sd for n in lengths),
        trace_floor=tuple(np.sqrt(n) * sd if np.isfinite(sd) else 0.0 for n in lengths))


def output_change(new: TargetOutputs, old: TargetOutputs) -> dict[str, float]:
    """Relative change of each monitored series: every output, and every recording of the per-recording
    outputs ("delivered[0]", "trace[0]", "snr[0]", ..., "footprint", "vif", "variance"). Each series is
    judged and extrapolated on its own, so neither a strong recording nor a fast-settling output can hide
    another. Traces are relative to the newer norm or their noise norm, whichever is larger; SNRs below 1
    count in absolute units."""
    out = {}
    for r in range(len(new.delivered)):
        out[f"delivered[{r}]"] = relative_change(old.delivered[r], new.delivered[r], new.delivered_floor[r])
        out[f"trace[{r}]"] = relative_change(old.trace[r], new.trace[r], new.trace_floor[r])
        out[f"snr[{r}]"] = scalar_change(old.snr[r], new.snr[r], 1.0)
    out["footprint"] = relative_change(old.footprint, new.footprint)
    out["vif"] = scalar_change(old.vif, new.vif, 1.0)
    out["variance"] = scalar_change(old.variance, new.variance)
    return out


def remaining_change(changes: list[float]) -> float:
    """Change still to come if the block changes keep shrinking geometrically, at their mean ratio over the
    last two steps (one step after only two blocks): change x ratio / (1 - ratio). Infinite after a single
    block, or when they do not shrink (``Assessment``)."""
    if len(changes) < 2 or not np.isfinite(changes[-1]):
        return float("inf")
    if changes[-1] == 0:
        return 0.0
    m = min(len(changes) - 1, 2)
    first = changes[-1 - m]
    ratio = (changes[-1] / first) ** (1 / m) if first > 0 else float("inf")
    return changes[-1] * ratio / (1 - ratio) if ratio < 1 else float("inf")


def projected_remaining(series: dict[str, list[float]]) -> float:
    """The largest remaining change over the monitored series, each extrapolated from its own history, so a
    fast-settling output cannot mask a steady drift in another. An estimate from recent decay, not a bound."""
    return max((remaining_change(v) for v in series.values()), default=float("inf"))


def fit(patch: Patch, params: Params, sigma: np.ndarray | None = None) -> State:
    """Alternate H- and C-steps from the canonical start. ``state.stop_reason`` records how the fit ended:
    - "targets converged": for every target, each monitored series (``TargetOutputs``: demixed trace, C_k
      and SNR per recording, H_k, VIF, coefficient variance under ``sigma``) changed by less than
      ``params.tol_target`` over the last block of ``params.block`` iterations, and each series' own
      geometric extrapolation projects less than that to come (``Assessment``);
    - "converged", for a patch without targets only: the objective's relative decrease over an iteration fell
      below ``params.tol``;
    - "iteration limit": ``params.n_outer`` iterations were made.

    With targets, a converged objective does not end the fit: a target can still drift along a direction
    where the objective is flat, so only its own outputs can show that it has settled. Convergence is about
    the optimization only; it does not show that the fluorescence was allocated correctly. The objective
    after every step is in ``state.history``."""
    kernel = Kernel.from_taus(params.tau_r, params.tau_d, params.fs)
    weights = high_pass_weights(patch, params)
    state = initial_state(patch, params, kernel, weights, canonical_penalties(patch, params))
    c_step(patch, state, params, kernel, weights)
    state.history.append(("init", objective(patch, state, weights)))
    state.stop_reason = "iteration limit"
    targets = [k for k, comp in enumerate(patch.components) if comp.target]
    sigma_problem = invalid_covariance(sigma, patch.n_pixels) if sigma is not None and targets else None
    previous = {k: target_outputs(patch, state, k, sigma, sigma_problem=sigma_problem) for k in targets}
    series: dict[int, dict[str, list[float]]] = {k: {} for k in targets}   # the block changes
    for it in range(params.n_outer):
        before = state.history[-1][1]
        gram, cross = sufficient_statistics(patch, state)
        state.h = h_step(patch, state, params, gram, cross)
        state.history.append(("h", objective(patch, state, weights,
                                             data_term_from_statistics(patch, state.h, gram, cross))))
        data = c_step(patch, state, params, kernel, weights)
        state.history.append(("c", objective(patch, state, weights, data)))
        state.iterations += 1
        if not targets and before - state.history[-1][1] <= params.tol * max(1.0, abs(before)):
            state.stop_reason = "converged"
            break
        if targets and (it + 1) % params.block == 0:
            current = {k: target_outputs(patch, state, k, sigma, sigma_problem=sigma_problem)
                       for k in targets}
            changes = {k: output_change(current[k], previous[k]) for k in targets}
            for k in targets:
                for name, value in changes[k].items():
                    series[k].setdefault(name, []).append(value)
            state.target_history.append(Assessment(
                state.iterations, changes, {k: projected_remaining(series[k]) for k in targets}))
            if all(state.target_converged(k, params.tol_target) for k in targets):
                state.stop_reason = "targets converged"
                break
            previous = current
    return state
