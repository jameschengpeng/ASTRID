"""The indicator's impulse response as an AR(2) kernel in cascade form, and non-negative deconvolution.

A rise stage feeds a decay stage, for t = 0, ..., T-1:

    x_t = a_r x_{t-1} + s_t,    c_t = a_d c_{t-1} + x_t,

the recurrence c_t = (a_d + a_r) c_{t-1} - a_d a_r c_{t-2} + s_t, whose impulse response is the paper's
difference of exponentials with k_0 = 1 (Supplementary, "Temporal models"). The initial states z = (c_{-1}, x_{-1})
are cascade coordinates: z >= 0 and s >= 0 give c >= 0, which non-negative lagged values (c_{-1}, c_{-2})
would not, because the second AR coefficient is negative. x_{-1} and the first drive trade exactly
(x_0 = a_r x_{-1} + s_0): only a_r x_{-1} + s_0 is identified, while c_{-1} is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from scipy.signal import lfilter

# ``deconvolve``'s FISTA loop runs compiled (``_fista``). False runs the reference loop
# (``_deconvolve_reference``) instead; the two give the same result bitwise (tests/test_compiled.py), so this
# switch is for tests and checks only.
COMPILED = True


@dataclass(frozen=True)
class Kernel:
    a_r: float      # rise root, exp(-dt / tau_r)
    a_d: float      # decay root, exp(-dt / tau_d)

    @classmethod
    def from_taus(cls, tau_r: float, tau_d: float, fs: float) -> Kernel:
        if not 0 < tau_r < tau_d:
            raise ValueError(f"need 0 < tau_r < tau_d, got {tau_r}, {tau_d}")
        return cls(float(np.exp(-1 / (fs * tau_r))), float(np.exp(-1 / (fs * tau_d))))

    @property
    def gammas(self) -> tuple[float, float]:
        """The AR(2) coefficients of the recurrence."""
        return self.a_d + self.a_r, -self.a_d * self.a_r

    @property
    def gain(self) -> float:
        """Sum of the impulse response: the steady calcium level per unit of constant drive."""
        return 1 / ((1 - self.a_r) * (1 - self.a_d))


def calcium(s: np.ndarray, k: Kernel, z: np.ndarray | None = None) -> np.ndarray:
    """c = G s + Q z for one trace; ``z = (c_{-1}, x_{-1})``."""
    s = np.asarray(s, dtype=np.float64)
    c0, x0 = (0.0, 0.0) if z is None else (float(z[0]), float(z[1]))
    x = lfilter([1.0], [1.0, -k.a_r], s, zi=[k.a_r * x0])[0]
    return lfilter([1.0], [1.0, -k.a_d], x, zi=[k.a_d * c0])[0]


FLUSH = 1e-200     # smaller responses don't matter; as subnormal floats they slow arithmetic ~50x


def initial_responses(n: int, k: Kernel) -> np.ndarray:
    """Q: the (n, 2) responses to unit initial states c_{-1} and x_{-1}. Their tails, below ``FLUSH``, are set
    to zero: left as subnormal floats they made every product with Q about 50 times slower."""
    zero = np.zeros(n)
    q = np.stack([calcium(zero, k, np.array([1.0, 0.0])), calcium(zero, k, np.array([0.0, 1.0]))], axis=1)
    q[np.abs(q) < FLUSH] = 0.0
    return q


def calcium_adjoint(g: np.ndarray, k: Kernel, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(G^T g, Q^T g). G is causal and Toeplitz, so G^T is G applied to the time-reversed signal."""
    g = np.asarray(g, dtype=np.float64)
    gs = calcium(g[::-1], k)[::-1]
    return gs, q.T @ g


@dataclass
class Deconvolved:
    s: np.ndarray       # drive, >= 0
    z: np.ndarray       # initial states (c_{-1}, x_{-1}), >= 0
    c: np.ndarray       # calcium, G s + Q z
    objective: float


def deconvolution_objective(y: np.ndarray, c: np.ndarray, s: np.ndarray, weight: float, lam: float) -> float:
    return float(0.5 * weight * np.sum((c - y) ** 2) + lam * np.sum(s))


def deconvolve(y: np.ndarray, weight: float, lam: float, k: Kernel, s0: np.ndarray | None = None,
               z0: np.ndarray | None = None, n_iter: int = 300, tol: float = 1e-9) -> Deconvolved:
    """min over s >= 0, z >= 0 of (weight/2) ||G s + Q z - y||^2 + lam * sum(s), by FISTA with monotone
    restart (Beck and Teboulle 2009). Starts from (s0, z0) and never returns a point worse than it.

    The loop runs compiled (``_fista``). It makes the reference loop's (``_deconvolve_reference``)
    floating-point operations in the same order:
    - lfilter's two first-order stages;
    - numpy's summation order;
    - the product q^T g through BLAS.
    So the result is the reference's bitwise (tests/test_compiled.py). ``COMPILED = False`` runs the
    reference."""
    y = np.ascontiguousarray(y, dtype=np.float64)
    n = y.size
    q = initial_responses(n, k)
    lip = weight * (k.gain ** 2 + float(np.sum(q ** 2)))
    s = np.zeros(n) if s0 is None else np.maximum(np.asarray(s0, dtype=np.float64), 0)
    z = np.zeros(2) if z0 is None else np.maximum(np.asarray(z0, dtype=np.float64), 0)
    if not COMPILED or n == 0:
        return _deconvolve_reference(y, weight, lam, k, s, z, q, lip, n_iter, tol)
    s_b, z_b, c_b, f_b = _fista(y, float(weight), float(lam), k.a_r, k.a_d, np.ascontiguousarray(q),
                                float(lip), np.ascontiguousarray(s), np.ascontiguousarray(z), int(n_iter),
                                float(tol), int(np.getbufsize()))
    return Deconvolved(s_b, z_b, c_b, float(f_b))


def _deconvolve_reference(y: np.ndarray, weight: float, lam: float, k: Kernel, s: np.ndarray, z: np.ndarray,
                          q: np.ndarray, lip: float, n_iter: int, tol: float) -> Deconvolved:
    """``deconvolve``'s FISTA loop in numpy and scipy's lfilter, as it ran before it was compiled. Kept as the
    reference that ``_fista`` is tested against; the fit calls it only with ``COMPILED = False``."""
    c = calcium(s, k, z)
    best = Deconvolved(s, z, c, deconvolution_objective(y, c, s, weight, lam))
    vs, vz, t = s.copy(), z.copy(), 1.0
    previous = best.objective
    for _ in range(n_iter):
        cv = calcium(vs, k, vz)
        gs, gz = calcium_adjoint(weight * (cv - y), k, q)
        s_new = np.maximum(vs - (gs + lam) / lip, 0)
        z_new = np.maximum(vz - gz / lip, 0)
        c_new = calcium(s_new, k, z_new)
        f_new = deconvolution_objective(y, c_new, s_new, weight, lam)
        if f_new > previous:                     # restart momentum from the last accepted point
            vs, vz, t = s.copy(), z.copy(), 1.0
            continue
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        vs = s_new + ((t - 1) / t_new) * (s_new - s)
        vz = z_new + ((t - 1) / t_new) * (z_new - z)
        s, z, t = s_new, z_new, t_new
        if f_new < best.objective:
            best = Deconvolved(s_new, z_new, c_new, f_new)
        if previous - f_new <= tol * max(1.0, abs(previous)):
            break
        previous = f_new
    return best


# ------------------------------------------------------------------------------------ the compiled FISTA loop
# ``_deconvolve_reference``'s steps with the same floating-point operations in the same order, so the result is
# the same bitwise (tests/test_compiled.py):
# - lfilter's two first-order stages, each y_t = a y_{t-1} + x_t, a product then a sum (numba makes no fused
#   multiply-add unless asked);
# - np.sum's order: pairwise sums within blocks of the iterator's buffer size (np.getbufsize()), added in
#   turn;
# - q^T g through BLAS, as numpy's matmul;
# - np.maximum(v, 0), which keeps a NaN.


@numba.njit(cache=True)
def _pairwise_sum(a: np.ndarray, lo: int, n: int) -> float:
    """numpy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src) of the contiguous a[lo:lo + n]."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    if n <= 128:
        r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
        r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
        i = 8
        end = n - (n % 8)
        while i < end:
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@numba.njit(cache=True)
def _numpy_sum(a: np.ndarray, bufsize: int) -> float:
    """np.sum of a contiguous float64 array: 0, then the pairwise sum of each block of ``bufsize`` added in
    turn."""
    total = 0.0
    start = 0
    n = a.size
    while start < n:
        m = min(bufsize, n - start)
        total += _pairwise_sum(a, start, m)
        start += m
    return total


@numba.njit(cache=True)
def _max0(v: float) -> float:
    """np.maximum(v, 0) for one value: v unless it is below 0; a NaN stays a NaN."""
    return v if (v >= 0.0 or v != v) else 0.0


@numba.njit(cache=True)
def _calcium_into(s: np.ndarray, a_r: float, a_d: float, x0: float, c0: float, out: np.ndarray) -> None:
    """``calcium(s, k, (c0, x0))`` into ``out``: lfilter's rise stage from x_{-1} = x0, then its decay stage
    from c_{-1} = c0, in one pass."""
    x = x0
    c = c0
    for i in range(s.size):
        x = a_r * x + s[i]
        c = a_d * c + x
        out[i] = c


@numba.njit(cache=True)
def _calcium_reversed_into(g: np.ndarray, a_r: float, a_d: float, out: np.ndarray) -> None:
    """``calcium(g[::-1], k)[::-1]`` (G^T g, zero initial states) into ``out``."""
    x = 0.0
    c = 0.0
    for i in range(g.size - 1, -1, -1):
        x = a_r * x + g[i]
        c = a_d * c + x
        out[i] = c


@numba.njit(cache=True)
def _objective_into(y: np.ndarray, c: np.ndarray, s: np.ndarray, weight: float, lam: float, tmp: np.ndarray,
                    bufsize: int) -> float:
    """``deconvolution_objective``: 0.5 weight sum((c - y)^2) + lam sum(s), with ``tmp`` as its (c - y)^2."""
    for i in range(y.size):
        d = c[i] - y[i]
        tmp[i] = d * d
    return 0.5 * weight * _numpy_sum(tmp, bufsize) + lam * _numpy_sum(s, bufsize)


@numba.njit(cache=True)
def _fista(y: np.ndarray, weight: float, lam: float, a_r: float, a_d: float, q: np.ndarray, lip: float,
           s: np.ndarray, z: np.ndarray, n_iter: int, tol: float, bufsize: int):
    """``_deconvolve_reference``'s loop from the starting point (s, z), which it overwrites; returns the best
    point's (s, z, c, objective)."""
    n = y.size
    q_t = q.T
    tmp = np.empty(n)
    c = np.empty(n)
    _calcium_into(s, a_r, a_d, z[1], z[0], c)
    best_s, best_z, best_c = s.copy(), z.copy(), c.copy()
    best_f = _objective_into(y, c, s, weight, lam, tmp, bufsize)
    vs, vz = s.copy(), z.copy()
    t = 1.0
    previous = best_f
    cv, g, gs = np.empty(n), np.empty(n), np.empty(n)
    s_new, c_new, z_new = np.empty(n), np.empty(n), np.empty(2)
    for _ in range(n_iter):
        _calcium_into(vs, a_r, a_d, vz[1], vz[0], cv)
        for i in range(n):
            g[i] = weight * (cv[i] - y[i])
        _calcium_reversed_into(g, a_r, a_d, gs)
        gz = np.dot(q_t, g)                          # BLAS, as numpy's q.T @ g
        for i in range(n):
            s_new[i] = _max0(vs[i] - (gs[i] + lam) / lip)
        z_new[0] = _max0(vz[0] - gz[0] / lip)
        z_new[1] = _max0(vz[1] - gz[1] / lip)
        _calcium_into(s_new, a_r, a_d, z_new[1], z_new[0], c_new)
        f_new = _objective_into(y, c_new, s_new, weight, lam, tmp, bufsize)
        if f_new > previous:                         # restart momentum from the last accepted point
            vs[:] = s
            vz[:] = z
            t = 1.0
            continue
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        m = (t - 1) / t_new
        for i in range(n):
            vs[i] = s_new[i] + m * (s_new[i] - s[i])
        vz[0] = z_new[0] + m * (z_new[0] - z[0])
        vz[1] = z_new[1] + m * (z_new[1] - z[1])
        s[:] = s_new
        z[:] = z_new
        t = t_new
        if f_new < best_f:
            best_s[:] = s_new
            best_z[:] = z_new
            best_c[:] = c_new
            best_f = f_new
        scale = abs(previous)                        # max(1.0, abs(previous)), as Python's max
        if not scale > 1.0:
            scale = 1.0
        if previous - f_new <= tol * scale:
            break
        previous = f_new
    return best_s, best_z, best_c, best_f


def warm_up() -> None:
    """Compile (or load from numba's cache) the FISTA loop for the arguments ``deconvolve`` passes, so that
    processes forked afterwards inherit it instead of each compiling it."""
    if COMPILED:
        deconvolve(np.linspace(0.0, 1.0, 16), 1.0, 0.1, Kernel.from_taus(0.05, 0.4, 30.9), n_iter=2)
