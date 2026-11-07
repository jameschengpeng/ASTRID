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

import numpy as np
from scipy.signal import lfilter


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
    restart (Beck and Teboulle 2009). Starts from (s0, z0) and never returns a point worse than it."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    q = initial_responses(n, k)
    lip = weight * (k.gain ** 2 + float(np.sum(q ** 2)))
    s = np.zeros(n) if s0 is None else np.maximum(np.asarray(s0, dtype=np.float64), 0)
    z = np.zeros(2) if z0 is None else np.maximum(np.asarray(z0, dtype=np.float64), 0)
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
