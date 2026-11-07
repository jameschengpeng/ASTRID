"""The patch model: data, components, parameters, state and objective.

For one patch with recordings r = 1..R (Methods, "Problem formulation"; Supplementary, "Joint fitting of
recordings"):

    min  sum_r 1/2 ||W^1/2 (Y_r - M_r - H C_r)||_F^2 + sum_{neuron-like i} lam_s,i sum_r sum_t s_i
         + sum_{astro j} lam_a,j sum_r Omega(C_j) + lam_b ||D H_np||^2
         + sum_{astro j} lam_t,j sum_p (1 - tau_p) H_j(p)
    s.t. H >= 0, columns summing to 1 on their supports; neuron-like rows C = G s + Q z, s >= 0, z >= 0;
         on each target's ROI the neuropil column at most its largest value outside the target's support.

Notation (code -> paper): Y - M -> X (M is the fixed baseline F0); H -> A (footprints); C -> S (traces);
s -> d (the drive); tau -> rho (the structural evidence); ``D`` takes differences between 4-neighbouring
pixels of the patch (the paper's nabla). Component 0 of a patch is the paper's component 1, the primary
neuronal source.

Penalty weights are dimensionless kappas times scales fixed once, at initialization, so the objective does
not change during a fit (q_i = H_i^T W H_i, wbar = mean W_pp, |h|_2 = norm of the kernel's impulse response;
paper names in brackets, Supplementary, "Weights of the objective"):
- lam_s,i = kappa_s sqrt(q_i) |h|_2  [lambda_d = kappa_d ...]: the drive's threshold in noise s.d.;
- lam_a,j = kappa_a q_j / 2  [lambda_a]: the row update shrinks each DCT coefficient by 1 / (1 + kappa_a w_k);
- lam_b = kappa_b wbar ||C_b||^2 / 16  [lambda_np = kappa_np ...]: roughness of the neuropil footprint;
- lam_t,j = kappa_t wbar ||C_j||^2 / |support_j|  [lambda_rho = kappa_rho ...]: astrocyte footprint where the
  structural evidence is weak.

The neuropil cap (Methods, "Spatial supports and constraints"): on each target's ROI the neuropil footprint is
capped at its largest value over the patch pixels outside the target's support, a level read at the start of
every footprint update (``solver.neuropil_cap_levels``). ``Params.neuropil_cap`` switches it off only to
reproduce the model without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from astrid import astro_prior

KINDS = ("neuron", "unclassified", "astro", "neuropil")
NEURON_LIKE = ("neuron", "unclassified")
ASTRO_LIKE = ("astro",)                          # time course under the astrocyte low-pass prior
SMOOTH = ("neuropil",)                           # footprint under the roughness penalty


@dataclass
class Component:
    name: str
    kind: str
    support: np.ndarray          # (P,) bool: the pixels of the patch the footprint may use
    target: bool = False
    roi: np.ndarray | None = None   # (P,) bool: a target's ROI for its demixed trace; default its support

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        self.support = np.asarray(self.support, dtype=bool)
        if self.support.ndim != 1 or not self.support.any():
            raise ValueError(f"{self.name}: support must be a non-empty 1-D mask")
        if self.roi is not None:
            self.roi = np.asarray(self.roi, dtype=bool)
            if self.roi.shape != self.support.shape or not self.roi.any():
                raise ValueError(f"{self.name}: roi must be a non-empty mask like the support")

    @property
    def roi_mask(self) -> np.ndarray:
        return self.support if self.roi is None else self.roi


@dataclass(frozen=True)
class Params:
    """The solver's parameters, in the code's names (``astrid.Settings`` holds them in the paper's names and
    makes this with the data's frame rate, ``Settings.params``)."""
    kappa_s: float = 0.0         # the paper's kappa_d: sparsity of the neuron-like drive, in noise s.d.
    kappa_a: float = 0.0         # kappa_a: shrinkage of astrocyte power above f_a
    kappa_b: float = 0.0         # kappa_np: roughness of the neuropil footprint
    kappa_t: float = 0.0         # kappa_rho: astrocyte footprint where the structural evidence is weak
    neuropil_cap: bool = False   # the neuropil cap (on in every result of the paper)
    f_a: float = 0.2             # Hz
    tau_r: float = 0.05          # s
    tau_d: float = 0.4           # s
    fs: float = 30.9             # Hz
    n_outer: int = 30
    n_h: int = 200
    n_deconv: int = 300
    tol: float = 1e-7            # relative objective decrease that ends the footprint steps
    block: int = 10              # iterations between checks of the targets' outputs
    tol_target: float = 1e-3     # relative change of every target output over a block (and to come)


@dataclass
class Patch:
    y: list[np.ndarray]          # per recording: (P, T_r) fluorescence
    m: list[np.ndarray]          # per recording: fixed baseline, broadcastable to (P, T_r)
    coords: np.ndarray           # (P, 2) integer pixel coordinates (row, col)
    w: np.ndarray                # noise weights: (P,) diagonal, or (P, P) inverse covariance
    tau: np.ndarray              # (P,) structural evidence in [0, 1]
    components: list[Component]
    # False: no evidence map, so tau carries no evidence and the astrocytes' evidence penalty is off
    tagrfp: bool = True

    def __post_init__(self) -> None:
        p = self.coords.shape[0]
        if any(y.shape[0] != p for y in self.y) or len(self.y) != len(self.m):
            raise ValueError("every recording needs (P, T) data and a baseline")
        if any(c.support.size != p for c in self.components):
            raise ValueError("component supports must have one entry per patch pixel")
        if self.w.shape not in ((p,), (p, p)):
            raise ValueError("w must be (P,) or (P, P)")

    @property
    def n_pixels(self) -> int:
        return int(self.coords.shape[0])

    @property
    def n_components(self) -> int:
        return len(self.components)

    @cached_property
    def centred_data(self) -> list[np.ndarray]:
        """Y_r - M_r per recording, computed once and read-only: the data and baseline must not change after
        the first use."""
        out = []
        for y, m in zip(self.y, self.m, strict=True):
            c = np.asarray(y, dtype=np.float64) - m
            c.flags.writeable = False
            out.append(c)
        return out

    def centred(self, r: int) -> np.ndarray:
        return self.centred_data[r]

    @cached_property
    def weighted_energy(self) -> float:
        """sum_r ||W^1/2 (Y_r - M_r)||_F^2, the constant of the data term."""
        return float(sum(np.sum(c * self.apply_w(c)) for c in self.centred_data))

    def apply_w(self, x: np.ndarray) -> np.ndarray:
        return self.w[:, None] * x if self.w.ndim == 1 else self.w @ x

    @cached_property
    def w_max(self) -> float:
        return float(self.w.max()) if self.w.ndim == 1 else float(np.linalg.eigvalsh(self.w)[-1])

    @cached_property
    def edges(self) -> np.ndarray:
        """(E, 2) index pairs of 4-neighbouring patch pixels."""
        index = {(int(r), int(c)): i for i, (r, c) in enumerate(self.coords)}
        pairs = [(i, index[(int(r), int(c) + 1)]) for i, (r, c) in enumerate(self.coords)
                 if (int(r), int(c) + 1) in index]
        pairs += [(i, index[(int(r) + 1, int(c))]) for i, (r, c) in enumerate(self.coords)
                  if (int(r) + 1, int(c)) in index]
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)

    def kind_index(self, *kinds: str) -> list[int]:
        return [i for i, c in enumerate(self.components) if c.kind in kinds]

    @cached_property
    def supports(self) -> np.ndarray:
        return np.stack([c.support for c in self.components], axis=1)


@dataclass(frozen=True)
class Penalties:
    """Absolute penalty weights per component (zero where a penalty does not apply), fixed for a fit."""
    lam_s: np.ndarray
    lam_a: np.ndarray
    lam_b: np.ndarray
    lam_t: np.ndarray


@dataclass(frozen=True)
class Assessment:
    """The targets' outputs at the end of a block of iterations, compared with the block's start (solver.fit).
    - ``changes``: per target, the relative change of each monitored series over the block (every output,
      per recording where it has one);
    - ``remaining``: per target, the largest over its series of the change still to come if that series
      keeps shrinking geometrically at its own recent ratio, i.e. change x ratio / (1 - ratio); infinite
      when a series does not shrink or when this fit has made only one block so far. A steady drift in any
      series never passes. It is an estimate from recent decay, not a bound."""
    iteration: int                               # cumulative outer iteration of the assessed state
    changes: dict[int, dict[str, float]]
    remaining: dict[int, float]

    def change(self, k: int) -> float:
        return max(self.changes[k].values()) if k in self.changes else float("inf")


@dataclass
class State:
    h: np.ndarray                                    # (P, K), columns sum to 1 on their supports
    c: list[np.ndarray]                              # per recording: (K, T_r)
    s: list[np.ndarray]                              # per recording: (K, T_r) drive, zero for other kinds
    z: list[np.ndarray]                              # per recording: (K, 2) initial states (c_{-1}, x_{-1})
    pen: Penalties
    history: list[tuple[str, float]] = field(default_factory=list)
    # after solver.fit: "targets converged" (every target's outputs stable over a block) or "iteration limit"
    stop_reason: str = "not fitted"
    target_history: list[Assessment] = field(default_factory=list)
    iterations: int = 0                              # outer iterations

    def assessment(self) -> Assessment | None:
        """The last assessment if it describes this state; None if the state changed after it (an iteration
        limit between block ends): the state is then unassessed."""
        last = self.target_history[-1] if self.target_history else None
        return last if last is not None and last.iteration == self.iterations else None

    def target_change(self, k: int) -> float:
        a = self.assessment()
        return a.change(k) if a is not None else float("inf")

    def target_remaining(self, k: int) -> float:
        a = self.assessment()
        return a.remaining.get(k, float("inf")) if a is not None else float("inf")

    def target_converged(self, k: int, tol: float) -> bool:
        """True when this state was assessed and target ``k``'s outputs changed by less than ``tol`` over the
        last block, with less than ``tol`` projected to come."""
        return self.target_change(k) < tol and self.target_remaining(k) < tol


def roughness(h: np.ndarray, edges: np.ndarray) -> float:
    return float(np.sum((h[edges[:, 0]] - h[edges[:, 1]]) ** 2)) if edges.size else 0.0


def roughness_gradient(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Gradient of ||D h||^2, i.e. 2 D^T D h."""
    g = np.zeros_like(h)
    if edges.size:
        d = h[edges[:, 0]] - h[edges[:, 1]]
        np.add.at(g, edges[:, 0], 2 * d)
        np.add.at(g, edges[:, 1], -2 * d)
    return g


def data_term(patch: Patch, state: State) -> float:
    total = 0.0
    for r in range(len(patch.y)):
        e = patch.centred(r) - state.h @ state.c[r]
        total += 0.5 * float(np.sum(e * patch.apply_w(e)))
    return total


def objective(patch: Patch, state: State, weights: list[np.ndarray], data: float | None = None) -> float:
    """The full objective; ``weights`` are the astrocyte high-pass weights of each recording. ``data`` is the
    data term when the caller already has it (solver.fit), else it is computed here."""
    pen = state.pen
    f = data_term(patch, state) if data is None else data
    for i in patch.kind_index(*NEURON_LIKE):
        f += pen.lam_s[i] * sum(float(np.sum(s[i])) for s in state.s)
    for j in patch.kind_index(*ASTRO_LIKE):
        f += pen.lam_a[j] * sum(astro_prior.penalty(c[j], w) for c, w in zip(state.c, weights, strict=True))
        f += pen.lam_t[j] * float(np.sum((1 - patch.tau) * state.h[:, j]))
    for b in patch.kind_index(*SMOOTH):
        f += pen.lam_b[b] * roughness(state.h[:, b], patch.edges)
    return f
