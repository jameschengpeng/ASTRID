"""Patches: the pixels and components ASTRID fits around each ROI (Methods, "Spatial supports and
constraints", and Supplementary, "Spatial supports"), and the pixel statistics computed before fitting.

**A patch** is the ROI dilated by ``MARGIN_PX`` (16 um), without the field's registration border
(``border_px``; ROIs that touch it are not fitted).

**Components** (the paper's table of components):
- **the primary neuronal source** (component 0 here, the paper's component 1): the ROI dilated by
  ``NEURON_SUPPORT_PX``; its ROI gives the demixed trace;
- **neighbours:** every other ROI with at least half of its mask in the patch, dilated the same way, as
  "neuron"; and, when a candidate map is given, each 4-connected piece of it with at least
  ``UNCLASSIFIED_MIN_PX`` pixels, at least half of them in the patch and less than half in ROIs, as
  "unclassified";
- **astrocyte components:** the ``MAX_ATOMS`` largest 4-connected regions (>= ``ATOM_MIN_PX``, 80 um^2) of
  the astrocyte territory within the patch, each dilated by ``ASTRO_SUPPORT_PX`` and clipped to the patch;
- **the local neuropil:** the whole patch.

**Lengths are in micrometres.** The constants are in pixels at the reference resolution of 2 um per pixel
(``REFERENCE_UM_PER_PX``); a field at another pixel size multiplies every length by ``Field.px_scale`` = 2 um /
its um per pixel, and every area by its square.

**Before fitting** (``pixel_statistics``): per pixel, the fixed baseline F0 (by default the 10th percentile of
its 1-s means in a moving 60-s window, interpolated to the frames) and the noise variance above ``NOISE_F_CUT``
(5 Hz), whose inverse is the pixel's weight. Pixels are indexed column-major (0-based) on the (H, W) frame
everywhere here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from astrid.model import Component, Patch
from astrid.noise import covariance_of_high_passed, high_passed, noise_covariance

MARGIN_PX = 8                   # 16 um: the margin of the patch around the ROI
BORDER_PX = 5                   # the registration border of the authors' recordings (the analysis default)
NEURON_SUPPORT_PX = 2           # 4 um: the margin of a neuronal spatial support beyond its mask
ASTRO_SUPPORT_PX = 1            # 2 um: the margin of an astrocyte spatial support beyond its region
MATCH = 0.5                     # the share of a neighbouring ROI (or candidate piece) that must lie in the patch
TERRITORY_MIN_EVENTS = 2        # territory = pixels touched by at least this many AQuA2 events
ATOM_MIN_PX = 20                # 80 um^2: the minimum area of a region of astrocyte territory
MAX_ATOMS = 3                   # regions of astrocyte territory kept in a patch
UNCLASSIFIED_MIN_PX = 5         # the minimum size of a candidate piece (pixels, not scaled)
BASELINE_S, BASELINE_Q = 60.0, 10.0
NOISE_F_CUT = 5.0               # Hz
RING_INNER, RING_OUTER = 2.0, 8.0   # the ring subtraction's ring, 4 to 16 um outside the ROI
RING_FACTOR = 0.7               # ring trace = ROI mean - 0.7 x ring mean
REFERENCE_UM_PER_PX = 2.0
FOUR = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]          # 4-connectivity


@dataclass
class Field:
    """One field of view (H, W): the ROIs as labels 1..n, the candidate pieces as labels (0: none), the
    astrocyte territory, the structural evidence in [0, 1], the frame rate, and the pixel scale."""
    name: str
    shape: tuple[int, int]
    neurons: np.ndarray                  # (H, W) int, ROI labels 1..n
    candidates: np.ndarray               # (H, W) int, labels of the candidate pieces
    territory: np.ndarray                # (H, W) bool, the astrocyte territory
    tau: np.ndarray                      # (H, W) structural evidence in [0, 1] (all zero without a map)
    fps: float
    px_scale: float = 1.0                # pixels per reference pixel: REFERENCE_UM_PER_PX / um per pixel
    tagrfp_channel: bool = True          # False: no evidence map, so no evidence penalty
    border_px: int = BORDER_PX
    inner: np.ndarray = field(init=False)   # the field without the registration border

    def __post_init__(self) -> None:
        b = int(self.border_px)
        self.inner = np.zeros(self.shape, dtype=bool)
        self.inner[b:self.shape[0] - b, b:self.shape[1] - b] = True

    def flat(self, mask: np.ndarray) -> np.ndarray:
        """Column-major indices of a mask's pixels (the order used everywhere here)."""
        return np.flatnonzero(mask.ravel(order="F"))


def dilate(mask: np.ndarray, px: float) -> np.ndarray:
    return ndi.distance_transform_edt(~mask) <= px if mask.any() else mask.copy()


@dataclass(frozen=True)
class BaselineRule:
    """How each pixel's baseline F0 is computed before fitting (``baseline``). The default is the paper's:
    the 10th percentile of its 1-s means in a moving 60-s window. ``window_s`` None: one percentile over the whole
    recording, along a straight line with a slope of ``trend_pct_per_s`` percent per second, negative for a
    recording that dims (the rule of the paper's short one-photon recordings, D7, with 3-s bins)."""
    bin_s: float = 1.0
    window_s: float | None = BASELINE_S
    percentile: float = BASELINE_Q
    trend_pct_per_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.bin_s > 0 or not 0 <= self.percentile <= 100:
            raise ValueError(f"baseline: need bin_s > 0 and a percentile in [0, 100], got {self}")
        if self.window_s is not None and (not self.window_s > 0 or self.trend_pct_per_s != 0):
            raise ValueError("baseline: a moving window needs window_s > 0 and no trend (give a trend with "
                             "window_s None)")


DEFAULT_BASELINE = BaselineRule()


def baseline(x: np.ndarray, fps: float, rule: BaselineRule | None = None) -> np.ndarray:
    """Per row of ``x`` (pixels x frames), its baseline under ``rule`` (default ``DEFAULT_BASELINE``):
    - a moving window (default): the moving ``percentile`` of the means of bins of round(bin_s fps) frames
      over a window of round(window_s / bin_s) | 1 bins, interpolated linearly between bin centres to the
      frames (nearest at the ends);
    - the whole recording (``window_s`` None): with the trend 1 + rate (t - the middle), the ``percentile`` of
      the bin means of x / trend, times the trend."""
    rule = DEFAULT_BASELINE if rule is None else rule
    n = x.shape[1]
    if rule.window_s is None:
        y = np.asarray(x, dtype=np.float64)
        trend = 1 + rule.trend_pct_per_s / 100 * (np.arange(n) / fps - (n - 1) / 2 / fps)
        m = int(round(rule.bin_s * fps))
        nb = n // m
        bins = (y / trend)[:, :nb * m].reshape(y.shape[0], nb, m).mean(axis=2)
        return np.percentile(bins, rule.percentile, axis=1)[:, None] * trend[None, :]
    step = max(1, int(round(rule.bin_s * fps)))
    size = int(round(rule.window_s / rule.bin_s)) | 1
    nb = n // step
    means = x[:, :nb * step].reshape(x.shape[0], nb, step).mean(axis=2)
    low = ndi.percentile_filter(means, rule.percentile, size=(1, size), mode="nearest")
    centres = (np.arange(nb) + 0.5) * step
    t = np.arange(n)
    return np.stack([np.interp(t, centres, row) for row in low])


def pixel_statistics(args: tuple[np.ndarray, float, BaselineRule]) -> tuple[np.ndarray, np.ndarray]:
    """(baseline as float32, noise variance) of the (n, T) traces ``y`` of ``args = (y, fps, rule)``, each
    row from its own trace alone: the variance above ``NOISE_F_CUT`` scaled up by the share of a white
    spectrum the high-pass removes (Supplementary, "Weights of the objective")."""
    y, fps, rule = args
    m = baseline(y, fps, rule).astype(np.float32)
    hp = high_passed(y - m, fps, NOISE_F_CUT)
    return m, np.var(hp, axis=1) / (1 - NOISE_F_CUT / (fps / 2))


@dataclass
class PixelData:
    """Traces, baselines and noise variances of a set of field pixels, per recording. ``hp``, if kept
    (``api.load_pixels``): recording -> (N, T) float64, each pixel's data minus baseline high-passed as
    ``build``'s noise covariance computes it, at the frame rate ``hp_fs``; None: ``build`` high-passes the
    patch's pixels itself."""
    flat: np.ndarray                     # (N,) column-major indices, sorted
    y: dict[str, np.ndarray]             # recording -> (N, T) float32
    m: dict[str, np.ndarray]             # recording -> (N, T) float32 baseline
    noise_var: np.ndarray                # (N,) mean over recordings of the variance above NOISE_F_CUT
    hp: dict[str, np.ndarray] | None = None
    hp_fs: float | None = None

    def rows(self, flat: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.flat, flat)

    def subset(self, flat: np.ndarray) -> PixelData:
        """The rows of these pixels (a patch's), as their own PixelData."""
        rows = self.rows(np.unique(flat))
        return PixelData(self.flat[rows], {r: self.y[r][rows] for r in self.y},
                         {r: self.m[r][rows] for r in self.m}, self.noise_var[rows],
                         hp=None if self.hp is None else {r: self.hp[r][rows] for r in self.hp},
                         hp_fs=self.hp_fs)


@dataclass
class Spec:
    """A patch to build: the ROI with label ``label``."""
    name: str
    label: int


@dataclass
class Built:
    patch: Patch
    names: list[str]
    pixels: np.ndarray                   # (P,) column-major field indices
    roi: np.ndarray                      # (P,) bool
    ring: np.ndarray                     # (P,) bool: the ring, minus ROIs and candidates
    sigma: np.ndarray                    # (P, P) noise covariance


def scaled(fld: Field, px: float) -> float:
    """A length in reference pixels, in this field's pixels."""
    return px * float(fld.px_scale)


def scaled_area(fld: Field, px: float) -> float:
    """An area in reference pixels, in this field's pixels: 20 reference pixels are 80 um^2."""
    return px * float(fld.px_scale) ** 2


def patch_region(fld: Field, roi: np.ndarray) -> np.ndarray:
    return dilate(roi, scaled(fld, MARGIN_PX)) & fld.inner


def patch_flat(fld: Field, spec: Spec) -> np.ndarray:
    return fld.flat(patch_region(fld, fld.neurons == spec.label))


def astrocyte_regions(fld: Field, region: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """A patch's astrocyte components (Supplementary, "Spatial supports"): the territory within the patch split into
    4-connected regions, those of at least ``ATOM_MIN_PX`` reference pixels (80 um^2) kept, the largest
    ``MAX_ATOMS`` first (ties toward the higher label), each dilated by ``ASTRO_SUPPORT_PX`` within the patch.
    (name, (H, W) support) pairs, named A-<label>."""
    terr = fld.territory & region
    labels, n = ndi.label(terr, structure=FOUR)
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) if n else np.array([])
    ranked = [int(i) for i in np.argsort(sizes)[::-1]]
    min_area = scaled_area(fld, ATOM_MIN_PX)
    kept = [i for i in ranked if sizes[i] >= min_area][:MAX_ATOMS]
    out = []
    for i in kept:
        support = dilate(labels == i + 1, scaled(fld, ASTRO_SUPPORT_PX)) & region
        if support.any():
            out.append((f"A-{i + 1}", support))
    return out


def patch_components(fld: Field, region: np.ndarray, roi: np.ndarray, target: str,
                     exclude_label: int | None, *, astro_components: bool = True
                     ) -> tuple[list[Component], list[str]]:
    """The components on the patch (``region``), in order: the primary neuronal source, the neighbouring
    ROIs, the unclassified candidate pieces, the astrocyte components, the local neuropil (module
    docstring). ``astro_components=False`` leaves out the astrocyte components (the paper's ablation
    without them)."""
    local = fld.flat(region)

    def on_patch(mask: np.ndarray) -> np.ndarray:
        return mask.ravel(order="F")[local]

    sup = scaled(fld, NEURON_SUPPORT_PX)
    comps = [Component(target, "neuron", on_patch(dilate(roi, sup) & region), target=True,
                       roi=on_patch(roi & region))]
    names = [target]
    accepted = fld.neurons > 0
    for v in np.unique(fld.neurons[region]):
        if v == 0 or v == exclude_label:
            continue
        cell = fld.neurons == v
        if (cell & region).sum() >= MATCH * cell.sum():
            comps.append(Component(f"M-{v:03d}", "neuron", on_patch(dilate(cell, sup) & region)))
            names.append(f"M-{v:03d}")
    for v in np.unique(fld.candidates[region]):
        if v == 0:
            continue
        piece = fld.candidates == v
        if (piece & (accepted | roi)).sum() >= MATCH * piece.sum():
            continue                      # the same cell as an ROI or the target
        if (piece & region).sum() >= MATCH * piece.sum() and piece.sum() >= UNCLASSIFIED_MIN_PX:
            comps.append(Component(f"U-{v}", "unclassified", on_patch(dilate(piece, sup) & region)))
            names.append(f"U-{v}")
    if astro_components:
        for name, support in astrocyte_regions(fld, region):
            comps.append(Component(name, "astro", on_patch(support)))
            names.append(name)
    comps.append(Component("neuropil", "neuropil", np.ones(local.size, dtype=bool)))
    names.append("neuropil")
    return comps, names


def build(fld: Field, data: PixelData, spec: Spec, *, astro_components: bool = True) -> Built:
    """The patch of ``spec`` (module docstring): its pixels' traces and baselines per recording, its
    components, the pixel weights (1 / noise variance, floored at 1e-12), the ring and the noise covariance."""
    roi = fld.neurons == spec.label
    region = patch_region(fld, roi)
    flat = fld.flat(region)
    rows = data.rows(flat)
    h_img = fld.shape[0]
    coords = np.stack([flat % h_img, flat // h_img], axis=1)
    comps, names = patch_components(fld, region, roi, spec.name, spec.label,
                                    astro_components=astro_components)
    ys = [data.y[r][rows].astype(np.float64) for r in data.y]
    ms = [data.m[r][rows].astype(np.float64) for r in data.y]
    roi_local = roi.ravel(order="F")[flat]
    tau = fld.tau.ravel(order="F")[flat]
    others = (fld.neurons > 0) | (fld.candidates > 0) | roi
    d = ndi.distance_transform_edt(~roi)
    ring = ((d > scaled(fld, RING_INNER)) & (d <= scaled(fld, RING_OUTER)) & ~others).ravel(order="F")[flat]
    w = 1 / np.maximum(data.noise_var[rows], 1e-12)
    if data.hp is not None and data.hp_fs == fld.fps and all(r in data.hp for r in data.y):
        # the field's pixels high-passed once (api.load_pixels): a row's high-pass is its own trace's, so the
        # covariance is bitwise the one below
        sigma, _ = covariance_of_high_passed([data.hp[r][rows] for r in data.y], fld.fps, NOISE_F_CUT)
    else:
        sigma, _ = noise_covariance([y - m for y, m in zip(ys, ms, strict=True)], fld.fps, NOISE_F_CUT)
    patch = Patch(ys, ms, coords, w, tau, comps, tagrfp=fld.tagrfp_channel)
    return Built(patch, names, flat, roi_local, ring, sigma)


def ring_trace(built: Built, r: int) -> np.ndarray:
    """The ring subtraction in fluorescence units: ROI mean - 0.7 x ring mean, of the baseline-subtracted
    data (Chen et al. 2013)."""
    x = built.patch.centred(r)
    return x[built.roi].mean(axis=0) - RING_FACTOR * x[built.ring].mean(axis=0)
