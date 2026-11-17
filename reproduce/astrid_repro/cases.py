"""The injection and leakage tests of D1 (Supplementary, "Injection and leakage tests"): the rules that make the
cases, the patch of a case, ASTRID's fit of it, the export of the cases for the peer methods and the per-case
scores.

D1 has no ground truth, so known signals are added to the real movie of one recording:
- **the injection test** copies the spatial footprint and ΔF/F0 of a real ROI (a source) to free sites inside and
  away from astrocyte territory, at x1 and x3; the copy's ROI average is the truth;
- **the leakage test** places the location's largest astrocyte ROI, with its ΔF/F0, over real ROIs near
  territory, at x0, x1 and x3; the change of an ROI's trace from x0 is the leakage.

**The rules.** Lengths are in pixels of the movies' grid, about 2 um on D1; ROI v is row v-1 of a recording's
ΔF/F0; with several recordings, a ROI's SNR and percentile are means over them.
- ``leakage_targets``: the ROIs of at least ``TARGET_MIN_PX`` pixels inside the registration border with
  territory within ``NEAR_PX`` of them (outside the ROI), the most territory pixels first, ties to the higher
  ``snr_proxy`` and then the higher label; the first ``LEAK_TARGETS``.
- ``injection_sources``: the ROIs of at least ``SOURCE_MIN_PX`` pixels inside the border, with no AQuA2 event
  within ``SOURCE_CLEAR_PX`` and a usable footprint (``usable_source``), the highest 99th percentile of ΔF/F0
  first; the first ``SOURCES``.
- ``free_sites``: up to ``SITES`` sites of each kind, in strong territory (a site pixel touched by at least
  ``SITE_MIN_EVENTS`` events, evidence at least ``SITE_MIN_EVIDENCE``) or away from it (no event within
  ``CONTROL_CLEAR_PX``, evidence below ``CONTROL_MAX_EVIDENCE``). The field's pixels are drawn in a random order
  (``SITE_SEED``) and a pixel is kept at least ``SITE_GAP_PX`` from the sites of its kind kept before, more than
  ``SITE_CLEARANCE_PX`` from every ROI and candidate pixel, and ``SITE_INSET_PX`` inside the field.
- ``injection``: the source's ROI and footprint moved so that the ROI's centroid lies on the site (rounded to
  whole pixels), carrying the source's ΔF/F0 times the scale. ``injection_cases``: the sources serve the sites
  in turn, the territory sites first, and x1 and x3 at a site share a source.
- ``astro_injection``: the astrocyte ROI moved so that its centroid lies ``PLACEMENT_OFFSETS_PX`` from the
  target's centroid in one of 8 directions: at the first of these distances at which some direction covers at
  least ``MIN_COVER`` of the ROI, the direction whose moved astrocyte lies most in territory (the first on
  ties); with no such distance, the placement with the largest cover.

**Footprints.** A source's footprint (``source_footprint``) is its profile in the location's maximum projection:
the projection minus the median over the ring of pixels more than ``PROFILE_RING_PX[0]`` and at most
``PROFILE_RING_PX[1]`` from the ROI that belong to no ROI, set to zero where negative and outside the ROI
dilated by 1 pixel, normalized to sum 1 (float32, as the projection). An astrocyte's footprint is its mask
smoothed by a Gaussian of s.d. 1 pixel and set to zero outside the mask, normalized to sum 1.

**The patch of a case** (``build_case``) is ``astrid.patch``'s patch of the case's ROI: an injected ROI's patch
leaves no ROI out of its neighbours, and a leakage target's leaves out the target's own label. The added
signal is footprint x trace, with the trace scaled by the baseline: an injection's trace is u F0 / mean(h) with
F0 and the footprint h averaged over the injected ROI, an added astrocyte's over its mask within the patch, and
u the copied ΔF/F0 times the scale; so the signal averaged over that mask and divided by its mean baseline is
u. It is added to the data Y only. The baseline F0 and the pixel weights come from the movie before the
addition (``astrid.api.load_pixels``), and the noise covariance Sigma from the data after it. An added
astrocyte is one more astrocyte component, ``A-inj``, placed before the local neuropil, on its mask dilated by
``ASTRO_SUPPORT_PX`` within the patch, and the evidence is raised to at least ``EVIDENCE_FLOOR`` over its mask.
``astro_components=False`` (the paper's ablation A) leaves out every astrocyte component, ``A-inj`` included;
the added signals and the evidence stay as they are.

**The fit** (``fit_case``): ``astrid.solver.fit`` with the settings' parameters, at most 600 iterations in the
injection test and 2,000 in the leakage test (``reproduce/datasets.yaml``). Its outputs, per case (one row per
recording): ``delivered``, the demixed trace of the ROI in fluorescence units; ``ring_trace``, the ring
subtraction (``astrid.patch.ring_trace``); ``baseline_roi``, the ROI average of F0; ``truth``, the injected ROI
average of the added signal, or ``astro_roi``, the added astrocyte's ROI average (both empty where the case has
none); ``names``, ``kinds``, ``pixels`` (column-major field indices), ``roi``, ``ring``, ``tau``, ``h`` and ``c``
(the fitted footprints and traces), ``stop``, ``iterations`` and ``converged``.

**The export** (``export_cases``) writes the case format of ``astrid_repro.peers`` (its docstring): per base (an
injection site or a leakage target) the raw movie cropped to the bounding box of the ROI dilated by
``EXPORT_MARGIN_PX`` (about 30 um) within the registration border, and the masks of the target, of the
neighbouring ROIs and candidate pieces of ASTRID's patch (undilated, cut by the crop) and of its astrocyte
components (in a leakage case with the added astrocyte's support); per case the added footprint and amplitude,
with the truth or the astrocyte's ROI average.

**The scores** (``injection_scores``, ``leakage_scores``), the columns ``stats.py d1`` reads, with the means
removed from every trace:
- injection, per case and method: ``corr`` (Pearson), ``gain`` (the slope of the trace on the truth) and the
  errors ||y - q|| / ||q|| unfiltered, below 2 Hz and below 0.5 Hz (``err_raw``, ``err_2hz``, ``err_05hz``);
- leakage, per target, scale (> 0) and method: ``leak`` and ``leak_05hz`` (||y+ - y0|| / ||q||, unfiltered and
  below 0.5 Hz) and ``kept_corr`` (the correlation of y+ and y0).
Every trace is scored in its own precision, as the paper's scores were: a float32 trace (TUnCaT's raw ROI
average) has its mean removed in float32 before the norms, so its unfiltered error, gain and leakage differ from
``metrics.py``'s float64 ones in the last digits; the low-pass works in float64.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from astrid import kernels
from astrid.api import FORK
from astrid.model import Component, Params, Patch
from astrid.noise import noise_covariance
from astrid.outputs import demixed_trace
from astrid.patch import (
    ASTRO_SUPPORT_PX,
    ATOM_MIN_PX,
    FOUR,
    MATCH,
    MAX_ATOMS,
    NOISE_F_CUT,
    REFERENCE_UM_PER_PX,
    RING_INNER,
    RING_OUTER,
    UNCLASSIFIED_MIN_PX,
    Built,
    Field,
    PixelData,
    dilate,
    patch_components,
    patch_region,
    ring_trace,
    scaled,
    scaled_area,
)
from astrid.solver import fit
from astrid_repro.metrics import lowpass_fourier

TERRITORY_MIN_EVENTS = 2            # territory: pixels touched by at least 2 AQuA2 events (astrid.patch's rule)
TARGET_MIN_PX = 10                  # leakage targets
NEAR_PX = 3                         # territory within 3 px of a leakage target
LEAK_TARGETS = 10
SOURCE_MIN_PX = 20                  # injection sources
SOURCE_CLEAR_PX = 5                 # no AQuA2 event within 5 px of a source
SOURCES = 4
PROFILE_RING_PX = (3, 6)            # the ring whose median is a source's background in the max projection
SITES = 8                           # per kind
SITE_MIN_EVENTS = 4
SITE_MIN_EVIDENCE = 0.5
CONTROL_CLEAR_PX = 6
CONTROL_MAX_EVIDENCE = 0.2
SITE_GAP_PX = 15.0
SITE_CLEARANCE_PX = 4.0
SITE_INSET_PX = 15
SITE_SEED = 0
SCALES = (1.0, 3.0)                 # injection test
LEAK_SCALES = (0.0, 1.0, 3.0)       # leakage test
PLACEMENT_OFFSETS_PX = (5.0, 2.5, 7.5, 0.0, 10.0, 12.5)
MIN_COVER = 0.5
EVIDENCE_FLOOR = 0.8                # the evidence over an added astrocyte's mask
EXPORT_MARGIN_PX = 15               # the peers' crop: the ROI dilated by 15 reference px (30 um)
TESTS = ("injection", "leakage")
METHODS = ("demix", "ring", "fissa", "tuncat", "tuncat_astro", "cnmf", "cnmf_astro", "roi", "suite2p")
PEER_KEYS = {"fissa": "fissa", "tuncat": "tuncat", "tuncat_astro": "tuncat_astro", "cnmf": "cnmf",
             "cnmf_astro": "cnmf_astro", "roi": "tuncat_raw_roi", "suite2p": "suite2p"}
BANDS = {"raw": None, "2hz": 2.0, "05hz": 0.5}


# --------------------------------------------------------------------------------------------- the location
@dataclass
class Location:
    """One imaging location as the tests read it: ``field`` (``astrid.patch.Field``: the ROIs as labels 1..n,
    the candidate pieces, the territory, the evidence, the frame rate, the pixel scale and the border), the
    number of AQuA2 events that touch each pixel over the location's recordings (``events``), the mean of the
    recordings' maximum projections (``max_proj``, float32) and the astrocyte ROIs (``astro_label``, labels
    1..m, 0 for none)."""
    field: Field
    events: np.ndarray
    max_proj: np.ndarray
    astro_label: np.ndarray

    @property
    def neurons(self) -> np.ndarray:
        return self.field.neurons

    @property
    def inner(self) -> np.ndarray:
        return self.field.inner

    @property
    def shape(self) -> tuple[int, int]:
        return self.field.shape


def location(name: str, neurons: np.ndarray, candidates: np.ndarray, events: np.ndarray, tau: np.ndarray,
             max_proj: np.ndarray, astro_label: np.ndarray, fps: float, um_per_px: float,
             border_px: int) -> Location:
    """A ``Location`` from its arrays (all (H, W)): the territory is the pixels touched by at least
    ``TERRITORY_MIN_EVENTS`` events, the candidates are relabelled as 4-connected pieces, and the evidence map
    is the tagRFP evidence ``tau`` in [0, 1]."""
    neurons = np.asarray(neurons)
    shape = (int(neurons.shape[0]), int(neurons.shape[1]))
    arrays = {"candidates": candidates, "events": events, "tau": tau, "max_proj": max_proj,
              "astro_label": astro_label}
    for what, a in arrays.items():
        if np.shape(a) != shape:
            raise ValueError(f"{name}: {what} is {np.shape(a)}, the ROIs {shape}")
    if not np.issubdtype(neurons.dtype, np.integer) or neurons.min() < 0:
        raise ValueError(f"{name}: neurons must be integer labels >= 0")
    tau = np.asarray(tau, dtype=np.float64)
    if not np.all(np.isfinite(tau)) or tau.min() < 0 or tau.max() > 1:
        raise ValueError(f"{name}: the evidence tau must lie in [0, 1]")
    events = np.asarray(events)
    if not np.array_equal(events, np.round(events)) or events.min() < 0:
        raise ValueError(f"{name}: events must be counts")
    labels, _ = ndi.label(np.asarray(candidates) > 0, structure=FOUR)
    fld = Field(name, shape, neurons, labels, events >= TERRITORY_MIN_EVENTS, tau, float(fps),
                px_scale=REFERENCE_UM_PER_PX / float(um_per_px), tagrfp_channel=True, border_px=int(border_px))
    return Location(fld, events, np.asarray(max_proj, dtype=np.float32), np.asarray(astro_label))


def largest_astrocyte(astro_label: np.ndarray) -> int:
    """The label of the largest astrocyte ROI (ties to the lower label), 0 if there is none."""
    a = np.asarray(astro_label)
    return int(np.bincount(a.ravel())[1:].argmax()) + 1 if a.max() > 0 else 0


# --------------------------------------------------------------------------------------------- the rules
def snr_proxy(dff: np.ndarray) -> float:
    """(99th percentile - median) over the noise s.d. from frame differences (a ranking key only)."""
    noise = 1.4826 * np.median(np.abs(np.diff(dff))) / np.sqrt(2)
    return float((np.percentile(dff, 99) - np.median(dff)) / max(noise, 1e-12))


def leakage_targets(loc: Location, dff: Mapping[str, np.ndarray], n: int = LEAK_TARGETS) -> list[int]:
    """The leakage test's ROIs (module docstring); ``dff``: recording -> (n_rois, T) ΔF/F0."""
    score = []
    terr = loc.field.territory
    for v in range(1, int(loc.neurons.max()) + 1):
        roi = loc.neurons == v
        if roi.sum() < TARGET_MIN_PX or (roi & ~loc.inner).any():
            continue
        near = dilate(roi, NEAR_PX) & ~roi
        active = int(terr[near].sum())
        snr = np.mean([snr_proxy(d[v - 1]) for d in dff.values()])
        score.append((active, snr, v))
    return [v for a, s, v in sorted(score, reverse=True) if a > 0][:n]


def _source_profile(loc: Location, label: int) -> tuple[np.ndarray, np.ndarray | None]:
    roi = loc.neurons == label
    d = ndi.distance_transform_edt(~roi)
    ring = (d > PROFILE_RING_PX[0]) & (d <= PROFILE_RING_PX[1]) & (loc.neurons == 0)
    if not ring.any():
        return roi, None
    bg = float(np.median(loc.max_proj[ring]))
    prof = np.where(d <= 1, np.clip(loc.max_proj - bg, 0, None), 0.0)
    return roi, prof if prof[roi].sum() > 0 else None


def usable_source(loc: Location, label: int) -> bool:
    """Whether a ROI's footprint can carry an injection: its profile ring has pixels outside every ROI, and its
    profile rises above that background on the ROI itself (else the truth would be zero)."""
    return _source_profile(loc, label)[1] is not None


def source_footprint(loc: Location, label: int) -> tuple[np.ndarray, np.ndarray]:
    """(ROI, footprint) of a source (module docstring); ValueError if it is not a ``usable_source``."""
    roi, prof = _source_profile(loc, label)
    if prof is None:
        raise ValueError(f"M-{label:03d} has no usable footprint (see usable_source)")
    return roi, prof / prof.sum()


def injection_sources(loc: Location, dff: Mapping[str, np.ndarray], n: int = SOURCES) -> list[int]:
    """The injection test's sources (module docstring); may be empty, and then there are no injections."""
    cand = []
    events = loc.events > 0
    for v in range(1, int(loc.neurons.max()) + 1):
        roi = loc.neurons == v
        if roi.sum() < SOURCE_MIN_PX or (dilate(roi, SOURCE_CLEAR_PX) & events).any() or (roi & ~loc.inner).any():
            continue
        if not usable_source(loc, v):
            continue
        cand.append((np.mean([np.percentile(d[v - 1], 99) for d in dff.values()]), v))
    return [v for s, v in sorted(cand, reverse=True)[:n]]


def free_sites(loc: Location, in_territory: bool, n: int = SITES, min_gap: float = SITE_GAP_PX,
               clearance: float = SITE_CLEARANCE_PX, seed: int = SITE_SEED) -> list[tuple[int, int]]:
    """Up to ``n`` injection sites (row, col) of one kind (module docstring), in the order they were kept."""
    fld = loc.field
    occupied = dilate((fld.neurons > 0) | (fld.candidates > 0), clearance)
    keep = np.zeros(fld.shape, dtype=bool)
    keep[SITE_INSET_PX:-SITE_INSET_PX, SITE_INSET_PX:-SITE_INSET_PX] = True
    if in_territory:
        ok = keep & ~occupied & (loc.events >= SITE_MIN_EVENTS) & (fld.tau >= SITE_MIN_EVIDENCE)
    else:
        ok = keep & ~occupied & ~dilate(loc.events > 0, CONTROL_CLEAR_PX) & (fld.tau < CONTROL_MAX_EVIDENCE)
    pts = np.argwhere(ok)
    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(len(pts))]
    chosen: list[tuple[int, int]] = []
    for p in pts:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap for q in chosen):
            chosen.append((int(p[0]), int(p[1])))
        if len(chosen) == n:
            break
    return chosen


def moved(mask: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """A mask shifted by whole pixels (pixels moved off the field are lost)."""
    return ndi.shift(mask.astype(float), shift, order=0, mode="constant", cval=0) > 0.5


@dataclass
class Injection:
    """An injected neuron: its ROI and footprint (H, W) at the site, its trace per recording (the source's
    ΔF/F0 times the scale), the source's name, the site and the scale."""
    roi: np.ndarray
    footprint: np.ndarray
    dff: dict[str, np.ndarray]
    source: str
    site: tuple[int, int]
    scale: float = 1.0


@dataclass
class AstroInjection:
    """An added astrocyte: its mask and footprint (H, W) over the target, its trace per recording (the astrocyte
    ROI's ΔF/F0 times the scale), its name, the scale and the shift that placed it."""
    shape: np.ndarray
    footprint: np.ndarray
    dff: dict[str, np.ndarray]
    source: str
    scale: float
    shift: tuple[int, int] = (0, 0)


@dataclass
class Case:
    """A case of the tests: an injected neuron (``inject``), or a real ROI (``label``) with an astrocyte added
    over it (``astro``)."""
    name: str
    label: int | None = None
    inject: Injection | None = None
    astro: AstroInjection | None = None

    @property
    def test(self) -> str:
        return "injection" if self.inject is not None else "leakage"

    @property
    def base(self) -> str:
        """The case's base: inj00-astro-x3 -> inj00-astro, M-039-astro-x1 -> M-039."""
        return self.name.rsplit("-x", 1)[0] if self.inject is not None else self.name.rsplit("-astro-x", 1)[0]

    @property
    def scale(self) -> float:
        if self.inject is not None:
            return self.inject.scale
        if self.astro is not None:
            return self.astro.scale
        return 0.0

    def roi(self, loc: Location) -> np.ndarray:
        return self.inject.roi if self.inject is not None else loc.neurons == self.label

    def excluded(self) -> int | None:
        """The label left out of the patch's neighbours: the target's own (none for an injected ROI)."""
        return None if self.inject is not None else self.label


def injection(loc: Location, dff: Mapping[str, np.ndarray], source: int, site: tuple[int, int],
              scale: float = 1.0) -> Injection:
    """Source ``source`` moved to ``site`` (module docstring); ``dff``: recording -> (n_rois, T) ΔF/F0."""
    roi, prof = source_footprint(loc, source)
    r0, c0 = np.argwhere(roi).mean(axis=0)
    shift = (int(round(site[0] - r0)), int(round(site[1] - c0)))
    foot = ndi.shift(prof, shift, order=0, mode="constant", cval=0)
    traces = {r: scale * d[source - 1] for r, d in dff.items()}
    return Injection(moved(roi, shift), foot / foot.sum(), traces, f"M-{source:03d}", site, scale)


def astro_injection(loc: Location, source: int, astro_dff: Mapping[str, np.ndarray], target: int,
                    scale: float, offsets_px: Sequence[float] = PLACEMENT_OFFSETS_PX,
                    min_cover: float = MIN_COVER) -> AstroInjection:
    """Astrocyte ROI ``source`` placed over ROI ``target`` (module docstring); ``astro_dff``: recording ->
    the astrocyte ROI's ΔF/F0 (T,). A ValueError if no placement covers any of the ROI."""
    shape0 = loc.astro_label == source
    a0 = np.argwhere(shape0).mean(axis=0)
    roi = loc.neurons == target
    t0 = np.argwhere(roi).mean(axis=0)
    terr = loc.field.territory
    best: tuple[np.ndarray, tuple[int, int]] | None = None
    best_cover = 0.0
    for off in offsets_px:
        chosen: tuple[np.ndarray, tuple[int, int]] | None = None
        chosen_score = -1.0
        for ang in np.arange(8 if off > 0 else 1) * np.pi / 4:
            c = t0 + off * np.array([np.sin(ang), np.cos(ang)])
            shift = (int(round(c[0] - a0[0])), int(round(c[1] - a0[1])))
            shape = moved(shape0, shift)
            cover = float((shape & roi).sum() / roi.sum())
            if cover > best_cover:
                best, best_cover = (shape, shift), cover
            score = float(terr[shape].mean()) if cover >= min_cover else -1.0
            if score > chosen_score:
                chosen, chosen_score = (shape, shift), score
        if chosen is not None:
            best = chosen
            break
    if best is None:
        raise ValueError(f"astrocyte {source} cannot be placed over ROI {target}")
    shape, shift = best
    foot = ndi.gaussian_filter(shape.astype(float), 1.0) * shape
    return AstroInjection(shape, foot / foot.sum(), {r: scale * d for r, d in astro_dff.items()},
                          f"A-{source:02d}", scale, shift)


def injection_cases(loc: Location, dff: Mapping[str, np.ndarray]) -> tuple[list[Case], dict[str, Any]]:
    """The injection test's cases of one recording (``dff``: recording -> its ΔF/F0), and the choices made:
    the sources and the sites (row, col, kind)."""
    sources = injection_sources(loc, dff, SOURCES)
    sites = [(s, True) for s in free_sites(loc, True, SITES)] + [(s, False) for s in free_sites(loc, False, SITES)]
    if not sources:
        sites = []
    cases = [Case(f"inj{j:02d}-{'astro' if astro else 'ctrl'}-x{scale:g}",
                  inject=injection(loc, dff, sources[j % len(sources)], s, scale))
             for j, (s, astro) in enumerate(sites) for scale in SCALES]
    choices = {"sources": sources, "sites": [[s[0], s[1], "astro" if a else "control"] for s, a in sites]}
    return cases, choices


def leakage_cases(loc: Location, dff: Mapping[str, np.ndarray], astro_dff: Mapping[str, np.ndarray],
                  source: int) -> tuple[list[Case], dict[str, Any]]:
    """The leakage test's cases of one recording: astrocyte ROI ``source`` (``astro_dff``: recording -> its
    ΔF/F0) over each target at x0, x1 and x3; and the choices: the targets and the astrocyte's shifts."""
    targets = leakage_targets(loc, dff, LEAK_TARGETS)
    cases, shifts = [], {}
    for v in targets:
        for s in LEAK_SCALES:
            a = astro_injection(loc, source, astro_dff, v, s)
            cases.append(Case(f"M-{v:03d}-astro-x{s:g}", label=v, astro=a))
        shifts[f"M-{v:03d}"] = list(a.shift)
    return cases, {"targets": targets, "astro_source": f"A-{source:02d}", "astro_shifts": shifts}


# --------------------------------------------------------------------------------------------- the patch
@dataclass
class CasePatch:
    """A case's patch (``astrid.patch.Built``) with the added signals' ROI averages per recording: the truth
    of an injection and the added astrocyte's average over the ROI (None where the case has none)."""
    built: Built
    truth: list[np.ndarray] | None
    astro_roi: list[np.ndarray] | None


def case_flat(loc: Location, case: Case) -> np.ndarray:
    """The column-major indices of the case's patch."""
    return loc.field.flat(patch_region(loc.field, case.roi(loc)))


def build_case(loc: Location, data: PixelData, case: Case, astro_components: bool = True) -> CasePatch:
    """The patch of ``case`` with its signal added (module docstring). ``data`` holds the pixels of the movie
    before any addition (``astrid.api.load_pixels``), one entry per recording."""
    fld = loc.field
    roi = case.roi(loc)
    region = patch_region(fld, roi)
    flat = fld.flat(region)
    rows = data.rows(flat)
    h_img = fld.shape[0]
    coords = np.stack([flat % h_img, flat // h_img], axis=1)
    comps, names = patch_components(fld, region, roi, case.name, case.excluded(),
                                    astro_components=astro_components)
    recs = list(data.y)
    ys = [data.y[r][rows].astype(np.float64) for r in recs]
    ms = [data.m[r][rows].astype(np.float64) for r in recs]
    roi_local = roi.ravel(order="F")[flat]
    tau = fld.tau.ravel(order="F")[flat]
    astro_roi = None
    if case.astro is not None:
        shape_local = case.astro.shape.ravel(order="F")[flat]
        if astro_components:
            support = dilate(case.astro.shape, scaled(fld, ASTRO_SUPPORT_PX)).ravel(order="F")[flat]
            comps.insert(len(comps) - 1, Component("A-inj", "astro", support))
            names.insert(len(names) - 1, "A-inj")
        tau = np.maximum(tau, EVIDENCE_FLOOR * shape_local)
        h_ast = case.astro.footprint.ravel(order="F")[flat]
        astro_roi = []
        for i, r in enumerate(recs):
            a = case.astro.dff[r] * float(ms[i][shape_local].mean()) / float(h_ast[shape_local].mean())
            ys[i] = ys[i] + np.outer(h_ast, a)
            astro_roi.append(float(h_ast[roi_local].mean()) * a)
    truth = None
    if case.inject is not None:
        h_inj = case.inject.footprint.ravel(order="F")[flat]
        truth = []
        for i, r in enumerate(recs):
            f0 = float(ms[i][roi_local].mean())
            a = case.inject.dff[r] * f0 / float(h_inj[roi_local].mean())
            ys[i] = ys[i] + np.outer(h_inj, a)
            truth.append(float(h_inj[roi_local].mean()) * a)
    others = (fld.neurons > 0) | (fld.candidates > 0) | roi
    d = ndi.distance_transform_edt(~roi)
    ring = ((d > scaled(fld, RING_INNER)) & (d <= scaled(fld, RING_OUTER)) & ~others).ravel(order="F")[flat]
    w = 1 / np.maximum(data.noise_var[rows], 1e-12)
    sigma, _ = noise_covariance([y - m for y, m in zip(ys, ms, strict=True)], fld.fps, NOISE_F_CUT)
    patch = Patch(ys, ms, coords, w, tau, comps, tagrfp=fld.tagrfp_channel)
    return CasePatch(Built(patch, names, flat, roi_local, ring, sigma), truth, astro_roi)


# --------------------------------------------------------------------------------------------- the fits
def _stacked(parts: list[np.ndarray] | None) -> np.ndarray:
    return np.array(parts) if parts is not None else np.zeros(0)


def fit_case(loc: Location, data: PixelData, case: Case, params: Params,
             astro_components: bool = True) -> dict[str, np.ndarray]:
    """ASTRID's fit of one case and its outputs (module docstring), as arrays."""
    t0 = time.time()
    cp = build_case(loc, data, case, astro_components)
    b = cp.built
    state = fit(b.patch, params, sigma=b.sigma)
    k = 0
    n_rec = len(b.patch.y)
    roi = b.patch.components[k].roi_mask
    return {"names": np.array(b.names), "kinds": np.array([c.kind for c in b.patch.components]),
            "pixels": b.pixels, "roi": b.roi, "ring": b.ring, "tau": b.patch.tau, "h": state.h,
            "c": np.array(list(state.c)),
            "delivered": np.array([demixed_trace(b.patch, state, k, roi, r)[0] for r in range(n_rec)]),
            "ring_trace": np.array([ring_trace(b, r) for r in range(n_rec)]),
            "baseline_roi": np.array([b.patch.m[r][b.roi].mean(axis=0) for r in range(n_rec)]),
            "truth": _stacked(cp.truth), "astro_roi": _stacked(cp.astro_roi),
            "stop": np.array(state.stop_reason), "iterations": np.array(int(state.iterations)),
            "converged": np.array(bool(state.target_converged(k, params.tol_target))),
            "seconds": np.array(time.time() - t0)}


_SHARED: dict[str, Any] = {}        # the location and pixels that forked workers inherit


def _job(args: tuple[Any, ...]) -> tuple[str, dict[str, np.ndarray]]:
    loc, data, case, params, astro = args
    return case.name, fit_case(loc, data, case, params, astro)


def _shared_job(args: tuple[Any, ...]) -> tuple[str, dict[str, np.ndarray]]:
    case, params, astro = args
    loc, data = _SHARED["loc"], _SHARED["data"]
    return case.name, fit_case(loc, data.subset(case_flat(loc, case)), case, params, astro)


def problem_size(loc: Location, case: Case, astro_components: bool = True) -> int:
    """A case's patch pixels times its components (the order in which ``fit_cases`` hands out the fits)."""
    roi = case.roi(loc)
    region = patch_region(loc.field, roi)
    comps, _ = patch_components(loc.field, region, roi, case.name, case.excluded(),
                                astro_components=astro_components)
    return int(region.sum()) * (len(comps) + (case.astro is not None and astro_components))


def fit_cases(loc: Location, data: PixelData, cases: Sequence[Case], params: Params, out: Path, *,
              astro_components: bool = True, workers: int = 0,
              log: Callable[[str], Any] = print) -> list[dict[str, Any]]:
    """Fit every case and write its outputs to ``out/<case>.npz``; returns one row per case, in the cases'
    order (``case``, ``stop``, ``iterations``, ``converged``, ``pixels``, ``components``, ``seconds``).
    ``workers`` > 1 fits that many cases at a time in processes, the largest patches first; each fit is
    independent of the others, so the order changes no result."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    t0 = time.time()

    def save(name: str, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(out / f"{name}.npz", **arrays)
        rows[name] = {"case": name, "stop": str(arrays["stop"]), "iterations": int(arrays["iterations"]),
                      "converged": bool(arrays["converged"]), "pixels": int(arrays["pixels"].size),
                      "components": int(arrays["names"].size), "seconds": float(arrays["seconds"])}
        log(f"  {name}: {rows[name]['stop']} after {rows[name]['iterations']} iterations "
            f"({len(rows)} of {len(cases)}, {time.time() - t0:.0f} s)")

    kernels.warm_up()
    if workers > 1 and len(cases) > 1:
        order = sorted(cases, key=lambda c: -problem_size(loc, c, astro_components))
        if FORK:
            _SHARED.update(loc=loc, data=data)
            jobs: list[tuple[Any, ...]] = [(c, params, astro_components) for c in order]
            fn: Callable[[tuple[Any, ...]], tuple[str, dict[str, np.ndarray]]] = _shared_job
            ctx = multiprocessing.get_context("fork")
        else:
            jobs = [(loc, data.subset(case_flat(loc, c)), c, params, astro_components) for c in order]
            fn, ctx = _job, None
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(cases)), mp_context=ctx) as pool:
                for fut in as_completed([pool.submit(fn, j) for j in jobs]):
                    save(*fut.result())
        finally:
            _SHARED.clear()
    else:
        for c in cases:
            save(c.name, fit_case(loc, data.subset(case_flat(loc, c)), c, params, astro_components))
    return [rows[c.name] for c in cases]


# --------------------------------------------------------------------------------------------- the export
def neighbours_and_atoms(loc: Location, roi: np.ndarray, exclude: int | None
                         ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The neighbouring cells (ROIs and candidate pieces, whole) and the astrocyte components (their supports)
    of ASTRID's patch around ``roi``, as (H, W) masks, in the order of ``astrid.patch.patch_components``."""
    fld = loc.field
    region = patch_region(fld, roi)
    cells = []
    for v in np.unique(fld.neurons[region]):
        if v and v != exclude and ((fld.neurons == v) & region).sum() >= MATCH * (fld.neurons == v).sum():
            cells.append(fld.neurons == v)
    accepted = fld.neurons > 0
    for v in np.unique(fld.candidates[region]):
        piece = fld.candidates == v
        if v == 0 or (piece & (accepted | roi)).sum() >= MATCH * piece.sum():
            continue
        if (piece & region).sum() >= MATCH * piece.sum() and piece.sum() >= UNCLASSIFIED_MIN_PX:
            cells.append(piece)
    territory, n = ndi.label(fld.territory & region, structure=FOUR)
    sizes = ndi.sum(np.ones_like(territory), territory, index=np.arange(1, n + 1)) if n else np.array([])
    atoms = [dilate(territory == i + 1, scaled(fld, ASTRO_SUPPORT_PX)) & region
             for i in np.argsort(sizes)[::-1][:MAX_ATOMS] if sizes[i] >= scaled_area(fld, ATOM_MIN_PX)]
    return cells, atoms


def crop_box(loc: Location, roi: np.ndarray) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1): the bounding box of ``roi`` dilated by ``EXPORT_MARGIN_PX`` reference pixels, within
    the registration border."""
    box = np.argwhere(dilate(roi, scaled(loc.field, EXPORT_MARGIN_PX)) & loc.inner)
    (r0, c0), (r1, c1) = box.min(axis=0), box.max(axis=0) + 1
    return r0, r1, c0, c1


def export_boxes(loc: Location, cases: Sequence[Case]) -> dict[str, tuple[int, int, int, int]]:
    """Each base's crop (``crop_box``), in the order of the cases' first appearance."""
    out: dict[str, tuple[int, int, int, int]] = {}
    for c in cases:
        if c.base not in out:
            out[c.base] = crop_box(loc, c.roi(loc))
    return out


def read_crops(movie: Any, boxes: Mapping[str, tuple[int, int, int, int]],
               block: int = 1000) -> dict[str, np.ndarray]:
    """The (T, h, w) float32 crops of a (T, H, W) movie (an array or anything that slices along time, such as an
    ``h5py.Dataset``), read in one pass over blocks of frames."""
    n = int(movie.shape[0])
    out = {k: np.empty((n, r1 - r0, c1 - c0), dtype=np.float32) for k, (r0, r1, c0, c1) in boxes.items()}
    for s in range(0, n, block):
        b = np.asarray(movie[s:s + block])
        for k, (r0, r1, c0, c1) in boxes.items():
            out[k][s:s + b.shape[0]] = b[:, r0:r1, c0:c1]
    return out


def export_cases(loc: Location, crops: Mapping[str, np.ndarray], cases: Sequence[Case],
                 fits: Mapping[str, Path], out: Path) -> list[dict[str, Any]]:
    """Write every case in the case format of ``astrid_repro.peers`` into ``out`` and return the index rows
    (``index.json``), in the cases' order. ``crops``: each base's movie crop (``read_crops``); ``fits``: per
    test, the folder of ASTRID's outputs, which hold each case's truth or astrocyte average (first
    recording)."""
    out.mkdir(parents=True, exist_ok=True)
    done: dict[str, tuple[tuple[slice, slice], int, int]] = {}
    index: list[dict[str, Any]] = []
    for c in cases:
        roi = c.roi(loc)
        if c.base not in done:
            r0, r1, c0, c1 = crop_box(loc, roi)
            extra = []
            if c.astro is not None:
                extra = [dilate(c.astro.shape, scaled(loc.field, ASTRO_SUPPORT_PX)) & patch_region(loc.field, roi)]
            crop = crops[c.base]
            if crop.shape[1:] != (r1 - r0, c1 - c0):
                raise ValueError(f"{c.base}: the crop is {crop.shape[1:]}, its box {(r1 - r0, c1 - c0)}")
            np.save(out / f"{c.base}_movie.npy", crop)
            cells, atoms = neighbours_and_atoms(loc, roi, c.excluded())
            sl = (slice(r0, r1), slice(c0, c1))
            np.savez_compressed(out / f"{c.base}_masks.npz", target=roi[sl],
                                neighbours=np.array([m[sl] for m in cells]).reshape(-1, r1 - r0, c1 - c0),
                                atoms=np.array([m[sl] for m in atoms + extra]).reshape(-1, r1 - r0, c1 - c0),
                                box=np.array([r0, r1, c0, c1]))
            done[c.base] = (sl, len(cells), len(atoms) + len(extra))
        sl, nc, na = done[c.base]
        with np.load(fits[c.test] / f"{c.name}.npz") as z:
            if c.inject is not None:
                truth = z["truth"][0]
                h = c.inject.footprint
                amp = truth / float(h[roi].mean())
                np.savez_compressed(out / f"{c.name}.npz", footprint=h[sl].astype(np.float32), amplitude=amp,
                                    truth=truth)
            else:
                assert c.astro is not None
                a = z["astro_roi"][0]
                h = c.astro.footprint
                amp = a / float(h[roi].mean()) if c.astro.scale > 0 else np.zeros(crops[c.base].shape[0])
                np.savez_compressed(out / f"{c.name}.npz", footprint=h[sl].astype(np.float32), amplitude=amp,
                                    astro_roi=a)
        index.append({"spec": c.name, "base": c.base, "test": c.test, "scale": c.scale,
                      "site": ("astro" if "astro" in c.base else "control") if c.inject is not None else "",
                      "neighbours": nc, "atoms": na})
    return index


def write_index(out: Path, index: list[dict[str, Any]]) -> None:
    (out / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------------------------- the scores
def ac(x: np.ndarray) -> np.ndarray:
    """``x`` minus its mean, in ``x``'s own precision."""
    return x - x.mean()


def band(x: np.ndarray, f: float | None, fps: float) -> np.ndarray:
    """``x`` low-passed below ``f`` Hz (None: unfiltered), then its mean removed."""
    return ac(x) if f is None else ac(lowpass_fourier(x, fps, f))


def case_traces(fits: Path, results: Sequence[Path], case: str) -> dict[str, np.ndarray]:
    """Every method's trace of one case: ASTRID's (``demix``) and the ring subtraction's (``ring``) from ASTRID's
    outputs (first recording), the peers' from their results files in ``results`` (``astrid_repro.peers``,
    ``suite2p_runner``; a later folder's array replaces an earlier one's), where present."""
    r: dict[str, np.ndarray] = {}
    for d in results:
        p = d / f"{case}.npz"
        if p.exists():
            with np.load(p) as z:
                r.update({k: z[k] for k in z.files})
    with np.load(fits / f"{case}.npz") as z:
        out = {"demix": z["delivered"][0], "ring": z["ring_trace"][0]}
    out.update({m: r[k] for m, k in PEER_KEYS.items() if k in r})
    return {m: out[m] for m in METHODS if m in out}


def injection_scores(cases: Path, fits: Path, index: Sequence[Mapping[str, Any]], fps: float,
                     results: Sequence[Path] = ()) -> pd.DataFrame:
    """One row per injected case and method (module docstring); empty without injected cases. ``cases``: the
    export (the truths), ``fits``: ASTRID's outputs of the injection test."""
    rows = []
    for c in [c for c in index if c["test"] == "injection"]:
        tr = case_traces(fits, results, c["spec"])
        truth = np.load(cases / f"{c['spec']}.npz")["truth"]
        for m, x in tr.items():
            row = {"case": c["spec"], "site": c["site"], "scale": c["scale"], "method": m,
                   "corr": float(np.corrcoef(x, truth)[0, 1]),
                   "gain": float(ac(x) @ ac(truth) / (ac(truth) @ ac(truth)))}
            for b, f in BANDS.items():
                row[f"err_{b}"] = float(np.linalg.norm(band(x, f, fps) - band(truth, f, fps))
                                        / np.linalg.norm(band(truth, f, fps)))
            rows.append(row)
    return pd.DataFrame(rows)


def leakage_scores(cases: Path, fits: Path, index: Sequence[Mapping[str, Any]], fps: float,
                   results: Sequence[Path] = ()) -> pd.DataFrame:
    """One row per leakage target, scale (> 0) and method present at that scale and at x0 (module
    docstring). ``cases``: the export (the astrocyte averages), ``fits``: ASTRID's outputs of the leakage
    test."""
    rows = []
    for c in [c for c in index if c["test"] == "leakage" and c["scale"] > 0]:
        tr = case_traces(fits, results, c["spec"])
        tr0 = case_traces(fits, results, f"{c['base']}-astro-x0")
        a = np.load(cases / f"{c['spec']}.npz")["astro_roi"]
        for m in [m for m in METHODS if m in tr and m in tr0]:
            d = tr[m] - tr0[m]
            leak_05hz = np.linalg.norm(band(d, 0.5, fps)) / np.linalg.norm(band(a, 0.5, fps))
            rows.append({"target": c["base"], "scale": c["scale"], "method": m,
                         "leak": float(np.linalg.norm(ac(d)) / np.linalg.norm(ac(a))),
                         "leak_05hz": float(leak_05hz),
                         "kept_corr": float(np.corrcoef(tr[m], tr0[m])[0, 1])})
    return pd.DataFrame(rows)


def summary(inj: pd.DataFrame, leak: pd.DataFrame) -> str:
    """The medians over the cases per method: the error below 0.5 Hz and the leakage, at each scale."""
    lines = []
    for name, t, value in (("error below 0.5 Hz", inj, "err_05hz"), ("leakage", leak, "leak")):
        if t.empty:
            continue
        med = t.pivot_table(index="scale", columns="method", values=value, aggfunc="median")
        med = med[[m for m in METHODS if m in med.columns]]
        n = t[t.method == "demix"].shape[0]
        lines += [f"{name}, median over the cases ({n} cases of ASTRID):", med.round(3).to_string(), ""]
    return "\n".join(lines) if lines else "no scored case"
