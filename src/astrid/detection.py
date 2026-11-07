"""ROI detection as in the paper (Supplementary, "ROI detection"). The paper does not evaluate it, and ASTRID
accepts ROI masks from any source.

**One recording** (``detect_recording``):
1. Candidate ROIs are proposed by SynQuant (Wang et al. 2020; ``astrid.synquant``, whose order-statistics test
   is adapted from the SynQuant and AQuA2 (Mi et al. 2025) MATLAB code of Guoqiang Yu's lab at Virginia Tech
   (now at Tsinghua University), with modifications by James Cheng Peng, used under the MIT license with the
   permission of Guoqiang Yu's lab). SynQuant gives each candidate component a z-score (``zmap``). It runs on
   the maximum and the mean projections of the movie, where components with a score of at least ``Z_CUT`` (10)
   are kept.
   It also runs on a map of fast activity
   (``fast_activity``): each pixel holds the variance of its fluctuations between 0.2 and 5 Hz in excess of the
   noise, relative to the noise. That map is first rescaled linearly to the intensity range of the maximum
   projection, and its highest-scoring components are kept, about as many pixels as the two projections
   proposed together.
2. Pixels within ``BORDER_PX`` (5) of the image edge, a margin left by registration, are removed from the
   proposals. So are the pixels of the recording's astrocyte mask (``astrocyte_mask``): the pixels touched by
   at least 4 AQuA2 events, in 4-connected regions of at least 100 pixels.
3. A 4-connected candidate is accepted if its average trace (``acceptance``), smoothed with a Gaussian of s.d.
   6 frames, meets three conditions. It rises above a moving 6-s baseline by at least three times its noise
   level. It reaches a dF/F0 of at least 50%. Its longest period above a dF/F0 of 10% lasts at most 5 s.

**One imaging location** (``detect_location``): the recordings' ROIs are merged into one set (``merge_rois``).
ROIs of different recordings that overlap by at least half of the smaller one are merged, with at most one ROI
from each recording. Each merged ROI keeps the largest 4-connected piece of the pixels that lie in at least half
of its ROIs. The astrocyte masks of all the recordings are then removed from the merged ROIs, and merged ROIs
with fewer than ``MIN_AREA_PX`` (5) pixels are dropped. The map of all candidates goes to ``astrid.demix`` as
``candidates``.

Pixel lists are 0-based column-major indices of the (H, W) frame, sorted, as MATLAB's ``bwconncomp`` lists
them. Movies are (T, H, W) array-likes. ``zmap`` may be replaced by any function from an image to a score map.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy.ndimage import correlate1d
from scipy.special import ndtri

BORDER_PX = 5
Z_CUT = 10.0
Z_FLOOR = -10.0                     # SynQuant's score where nothing was found, and the trimmed edge's
SCALE_PERCENTILES = (1.0, 99.9)     # the activity map is matched to the max projection at these percentiles
F_LO, F_HI = 0.2, 5.0               # Hz: the band of fast activity; above F_HI only noise is assumed
ASTRO_MIN_EVENTS, ASTRO_MIN_PX = 4, 100
SMOOTH_FRAMES = 6
BASELINE_WINDOW_S = 6.0
BASELINE_QUANTILE, BIAS_QUANTILE = 0.1, 0.15
MIN_RISE_NOISE, MIN_DFF, EVENT_DFF, MAX_EVENT_S = 3.0, 0.5, 0.1, 5.0
MIN_OVERLAP = 0.5
MIN_AREA_PX = 5
PIXEL_BLOCK = 2048
BLOCK = 1000
FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

ZMap = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------------------------- components
def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as pixel lists, numbered in column-major scan order (MATLAB's bwconncomp)."""
    lab_t, n = ndi.label(np.asarray(mask, dtype=bool).T, structure=FOUR_CONN)
    flat = lab_t.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n + 1)
    bounds = np.cumsum(counts)
    return [order[bounds[k - 1]:bounds[k]] for k in range(1, n + 1)]


def from_linear(shape: tuple[int, int], pixels: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.unravel_index(pixels, shape, order="F"))


def mask_from_pixels(shape: tuple[int, int], pixels: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[from_linear(shape, pixels)] = True
    return m


def masks_from_lists(shape: tuple[int, int], lists: Sequence[np.ndarray]) -> np.ndarray:
    """(n, H, W) boolean masks from pixel lists (the input ``astrid.demix`` takes)."""
    out = np.zeros((len(lists), *shape), dtype=bool)
    for k, pix in enumerate(lists):
        out[k][from_linear(shape, pix)] = True
    return out


def inner_mask(shape: tuple[int, int], border: int = BORDER_PX) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[border:shape[0] - border, border:shape[1] - border] = True
    return m


def border_trimming(z_map: np.ndarray, border: int = BORDER_PX) -> np.ndarray:
    """The interior of ``z_map``, with a border of ``border`` pixels set to ``Z_FLOOR``."""
    out = np.full(z_map.shape, Z_FLOOR)
    b = border
    out[b:z_map.shape[0] - b, b:z_map.shape[1] - b] = z_map[b:z_map.shape[0] - b, b:z_map.shape[1] - b]
    return out


# --------------------------------------------------------------------------------------------- the movie
def projections(movie: Any) -> tuple[np.ndarray, np.ndarray]:
    """(maximum, mean) projections of a (T, H, W) movie as float32; the mean is summed in float64."""
    n = movie.shape[0]
    mx = np.full(movie.shape[1:], -np.inf, dtype=np.float32)
    acc = np.zeros(movie.shape[1:], dtype=np.float64)
    for s in range(0, n, BLOCK):
        b = np.asarray(movie[s:s + BLOCK])
        mx = np.maximum(mx, b.max(axis=0))
        acc += b.sum(axis=0, dtype=np.float64)
    return mx, (acc / n).astype(np.float32)


def fast_snr(x: np.ndarray, fps: float) -> np.ndarray:
    """Per row of ``x`` (n, T): the 0.2-5 Hz variance above the white noise expected in that band, over that
    noise variance (read above 5 Hz); 0 where the band holds no more than noise."""
    t = x.shape[1]
    freq = np.fft.rfftfreq(t, 1 / fps)
    band, noise = (freq >= F_LO) & (freq <= F_HI), freq > F_HI
    p = np.abs(sfft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    band_var = 2 * p[:, band].sum(axis=1) / t ** 2
    noise_var = 2 * p[:, noise].mean(axis=1) * band.sum() / t ** 2
    return np.clip(band_var - noise_var, 0, None) / np.maximum(noise_var, 1e-12)


def fast_activity(movie: Any, fps: float) -> np.ndarray:
    """(H, W) map of fast activity (``fast_snr`` of every pixel). The movie is read into memory once."""
    n, h, w = movie.shape
    mov = np.empty((n, h * w), dtype=np.float32)
    for s in range(0, n, BLOCK):
        b = np.asarray(movie[s:s + BLOCK])
        mov[s:s + b.shape[0]] = b.reshape(b.shape[0], -1, order="F")
    out = np.empty(h * w)
    for a in range(0, h * w, PIXEL_BLOCK):
        out[a:a + PIXEL_BLOCK] = fast_snr(mov[:, a:a + PIXEL_BLOCK].T.astype(np.float64), fps)
    return out.reshape((h, w), order="F")


def regional_traces(movie: Any, rois: Sequence[np.ndarray]) -> np.ndarray:
    """(n, T) average of each pixel list over the frames, averaged in single precision as the movie's frames."""
    n = movie.shape[0]
    out = np.empty((len(rois), n))
    for s in range(0, n, BLOCK):
        b = np.asarray(movie[s:s + BLOCK])
        b = b.reshape(b.shape[0], -1, order="F")
        for i, pix in enumerate(rois):
            out[i, s:s + b.shape[0]] = b[:, pix].mean(axis=1, dtype=np.float32)
    return out


# --------------------------------------------------------------------------------------------- astrocytes
def astrocyte_mask(event_counts: np.ndarray, min_events: int = ASTRO_MIN_EVENTS,
                   min_px: int = ASTRO_MIN_PX) -> np.ndarray:
    """The pixels touched by at least ``min_events`` AQuA2 events, in 4-connected regions of at least ``min_px``
    pixels (``astrid.aqua2.event_counts`` gives the counts of one recording)."""
    counts = np.asarray(event_counts)
    out = np.zeros(counts.shape, dtype=bool)
    for comp in connected_components(counts >= min_events):
        if comp.size >= min_px:
            out[from_linear(counts.shape, comp)] = True
    return out


# --------------------------------------------------------------------------------------------- acceptance
def matlab_round(x: float | np.ndarray) -> np.ndarray:
    """Halves away from zero (numpy rounds halves to even)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def gaussian_1d(signal: np.ndarray, sigma: float) -> np.ndarray:
    """A 1-D Gaussian of 2 ceil(2 sigma) + 1 taps, replicate padding (MATLAB's imgaussfilt on a row)."""
    radius = int(np.ceil(2 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-(x**2) / (2 * sigma**2))
    return correlate1d(np.asarray(signal, dtype=np.float64), w / w.sum(), mode="nearest")


def moving_baseline(seq: np.ndarray, fps: float, half_window_s: float) -> np.ndarray:
    """Per frame i (1-based), the window seq(start:end), start = max(1, i - round(fps h)) and
    end = min(T, start + round(2 fps h)): its 10% quantile minus sigma norminv(0.15), sigma its noise level
    sqrt(mean(diff^2) / 2)."""
    seq = np.asarray(seq, dtype=np.float64)
    n = seq.size
    i = np.arange(1, n + 1)
    start = np.maximum(1, i - matlab_round(fps * half_window_s).astype(np.int64))
    end = np.minimum(n, start + matlab_round(fps * 2 * half_window_s).astype(np.int64))
    lengths = end - start + 1
    base = np.empty(n)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        win = seq[(start[rows] - 1)[:, None] + np.arange(length)]
        sigma = np.sqrt(0.5 * np.mean(np.diff(win, axis=1) ** 2, axis=1))
        base[rows] = np.quantile(win, BASELINE_QUANTILE, axis=1, method="hazen") - sigma * ndtri(BIAS_QUANTILE)
    return base


def longest_event_s(seq: np.ndarray, threshold: float, fps: float) -> float | None:
    """The longest run of ``seq`` >= ``threshold`` that starts and ends inside the trace, in seconds; None if
    there is none."""
    binary = (np.asarray(seq) >= threshold).astype(np.int8)
    changes = np.r_[0, np.diff(binary)]
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1) + 1
    if starts.size and not ends.size:
        return None                                  # an event that never ends
    rs: list[int] = []
    re: list[int] = []
    jj = 0
    for ss in starts:
        ee = ends[jj]
        while ee <= ss and jj < ends.size - 1:
            jj += 1
            ee = ends[jj]
        if ee > ss:
            rs.append(int(ss))
            re.append(int(ee))
    if not rs:
        return None
    return float(np.max((np.array(re) - np.array(rs)) / fps))


@dataclass(frozen=True)
class Acceptance:
    dff: np.ndarray
    accepted: bool
    reason: str = ""


def acceptance(trace: np.ndarray, fps: float) -> Acceptance:
    """Whether a candidate's average trace looks like an active neuron (module docstring, step 3). A trace
    whose events cannot be timed (an event that never ends, or none that starts and ends inside it while it
    passes the amplitude conditions) is not accepted; the analysis code stopped there, as MATLAB did."""
    seq = gaussian_1d(trace, SMOOTH_FRAMES)
    base = moving_baseline(seq, fps, 0.5 * BASELINE_WINDOW_S)
    dff, df = (seq - base) / base, seq - base
    d = np.diff(seq)
    noise = float(np.sqrt(0.5 * np.mean(d ** 2)))
    ok = bool(np.nanmax(df) >= MIN_RISE_NOISE * noise) and bool(np.nanmax(dff) >= MIN_DFF)
    if not ok:
        return Acceptance(dff, False, "too small")
    longest = longest_event_s(dff, EVENT_DFF, fps)
    if longest is None:
        return Acceptance(dff, False, "no event that starts and ends inside the trace")
    return Acceptance(dff, longest <= MAX_EVENT_S, "" if longest <= MAX_EVENT_S else "event too long")


# --------------------------------------------------------------------------------------------- proposals
def scaled_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``img`` mapped linearly so that its 1st and 99.9th percentiles inside the border match ``ref``'s, then
    clipped to ``ref``'s range."""
    inner = inner_mask(img.shape)
    lo_r, hi_r = np.percentile(ref[inner], SCALE_PERCENTILES)
    lo, hi = np.percentile(img[inner], SCALE_PERCENTILES)
    if hi <= lo:
        return np.full(img.shape, ref.min(), dtype=np.float32)
    return np.clip(lo_r + (img - lo) * (hi_r - lo_r) / (hi - lo), ref.min(), ref.max()).astype(np.float32)


def activity_proposals(activity: np.ndarray, max_proj: np.ndarray, budget: int, zmap: ZMap) -> np.ndarray:
    """The ``budget`` scored pixels of highest z inside the border, from ``zmap`` on the activity map scaled to
    the max projection; a component scored as a whole is kept whole at the cut."""
    inner = inner_mask(activity.shape)
    if budget <= 0:
        return np.zeros(activity.shape, dtype=bool)
    img = scaled_to(activity, max_proj.astype(np.float32))
    z = border_trimming(zmap(img))
    scored = inner & (z > Z_FLOOR)
    if not scored.any():
        return scored
    cut = np.sort(z[scored])[::-1][min(budget, int(scored.sum())) - 1]
    return scored & (z >= cut)


# --------------------------------------------------------------------------------------------- detection
@dataclass(frozen=True)
class RecordingDetection:
    candidates: list[np.ndarray]     # pixel lists, in scan order
    acceptance: list[Acceptance]
    rois: list[np.ndarray]           # the accepted candidates' 4-connected pieces, in scan order
    shape: tuple[int, int]
    max_proj: np.ndarray
    mean_proj: np.ndarray
    activity: np.ndarray | None


def detect_recording(movie: Any, fps: float, astro_mask: np.ndarray, zmap: ZMap | None = None,
                     activity: bool = True) -> RecordingDetection:
    """The ROIs of one recording (module docstring, steps 1-3). ``astro_mask``: (H, W), from
    ``astrocyte_mask`` (all False without AQuA2); ``zmap``: default SynQuant's (``astrid.synquant.zmap``)."""
    if zmap is None:
        from astrid.synquant import zmap as synquant_zmap
        zmap = synquant_zmap
    max_proj, mean_proj = projections(movie)
    keep_max = border_trimming(zmap(max_proj.astype(np.float32))) >= Z_CUT
    keep_avg = border_trimming(zmap(mean_proj.astype(np.float32))) >= Z_CUT
    proposals = keep_max | keep_avg
    act = None
    if activity:
        act = fast_activity(movie, fps)
        proposals = proposals | activity_proposals(act, max_proj, int(proposals.sum()), zmap)
    cand_mask = proposals & ~np.asarray(astro_mask, dtype=bool)
    candidates = connected_components(cand_mask)
    traces = regional_traces(movie, candidates) if candidates else np.empty((0, 0))
    accepted = [acceptance(tr, fps) for tr in traces]
    kept = cand_mask.copy()
    for comp, acc in zip(candidates, accepted, strict=True):
        if not acc.accepted:
            kept[from_linear(cand_mask.shape, comp)] = False
    return RecordingDetection(candidates, accepted, connected_components(kept), cand_mask.shape, max_proj,
                              mean_proj, act)


@dataclass(frozen=True)
class Registry:
    rois: list[np.ndarray]                   # merged ROIs: pixel lists, disjoint
    detected_in: np.ndarray                  # (n_recordings, n_rois) bool
    members: list[dict[int, int]]            # per merged ROI: {recording index: index of its ROI there}


def merge_rois(shape: tuple[int, int], per_recording: list[list[np.ndarray]],
               min_overlap: float = MIN_OVERLAP) -> Registry:
    """One ROI set from the recordings' ROIs (module docstring). Matches are merged strongest first; a pixel
    claimed by several merged ROIs goes to the one whose members contain it most often, and is dropped on a
    tie; merged ROIs are ordered by their first pixel in scan order."""
    nodes = [(r, i) for r, rois in enumerate(per_recording) for i in range(len(rois))]
    sets = {(r, i): set(per_recording[r][i].tolist()) for r, i in nodes}
    edges = []
    for a, (ra, ia) in enumerate(nodes):
        for rb, ib in nodes[a + 1:]:
            if ra == rb:
                continue
            sa, sb = sets[(ra, ia)], sets[(rb, ib)]
            inter = len(sa & sb)
            if inter:
                score = inter / min(len(sa), len(sb))
                if score >= min_overlap:
                    edges.append((score, (ra, ia), (rb, ib)))
    parent = {n: n for n in nodes}
    recs = {n: {n[0]} for n in nodes}

    def find(n: tuple[int, int]) -> tuple[int, int]:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for _, node_a, node_b in sorted(edges, key=lambda e: -e[0]):
        root_a, root_b = find(node_a), find(node_b)
        if root_a != root_b and not (recs[root_a] & recs[root_b]):      # at most one ROI per recording
            parent[root_b] = root_a
            recs[root_a] |= recs[root_b]
    clusters: dict[tuple[int, int], dict[int, int]] = {}
    for n in nodes:
        clusters.setdefault(find(n), {})[n[0]] = n[1]
    if not clusters:
        return Registry([], np.zeros((len(per_recording), 0), dtype=bool), [])
    support = []
    for members in clusters.values():
        count = np.zeros(shape)
        for r, i in members.items():
            count += mask_from_pixels(shape, per_recording[r][i])
        frac = count / len(members)
        support.append(np.where(frac >= 0.5, frac, 0.0))
    stack = np.array(support)
    best = stack.max(axis=0)
    unique_best = (stack == best).sum(axis=0) == 1
    cells = []
    for members, frac in zip(clusters.values(), support, strict=True):
        mask = (frac > 0) & (frac == best) & unique_best
        pieces = connected_components(mask)
        if pieces:
            cells.append((max(pieces, key=len), members))
    cells.sort(key=lambda c: int(c[0].min()))
    detected = np.zeros((len(per_recording), len(cells)), dtype=bool)
    for k, (_, members) in enumerate(cells):
        detected[list(members), k] = True
    return Registry([c[0] for c in cells], detected, [c[1] for c in cells])


def largest_piece_outside(shape: tuple[int, int], roi: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """``roi`` without the pixels where ``inside`` (column-major flat) is True, cut to its largest 4-connected
    piece; empty when nothing is left."""
    rest = roi[~inside[roi]]
    if not rest.size:
        return rest
    return max(connected_components(mask_from_pixels(shape, rest)), key=len)


def remove_pixels(reg: Registry, shape: tuple[int, int], mask: np.ndarray) -> Registry:
    """Every merged ROI without the pixels inside ``mask``, cut to its largest 4-connected piece; ROIs left
    empty are dropped and the rest stay ordered by their first pixel."""
    inside = np.asarray(mask, dtype=bool).ravel(order="F")
    cells = []
    for k, c in enumerate(reg.rois):
        piece = largest_piece_outside(shape, c, inside)
        if piece.size:
            cells.append((piece, k))
    cells.sort(key=lambda c: int(c[0].min()))
    keep = [k for _, k in cells]
    return Registry([c for c, _ in cells], reg.detected_in[:, keep], [reg.members[k] for k in keep])


def keep_rois(reg: Registry, keep: list[int]) -> Registry:
    return Registry([reg.rois[i] for i in keep], reg.detected_in[:, keep], [reg.members[i] for i in keep])


@dataclass(frozen=True)
class LocationDetection:
    rois: np.ndarray                 # (n, H, W) the merged ROI masks, the input of astrid.demix
    candidates: np.ndarray           # (H, W) every recording's candidates (astrid.demix's ``candidates``)
    detected_in: np.ndarray          # (n_recordings, n) which recordings detected each merged ROI
    astro_union: np.ndarray          # (H, W) the union of the recordings' astrocyte masks
    recordings: list[RecordingDetection]


def detect_location(movies: Sequence[Any], fps: float, astro_masks: Sequence[np.ndarray],
                    zmap: ZMap | None = None, activity: bool = True,
                    min_area_px: int = MIN_AREA_PX) -> LocationDetection:
    """The ROIs of one imaging location: every recording detected on its own (``detect_recording``, with its
    own astrocyte mask), the detections merged (``merge_rois``), the astrocyte masks of all recordings removed
    from the merged ROIs, and those under ``min_area_px`` pixels dropped (module docstring)."""
    if len(movies) != len(astro_masks):
        raise ValueError("give one astrocyte mask per recording")
    dets = [detect_recording(m, fps, a, zmap, activity) for m, a in zip(movies, astro_masks, strict=True)]
    shape = dets[0].shape
    astro_union = np.any(np.stack([np.asarray(a, dtype=bool) for a in astro_masks]), axis=0)
    merged = remove_pixels(merge_rois(shape, [d.rois for d in dets]), shape, astro_union)
    sized = keep_rois(merged, [i for i, c in enumerate(merged.rois) if c.size >= min_area_px])
    cand = np.zeros(shape, dtype=bool)
    for d in dets:
        for c in d.candidates:
            cand[from_linear(shape, c)] = True
    return LocationDetection(masks_from_lists(shape, sized.rois), cand, sized.detected_in, astro_union, dets)
