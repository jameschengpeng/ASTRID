"""The ROI detection (astrid.detection, astrid.synquant) on synthetic data."""

from __future__ import annotations

import numpy as np

import astrid
from astrid import detection as D
from astrid import synquant as S

FPS = 30.0


def blob_image(shape=(48, 48), seed=0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    img = 150 + rng.normal(0, 5, shape)
    centres = [(14, 14), (30, 34)]
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for cy, cx in centres:
        img += 250 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 6.0)
    return img.astype(np.float32), centres


def test_components_in_scan_order() -> None:
    m = np.zeros((6, 6), dtype=bool)
    m[4, 0] = m[0, 3] = m[1, 3] = True
    comps = D.connected_components(m)
    assert [list(c) for c in comps] == [[4], [18, 19]]            # column-major: (4,0) first, then (0..1, 3)


def test_astrocyte_mask() -> None:
    counts = np.zeros((40, 40))
    counts[5:20, 5:20] = 5                                        # 225 px with >= 4 events: kept
    counts[30:35, 30:35] = 9                                      # 25 px: too small
    counts[22:30, 5:20] = 3                                       # too few events
    mask = D.astrocyte_mask(counts)
    assert mask.sum() == 225 and mask[10, 10] and not mask[32, 32] and not mask[25, 10]


def test_synquant_finds_blobs() -> None:
    img, centres = blob_image()
    z = S.zmap(img)
    for cy, cx in centres:
        assert z[cy, cx] >= D.Z_CUT
    assert np.median(z) == D.Z_FLOOR


def test_fast_snr() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(6000) / FPS
    noise = rng.normal(0, 1, (2, t.size))
    snr = D.fast_snr(noise, FPS)
    assert np.all(snr < 0.2)
    active = noise + 3 * np.sin(2 * np.pi * 1.0 * t)
    assert D.fast_snr(active, FPS)[0] > 5


def transient_trace(t: np.ndarray, times: list[float], amp: float = 2.0) -> np.ndarray:
    tr = np.full(t.size, 100.0)
    for t0 in times:
        on = t >= t0
        tr[on] += 100 * amp * np.exp(-(t[on] - t0) / 0.5)
    return tr


def test_acceptance() -> None:
    rng = np.random.default_rng(2)
    t = np.arange(1800) / FPS
    neuron = transient_trace(t, [10, 25, 40]) + rng.normal(0, 1, t.size)
    assert D.acceptance(neuron, FPS).accepted
    flat = 100 + rng.normal(0, 1, t.size)
    acc = D.acceptance(flat, FPS)
    assert not acc.accepted and acc.reason == "too small"


def test_merge_rois() -> None:
    shape = (20, 20)
    a = D.connected_components(np.pad(np.ones((4, 4), bool), ((2, 14), (2, 14))))[0]
    b = D.connected_components(np.pad(np.ones((4, 4), bool), ((3, 13), (2, 14))))[0]     # overlaps a by 3/4
    c = D.connected_components(np.pad(np.ones((3, 3), bool), ((12, 5), (12, 5))))[0]
    reg = D.merge_rois(shape, [[a, c], [b]])
    assert len(reg.rois) == 2
    assert reg.detected_in.tolist() == [[True, True], [True, False]]


def test_detect_and_demix() -> None:
    rng = np.random.default_rng(3)
    shape, t = (48, 48), np.arange(1500) / FPS
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    movies = []
    for k in range(2):
        mov = 150 + rng.normal(0, 3, (t.size, *shape))
        for (cy, cx), times in (((14, 14), [5, 20, 35]), ((30, 34), [12, 30])):
            disc = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 9).astype(float)
            bright = 120 * disc
            mov += bright[None] + disc[None] * (transient_trace(t, [x + k for x in times], 4.0) - 100)[:, None, None]
        movies.append(mov.astype(np.float32))
    det = astrid.detect_rois(movies, fps=FPS)
    assert det.rois.shape[0] == 2 and det.detected_in.all()
    for cy, cx in ((14, 14), (30, 34)):
        assert det.rois[:, cy, cx].any()
    res = astrid.run_pipeline(movies, fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4)
    assert res.provenance["rois_detected"] and res.arrays()["fitted"].all()
