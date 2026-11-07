"""Tests of the TUnCaT evaluation port on synthetic data (GPL-2, as the rest of this folder)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuncat_eval import evaluation as ev  # noqa: E402
from tuncat_eval import transients as tr  # noqa: E402

CENTRES = (215, 925)


def trace_with(centres: tuple[int, ...] = CENTRES, n: int = 2000, amp: float = 10.0, noise: float = 0.01,
               seed: int = 0) -> np.ndarray:
    """Smooth transients (Gaussian, s.d. 5 frames) on small white noise: thresholds of tens of noise s.d., as
    TUnCaT's evaluation uses (30 to 80), sit far above the noise's wiggles."""
    t = np.arange(n)
    x = np.random.default_rng(seed).normal(0, noise, n)
    for c in centres:
        x += amp * np.exp(-0.5 * ((t - c) / 5.0) ** 2)
    return x


def labels_for(centres: tuple[int, ...] = CENTRES) -> np.ndarray:
    return np.array([[c - 9, c + 11, 1] for c in centres])     # 1-based [start, end, valid]


def test_noise_and_baseline() -> None:
    x = np.random.default_rng(1).normal(5.0, 2.0, 20000)
    assert abs(tr.noise_psd(x)[0] - 2.0) < 0.1                   # white noise: the PSD level is its s.d.
    assert abs(tr.ksd_mode(x) - 5.0) < 0.2


def test_detect_and_match() -> None:
    x = trace_with(noise=0.0)
    det = tr.detect(x, 0.0, 0.01, 30.0)
    assert len(det) == 2
    gt = labels_for()[:, :2] - 1
    assert tr.match(gt, det) == (2, 2)
    assert tr.match(gt, np.array([[500, 520]])) == (0, 0)        # no overlap: no match


def test_score_f1() -> None:
    x = np.stack([trace_with(seed=s) for s in range(3)])
    labels: list[np.ndarray | None] = [labels_for() for _ in range(3)]
    s = tr.score(x, labels, 30.0)
    assert (s.recall, s.precision, s.f1, s.n_neurons) == (1.0, 1.0, 1.0, 3)
    labels[1] = None                                            # an unlabelled neuron is left out
    assert tr.score(x, labels, 30.0).n_neurons == 2


def test_gt_from_clean() -> None:
    t = np.arange(3000)
    clean = np.zeros(t.size)
    for t0 in (300, 1200, 2200):
        clean[t0:] += np.exp(-(t[t0:] - t0) / 20.0)
    gt = tr.gt_transients_from_clean(clean)[0]
    assert gt.shape == (3, 3) and np.all(gt[:, 2] == 1)
    assert np.all(np.abs(gt[:, 0] - np.array([301, 1201, 2201])) <= 1)


def test_scorer_interface() -> None:
    x = np.stack([trace_with(seed=s) for s in range(2)])
    labels: list[np.ndarray | None] = [labels_for() for _ in range(2)]
    rec, pre, f1, n_neurons, n_gt = ev.Scorer(labels, None)(x, np.array([30.0, 5000.0]))
    assert f1[0] == 1.0 and f1[1] == 0.0 and n_neurons == 2 and n_gt == 4    # 5000 s.d.: above every peak


def test_cnmf_fill() -> None:
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 1:3, 1:3] = masks[1, 4:6, 4:6] = masks[2, 7:9, 7:9] = True
    kept = [0, 2]                                               # CaImAn removed the second neuron
    a = np.stack([masks[i].ravel(order="F") for i in kept], axis=1).astype(float)
    c = np.arange(2 * 5, dtype=float).reshape(2, 5) + 1
    out, removed = ev.cnmf_fill_missing({"C_gt": c, "YrA_gt": np.zeros_like(c), "A_thr": a}, masks)
    assert removed == [2] and out.shape == (3, 5)
    assert np.array_equal(out[0], c[0]) and not out[1].any() and np.array_equal(out[2], c[1])
