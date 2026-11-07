"""Tests of the package on synthetic data: settings, the fit, the outputs, the readers and the command line."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import astrid
from astrid import Settings
from astrid.api import make_field, resolve_overlaps
from astrid.patch import BaselineRule, baseline

FPS = 30.0


def disc(shape: tuple[int, int], cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def scene(seed: int = 0, t: int = 600, astro: bool = True) -> dict:
    """Two neurons, a slow astrocyte signal over a region next to the first, noise on a constant baseline."""
    rng = np.random.default_rng(seed)
    shape = (40, 48)
    rois = np.stack([disc(shape, 14, 14, 3), disc(shape, 25, 32, 3)])
    lag = np.arange(90) / FPS
    kernel = np.exp(-lag / 0.4) - np.exp(-lag / 0.05)
    spikes = (rng.random((2, t)) < 0.02).astype(float)
    neurons = np.stack([np.convolve(s, kernel)[:t] for s in spikes]) * 40
    mov = 100 + rng.normal(0, 2, (t, *shape))
    for i in range(2):
        mov += rois[i][None] * neurons[i][:, None, None]
    terr = disc(shape, 16, 22, 6)
    astro_trace = 30 * np.sin(2 * np.pi * 0.05 * np.arange(t) / FPS) ** 2
    if astro:
        mov += (terr * 1.0)[None] * astro_trace[:, None, None]
    return {"movie": mov.astype(np.float32), "rois": rois, "neurons": neurons, "territory": terr,
            "astro": astro_trace}


@pytest.fixture(scope="module")
def fitted() -> tuple[dict, astrid.Result]:
    sc = scene()
    res = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                       territory=sc["territory"], export_components=True)
    return sc, res


# --------------------------------------------------------------------------------------------- settings
def test_defaults_are_the_papers() -> None:
    s = Settings()
    assert (s.kappa_d, s.kappa_a, s.kappa_np, s.kappa_rho, s.f_a) == (2.0, 10.0, 0.1, 1.0, 0.2)
    assert (s.max_iterations, s.footprint_steps, s.drive_steps, s.check_every, s.tolerance) == \
        (600, 200, 300, 10, 1e-3)
    assert s.neuropil_cap is True
    assert s.baseline_rule() == BaselineRule(1.0, 60.0, 10.0, 0.0)
    assert s.tau_r is None and s.tau_d is None                    # set for each indicator


def test_code_names() -> None:
    p = Settings(tau_r=0.05, tau_d=0.4, kappa_d=3, kappa_np=0.2, kappa_rho=0.5, kappa_a=7).params(25.0)
    assert (p.kappa_s, p.kappa_b, p.kappa_t, p.kappa_a, p.fs) == (3.0, 0.2, 0.5, 7.0, 25.0)
    assert (p.n_outer, p.n_h, p.n_deconv, p.block, p.tol_target, p.tol) == (600, 200, 300, 10, 1e-3, 1e-7)


def test_settings_refused() -> None:
    with pytest.raises(ValueError):
        Settings(kappa_d=-1)
    with pytest.raises(ValueError):
        Settings(tau_r=0.5, tau_d=0.1)
    with pytest.raises(ValueError):
        Settings.from_dict({"kappa_s": 2})                         # the code's name, not the paper's
    with pytest.raises(ValueError, match="time constants"):
        Settings().with_time_constants(None, None)
    with pytest.raises(ValueError):
        Settings(baseline_window_s=60.0, trend_pct_per_s=-0.1)    # a trend needs the whole recording


def test_settings_yaml(tmp_path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("kappa_d: 4\nkappa_np: 0.3\ntau_r: 0.045\ntau_d: 0.142\n", encoding="utf-8")
    s = Settings.from_yaml(path)
    assert (s.kappa_d, s.kappa_np, s.tau_r, s.tau_d) == (4, 0.3, 0.045, 0.142)


# --------------------------------------------------------------------------------------------- baseline
def test_baseline_moving_flat() -> None:
    x = np.full((3, 3000), 50.0, dtype=np.float32)
    assert np.allclose(baseline(x, FPS), 50.0)


def test_baseline_trend() -> None:
    n, fps, rate = 600, 20.0, -0.5
    t = np.arange(n) / fps
    trend = 1 + rate / 100 * (t - (n - 1) / 2 / fps)
    x = (200.0 * trend)[None, :] * np.ones((2, 1))
    b = baseline(x, fps, BaselineRule(3.0, None, 10.0, rate))
    assert np.allclose(b, x)


# --------------------------------------------------------------------------------------------- the fit
def test_fit_recovers_neurons(fitted) -> None:
    sc, res = fitted
    a = res.arrays()
    assert a["fitted"].all() and a["converged"].all()
    assert list(a["n_astro"]) == [1, 1]
    for i in range(2):
        assert np.corrcoef(a["delta_f__rec0"][i], sc["neurons"][i])[0, 1] > 0.99
        assert np.corrcoef(res.dff[i], sc["neurons"][i])[0, 1] > 0.99


def test_contributions_add_up(fitted) -> None:
    sc, res = fitted
    a = res.arrays()
    raw = np.stack([sc["movie"][:, m].mean(axis=1) for m in sc["rois"]]).astype(np.float64)
    base = a["baseline__rec0"]
    total = a["dff__rec0"] + sum(a[f"contrib_{k}__rec0"] for k in ("neighbours", "unclassified", "astro",
                                                                   "neuropil"))
    assert np.allclose(total, (raw - base) / base, atol=1e-6)


def test_component_rows(fitted) -> None:
    _, res = fitted
    a = res.arrays()
    assert list(a["component_kind"]) == ["neuron", "astro", "neuropil"] * 2
    assert list(a["component_roi"]) == [0, 0, 0, 1, 1, 1]
    for j in range(len(a["component_roi"])):
        fp = astrid.component_footprint(a, j, res.shape)
        assert fp.min() >= 0 and abs(fp.sum() - 1) < 1e-9
    assert a["component_dff__rec0"].shape == (6, 600)


def test_no_territory_no_astro() -> None:
    sc = scene(astro=False)
    res = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4)
    assert list(res.arrays()["n_astro"]) == [0, 0]
    assert res.provenance["territory"] is False


def test_save(fitted, tmp_path) -> None:
    _, res = fitted
    path = res.save(tmp_path)
    with np.load(path) as z:
        assert np.array_equal(z["dff__rec0"], res.dff, equal_nan=True)
    prov = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert prov["settings"]["kappa_d"] == 2.0 and prov["fps"] == FPS


def test_border_and_overlaps() -> None:
    sc = scene(t=300)
    rois = sc["rois"].copy()
    rois[0, 0:3, 0:3] = True                                         # touches the edge
    res = astrid.demix(sc["movie"], rois, fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4, border_px=2)
    assert not res.rois[0].fitted and "border_px" in res.rois[0].error
    overlapping = np.stack([sc["rois"][0], sc["rois"][0] | disc((40, 48), 14, 17, 3)])
    with pytest.raises(ValueError, match="more than one mask"):
        make_field(overlapping, FPS, 2.0)
    resolved, report = resolve_overlaps(overlapping)
    assert (resolved.sum(axis=0) <= 1).all() and report["pixels_reassigned"] > 0


def test_joint_recordings() -> None:
    a, b = scene(seed=1, t=300), scene(seed=2, t=400)
    res = astrid.demix([a["movie"], b["movie"]], a["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                       territory=a["territory"], recording_names=["first", "second"])
    assert res.trace("dff", "first").shape == (2, 300) and res.trace("dff", "second").shape == (2, 400)


def test_parallel_is_the_same(fitted) -> None:
    sc, res = fitted
    par = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                       territory=sc["territory"], workers=2)
    assert np.array_equal(par.dff, res.dff)


def test_evidence_map() -> None:
    img = np.full((60, 60), 100.0)
    img[25:35, 25:35] = 300.0
    rho = astrid.evidence.evidence_map(img)
    assert rho.min() >= 0 and rho.max() <= 1 and rho[30, 30] == 1.0
    sc = scene(t=300)
    res = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                       territory=sc["territory"], evidence=np.full((40, 48), 0.5))
    assert res.provenance["evidence"] is True and res.rois[0].fitted


def test_ablation_without_astro_components(fitted) -> None:
    sc, _ = fitted
    res = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                       territory=sc["territory"], astro_components=False)
    assert list(res.arrays()["n_astro"]) == [0, 0]


def test_frozen_settings_change_the_fit(fitted) -> None:
    sc, res = fitted
    other = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, tau_r=0.05, tau_d=0.4,
                         territory=sc["territory"], settings=dataclasses.replace(Settings(), kappa_np=10.0))
    assert not np.array_equal(other.dff, res.dff)
