"""D7's input files and runner (``reproduce/astrid_repro/d7.py``) on a small synthetic recording: the reader and
its checks against ``datasets.yaml``, D7's baseline and ΔF/F0, D7's rules (no minimum ROI size, territory sites
on territory, no evidence map), and one recording run end to end into the layout that ``stats.py d7`` reads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

pd = pytest.importorskip("pandas")

from astrid.api import Movie  # noqa: E402
from astrid.patch import BaselineRule, baseline, dilate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cases as C  # noqa: E402
from astrid_repro import d7 as D  # noqa: E402
from astrid_repro import metrics, peers, stats  # noqa: E402

LOC, REC = "D7_location1", "rec1"
SHAPE = (70, 90)


def disc(cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


# small nuclei: ROI 3 has 9 pixels (below D1's minima), ROI 5 lies on the border
ROIS = [disc(22, 22, 2), disc(22, 66, 2), disc(48, 30, 1.5), disc(50, 52, 2), disc(3, 45, 2)]


def toy(t: int = 160, seed: int = 0) -> dict[str, np.ndarray]:
    """One recording: five ROIs, a blob of territory (1 to 2 events) around ROIs 3 and 4 holding the largest
    event, a bleaching movie with the ROIs' activity and an astrocyte signal on the largest event."""
    rng = np.random.default_rng(seed)
    e = D.d7_entry(LOC)
    fps, rate = float(e["fps"]), float(e["settings"]["trend_pct_per_s"])
    neurons = np.zeros(SHAPE, dtype=np.int32)
    for i, m in enumerate(ROIS):
        neurons[m] = i + 1
    events = np.zeros(SHAPE, dtype=np.int32)
    events[disc(50, 40, 12)] = 1
    events[disc(50, 42, 7)] = 2
    largest = disc(50, 42, 7)
    act = rng.gamma(1.0, 0.15, (len(ROIS), t))
    trend = 1 + rate / 100 * (np.arange(t) / fps - (t - 1) / 2 / fps)
    astro = 0.4 * (1 + np.sin(2 * np.pi * 0.1 * np.arange(t) / fps))
    mov = 100 + rng.normal(0, 2, (t, *SHAPE))
    for v, m in enumerate(ROIS):
        mov += m[None] * (80 * act[v])[:, None, None]
    mov += largest[None] * (40 * astro)[:, None, None]
    mov *= trend[:, None, None]
    max_proj = mov.max(axis=0).astype(np.float32)
    return {"neurons": neurons, "events": events, "largest": largest, "mov": mov.astype(np.float32),
            "max_proj": max_proj, "fps": fps, "rate": rate}


def write_inputs(folder: Path, t: int = 160, **change: float) -> Path:
    """The toy recording in D7's input files (``d7``'s docstring), the location's constants from
    ``datasets.yaml`` unless changed."""
    sc = toy(t)
    e = D.d7_entry(LOC)
    d = folder / LOC
    d.mkdir(parents=True)
    const = {"fps": e["fps"], "um_per_px": e["um_per_px"], "border_px": e["border_px"],
             "rate_pct_per_s": e["settings"]["trend_pct_per_s"], "tau_r": e["tau_r"], "tau_d": e["tau_d"]} | change
    np.savez(d / "location.npz", neurons=sc["neurons"], recordings=np.array([REC]),
             **{k: np.float64(v) if k != "border_px" else np.int64(v) for k, v in const.items()})
    with h5py.File(d / f"{REC}.h5", "w") as f:
        f["mov"] = sc["mov"]
        f["events"] = sc["events"]
        f["largest_event"] = sc["largest"]
        f["max_proj"] = sc["max_proj"]
    return d


def test_inputs_and_settings(tmp_path: Path) -> None:
    d = write_inputs(tmp_path / "data")
    info = D.read_location(d)
    rec = D.read_recording(d, REC, info)
    assert info.name == LOC and rec.frames == 160 and rec.largest_event.dtype == bool
    s = D.d7_settings(info)
    e = D.d7_entry(LOC)
    assert (s["injection"].tau_r, s["injection"].tau_d) == (e["tau_r"], e["tau_d"])
    assert (s["injection"].max_iterations, s["leakage"].max_iterations) == (600, 2000)
    rule = s["injection"].baseline_rule()
    assert rule == BaselineRule(3.0, None, 10.0, e["settings"]["trend_pct_per_s"])
    loc = D.location(info, rec)
    fld = loc.field
    assert np.array_equal(fld.territory, rec.events >= 1) and not fld.tagrfp_channel and not fld.tau.any()
    assert fld.px_scale == 2.0 / 2.07 and not fld.candidates.any() and loc.max_proj.dtype == np.float64
    assert np.array_equal(loc.astro_label == 1, rec.largest_event)
    with pytest.raises(ValueError, match="not datasets.yaml"):
        D.d7_settings(D.read_location(write_inputs(tmp_path / "other", rate_pct_per_s=-0.1)))
    with pytest.raises(ValueError, match="not a D7 location"):
        D.d7_entry("D1")
    with h5py.File(d / f"{REC}.h5", "a") as f:
        del f["events"]
        f["events"] = np.zeros((5, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="events"):
        D.read_recording(d, REC, info)


def test_traces() -> None:
    sc = toy()
    fps, rate = sc["fps"], sc["rate"]
    rule = BaselineRule(3.0, None, 10.0, rate)
    t = np.arange(400)
    trend = 1 + rate / 100 * (t / fps - (t.size - 1) / 2 / fps)
    line = 50.0 * trend                                      # a dimming line is its own baseline
    np.testing.assert_allclose(baseline(line[None], fps, rule)[0], line, rtol=1e-12)
    x = np.random.default_rng(2).normal(0, 1, 300)
    w = np.exp(-np.arange(-3, 4) ** 2 / (2 * 1.3 ** 2))
    np.testing.assert_allclose(D.gaussian_1d(x, 1.3), ndi.correlate1d(x, w / w.sum(), mode="nearest"))
    info = D.LocationInfo(LOC, sc["neurons"], fps, 2.07, 5, rate, 0.0246, 2.4126, [REC])
    rec = D.Recording(REC, Path("unused"), sc["mov"].shape[0], sc["events"], sc["largest"], sc["max_proj"])
    loc = D.location(info, rec)
    movie = Movie(REC, data=sc["mov"])
    smoothed, raw = D.roi_dff(loc, movie, rule)
    assert smoothed.shape == raw.shape == (len(ROIS), 160)
    roi, ring = D.ring_mask(loc, 2)
    assert not (ring & roi).any() and not (ring & (loc.neurons > 0)).any() and ring.any()
    d = ndi.distance_transform_edt(~roi)
    assert np.array_equal(ring, (d > 2 * 2 / 2.07) & (d <= 8 * 2 / 2.07) & (loc.neurons == 0))
    mov = sc["mov"].astype(np.float64)
    f_roi, f_ring = mov[:, roi].mean(axis=1), mov[:, ring].mean(axis=1)
    f = f_roi - 0.7 * f_ring
    expect = (f - baseline(f[None], fps, rule)[0]) / baseline(f_roi[None], fps, rule)[0]
    np.testing.assert_allclose(raw[1], expect, rtol=1e-10, atol=1e-12)    # over the ROI's own baseline
    sigma = 6 / 30.9 * fps
    np.testing.assert_allclose(smoothed[1], D.gaussian_1d(metrics.lowpass_fourier(raw[1], fps, 5.0), sigma))
    a = D.astro_dff(loc, movie, sc["largest"], rule)
    tr = mov[:, sc["largest"]].mean(axis=1)
    f0 = baseline(tr[None], fps, rule)[0]
    np.testing.assert_allclose(a, metrics.lowpass_fourier((tr - f0) / f0, fps, 5.0), rtol=1e-10, atol=1e-12)


def test_rules() -> None:
    sc = toy()
    info = D.LocationInfo(LOC, sc["neurons"], sc["fps"], 2.07, 5, sc["rate"], 0.0246, 2.4126, [REC])
    rec = D.Recording(REC, Path("unused"), 160, sc["events"], sc["largest"], sc["max_proj"])
    loc = D.location(info, rec)
    dff = {REC: np.random.default_rng(1).gamma(1.0, 0.1, (len(ROIS), 160))}
    terr = loc.field.territory
    assert C.leakage_targets(loc, dff, 10, C.D7_RULES.target_min_px) == [3, 4]      # ROI 3: 9 px, kept
    assert C.leakage_targets(loc, dff) == [4]                                       # D1's minimum drops it
    assert set(C.injection_sources(loc, dff, 4, C.D7_RULES.source_min_px)) == {1, 2}
    assert C.injection_sources(loc, dff) == []                                      # D1's 20 px
    kw = {"min_events": C.D7_RULES.site_min_events, "evidence": C.D7_RULES.site_evidence}
    inside, away = C.free_sites(loc, True, **kw), C.free_sites(loc, False, **kw)
    assert inside and away and C.free_sites(loc, True, 0, **kw) == []
    near = dilate(terr, C.CONTROL_CLEAR_PX)
    assert all(terr[r, c] for r, c in inside) and not any(near[r, c] for r, c in away)
    assert C.free_sites(loc, True) == []                  # D1's rule: 4 events and evidence, none here
    cases, ch = C.injection_cases(loc, dff, C.D7_RULES)
    assert len(cases) == 2 * len(ch["sites"]) and ch["sites"][0][2] == "astro"
    leak, ch = C.leakage_cases(loc, dff, {REC: np.ones(160)}, 1, C.D7_RULES)
    assert ch["targets"] == [3, 4] and ch["astro_source"] == "A-01" and len(leak) == 6
    assert (leak[1].astro.shape & (loc.neurons == 3)).sum() >= 0.5 * (loc.neurons == 3).sum()


def test_one_recording(tmp_path: Path) -> None:
    write_inputs(tmp_path / "data")
    config = yaml.safe_load(D.CONFIG.read_text(encoding="utf-8"))
    config["datasets"][LOC]["settings"]["max_iterations"] = 30          # short fits
    config["datasets"][LOC]["leakage_test_settings"] = {"max_iterations": 40}
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "out"
    run = D.run_recording(tmp_path / "data", LOC, REC, root, config=cfg, log=lambda s: None)
    assert run == root / f"{LOC}_{REC}"
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["fps"] == 22.7 and info["settings"]["leakage"]["max_iterations"] == 40
    assert info["settings"]["injection"]["baseline_window_s"] is None
    assert info["choices"]["leakage"]["targets"] == [3, 4] and info["choices"]["leakage"]["astro_source"] == "A-01"
    every, todo = peers.pending(run / "compare")
    assert every == info["cases"]["injection"] + info["cases"]["leakage"] and todo == every
    results = tmp_path / "peer_results"                    # the raw ROI's trace, as TUnCaT gives it
    results.mkdir()
    for name in every:
        movie, masks = peers.case_movie(run / "compare", name)
        np.savez(results / f"{name}.npz", tuncat_raw_roi=movie[:, masks["target"]].mean(axis=1))
    assert D.main(["score", "--run", str(run), "--results", str(results)]) == 0
    index = json.loads((run / "compare" / "index.json").read_text())
    inj, leak = stats.d7_injection_table(run, index), stats.d7_leakage_table(run, index)
    assert set(inj.method) == set(stats.METHODS) and set(leak.method) == set(stats.METHODS)
    ok = inj[inj.status == "ok"]
    assert set(ok.method) == {"demix", "ring", "roi"}                      # the others missing: failures
    np.testing.assert_allclose(leak[leak.method == "roi"].leak, 1.0, atol=1e-5)   # 1 by construction (float32)
    scores = pd.read_csv(run / "scores" / "injection_scores.csv", float_precision="round_trip")
    first = scores[(scores.case == index[0]["spec"]) & (scores.method == "roi")].iloc[0]
    truth = np.load(run / "compare" / f"{index[0]['spec']}.npz")["truth"]
    roi = np.load(results / f"{index[0]['spec']}.npz")["tuncat_raw_roi"]
    assert first.err_05hz == metrics.error(roi, truth, 22.7)               # scored at the location's rate
