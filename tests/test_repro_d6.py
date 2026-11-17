"""D6's input files and runner (``reproduce/astrid_repro/d6.py``) and its scores (``stats.py d6-scores``, ``d6``) on a
small synthetic location: the reader and its checks against ``datasets.yaml``, the two movies made from the
channels, the cases and their territory classes, the estimators of the channels' scale, the kept transient size,
and one recording run end to end into the layout that ``stats.py`` reads, scored under both baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

pd = pytest.importorskip("pandas")

from astrid.patch import DEFAULT_BASELINE, baseline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import d6 as D  # noqa: E402
from astrid_repro import metrics, peers, stats  # noqa: E402

LOC, RECS = "A-M1_loc01", ["rec1", "rec2"]
SHAPE = (60, 80)
FPS = 30.9


def disc(cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


# ROI 1 in territory, 2 half in it, 3 away from it, 4 on the border, 5 not found in rec1
ROIS = [disc(20, 20, 3), disc(20, 33, 3), disc(42, 60, 3), disc(3, 50, 3), disc(44, 25, 3)]
TERRITORY = disc(20, 22, 9)


def toy(t: int = 1300, seed: int = 0) -> dict[str, np.ndarray]:
    """Two channels of one recording: the neurons' transients in ch2 and an astrocyte signal over the territory
    in ch1, which leaks into ch2 with b; photon-like noise in both."""
    rng = np.random.default_rng(seed)
    neurons = np.zeros(SHAPE, dtype=np.int32)
    for i, m in enumerate(ROIS):
        neurons[m] = i + 1
    events = np.zeros(SHAPE, dtype=np.int32)
    events[disc(20, 22, 12)] = 1
    events[TERRITORY] = 3
    tt = np.arange(t) / FPS
    astro = 1 + 0.5 * np.sin(2 * np.pi * 0.05 * tt) ** 2
    ch1 = 20 + 15 * TERRITORY[None] * astro[:, None, None] + rng.normal(0, 2, (t, *SHAPE))
    ch2 = 30 + rng.normal(0, 3, (t, *SHAPE))
    for m in ROIS:
        spikes = (rng.random(t) < 0.004).astype(float)
        act = np.convolve(spikes, np.exp(-np.arange(60) / 12.0))[:t]
        ch2 += m[None] * (40 * act)[:, None, None]
    g, b = 5.5, 0.02
    ch2 += b * ch1
    return {"neurons": neurons, "events": events, "ch1": ch1.astype(np.float32), "ch2": ch2.astype(np.float32),
            "g": g, "b": b}


def write_inputs(folder: Path, t: int = 1300, **change: float) -> Path:
    """The toy location in D6's input files (``d6``'s docstring), its constants from ``datasets.yaml`` unless
    changed; rec1 detected every ROI but ROI 5."""
    sc = toy(t)
    e = D.d6_entry()
    d = folder / LOC
    d.mkdir(parents=True)
    detected = np.ones((2, len(ROIS)), dtype=bool)
    detected[0, 4] = False
    cand = np.zeros(SHAPE, dtype=np.int32)
    cand[disc(50, 45, 2)] = 7
    const = {"fps": e["fps"], "um_per_px": e["um_per_px"], "border_px": e["border_px"]} | change
    np.savez(d / "location.npz", neurons=sc["neurons"], candidates=cand, events=sc["events"],
             recordings=np.array(RECS), detected=detected,
             **{k: np.float64(v) if k != "border_px" else np.int64(v) for k, v in const.items()})
    with h5py.File(d / f"{RECS[0]}.h5", "w") as f:
        f["ch1"], f["ch2"] = sc["ch1"], sc["ch2"]
        f["g"], f["b"] = np.float64(sc["g"]), np.float64(sc["b"])
    return d


def test_inputs_and_settings(tmp_path: Path) -> None:
    d = write_inputs(tmp_path / "data")
    info = D.read_location(d)
    rec = D.read_recording(d, RECS[0], info)
    assert info.name == LOC and info.recordings == RECS and rec.frames == 1300 and (rec.g, rec.b) == (5.5, 0.02)
    s = D.d6_settings(info)
    e = D.d6_entry()
    assert (s.tau_r, s.tau_d, s.max_iterations) == (e["tau_r"], e["tau_d"], 600)
    assert s.baseline_rule() == DEFAULT_BASELINE
    loc = D.location(info)
    fld = loc.field
    assert np.array_equal(fld.territory, info.events >= 2) and not fld.tagrfp_channel and not fld.tau.any()
    assert fld.px_scale == 1.0 and fld.candidates.max() == 1 and fld.border_px == 5
    with pytest.raises(ValueError, match="not D6's"):
        D.d6_settings(D.read_location(write_inputs(tmp_path / "other", fps=30.0)))
    with pytest.raises(ValueError, match="not a recording"):
        D.read_recording(d, "rec9", info)
    with pytest.raises(OSError):
        D.read_recording(d, RECS[1], info)                       # listed, but its file is not there
    with np.load(d / "location.npz") as z:
        a = dict(z)
    np.savez(d / "location.npz", **{**a, "detected": a["detected"][:, :3]})
    with pytest.raises(ValueError, match="detected"):
        D.read_location(d)


def test_movies(tmp_path: Path) -> None:
    d = write_inputs(tmp_path / "data", t=200)
    info = D.read_location(d)
    rec = D.read_recording(d, RECS[0], info)
    made = D.make_movies(rec, tmp_path / "movies", block=64)
    assert made["frames"] == 200 and made["check_max_rel"] <= D.CHECK_TOL
    with h5py.File(d / f"{RECS[0]}.h5", "r") as f:
        c1, c2 = f["ch1"][()].astype(np.float64), f["ch2"][()].astype(np.float64)
    with h5py.File(tmp_path / "movies" / "M.h5", "r") as fm, h5py.File(tmp_path / "movies" / "R.h5", "r") as fr:
        m, r = fm["mov"][()], fr["mov"][()]
    expect_r = (5.5 * (c2 - 0.02 * c1)).astype(np.float32)
    assert r.dtype == m.dtype == np.float32 and r.shape == (200, *SHAPE)
    np.testing.assert_array_equal(r, expect_r)
    np.testing.assert_array_equal(m, (c1 + expect_r).astype(np.float32))      # from ch1 and the stored R


def test_cases_and_classes(tmp_path: Path) -> None:
    info = D.read_location(write_inputs(tmp_path / "data", t=50))
    loc = D.location(info)
    assert D.cases_of(loc, info, "rec1") == [1, 2, 3]                   # 4 crosses the border, 5 not found
    assert D.cases_of(loc, info, "rec2") == [1, 2, 3, 5]
    assert D.territory_class(loc, 1) == ("in", 1.0)
    cls, share = D.territory_class(loc, 2)
    roi = loc.neurons == 2
    assert share == pytest.approx((roi & TERRITORY).sum() / roi.sum()) and cls == ("in" if share >= 0.5 else
                                                                                   "between")
    assert D.territory_class(loc, 3) == ("out", 0.0)


def test_scale(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    lam = rng.uniform(2, 20, (6, 512))
    raw1 = 7.0 * rng.poisson(lam, (700, 6, 512))                    # counts per photon 7 and 1.4
    raw2 = 1.4 * rng.poisson(lam, (700, 6, 512))
    assert D.gain_ratio(raw1, raw2) == pytest.approx(5.0, rel=0.03)
    mid = raw1[50:650][:, :, D.GAIN_COLUMNS]
    assert D.counts_per_photon(D.middle_frames(raw1)) == D.counts_per_photon(mid)
    # bleed-through: ch2 holds 0.03 of ch1's fine structure at its surges, with diffuse light of its own
    n, h, w = 400, 40, 48
    pattern = rng.normal(0, 1, (h, w))
    surge = np.zeros(n)
    for p in (60, 130, 200, 270, 340):
        surge[p - 3:p + 4] += np.hanning(7)
    sec1 = 50 + surge[:, None, None] * 20 * (1 + 0.5 * pattern)[None] + rng.normal(0, 0.05, (n, h, w))
    diffuse = 3 * np.sin(np.arange(n) / 9.0)[:, None, None] * np.ones((h, w))[None]
    sec2 = 0.03 * sec1 + 10 + diffuse + rng.normal(0, 0.05, (n, h, w))
    b, k = D.bleedthrough(sec1.astype(np.float32), sec2.astype(np.float32))
    assert k == 5 and b == pytest.approx(0.03, rel=0.02)
    mov = rng.normal(0, 1, (31 * 5 + 7, 4, 3)).astype(np.float32)
    np.testing.assert_array_equal(D.one_second(mov, block=62), mov[:155].reshape(5, 31, 4, 3).mean(1))
    # calibrate: b from the input file, g from raw files stored as MATLAB stores them (h5py: T, W, H)
    d = write_inputs(tmp_path / "data", t=200)
    files = []
    for name, x in (("raw1.h5", raw1), ("raw2.h5", raw2)):
        with h5py.File(tmp_path / name, "w") as f:
            f["data"] = x.transpose(0, 2, 1).astype(np.uint16)
        files.append(str(tmp_path / name))
    out = D.calibrate(d.parent, LOC, RECS[0], (files[0], files[1]))
    assert out["surges"] == 0 and np.isnan(out["b"]) and out["b_file"] == 0.02     # 6 s: no surge
    assert out["g"] == D.gain_ratio(raw1.astype(np.uint16), raw2.astype(np.uint16)) and out["g_file"] == 5.5


def test_transient_sizes() -> None:
    rng = np.random.default_rng(4)
    t = 3000
    spikes = np.zeros(t)
    spikes[[300, 900, 1500, 2100, 2700]] = 1
    neuron = 30 * np.convolve(spikes, np.exp(-np.arange(80) / 15.0))[:t]
    raw = 100 + neuron + rng.normal(0, 1, t)
    base = np.full(t, 95.0)
    ring = raw - base - 0.7 * 5.0                            # a flat ring of 5 above its baseline
    tr = dict.fromkeys(stats.METHODS)
    tr.update(roi=raw, ring=ring, demix=0.5 * neuron + rng.normal(0, 0.5, t), fissa=neuron.copy())
    row = stats.case_sizes(tr, raw, base)
    assert row["n_peaks"] == row["n_events"] == 5
    assert row["tsize_roi"] == 1.0 and row["tsize_fissa"] == pytest.approx(1.0, abs=0.05)
    assert row["tsize_demix"] == pytest.approx(0.5, abs=0.05) and np.isnan(row["tsize_cnmf"])
    ring_up = ring - 0.7 * neuron                            # the ring rises with the ROI: no neuron-only transient
    assert stats.case_sizes({**tr, "ring": ring_up}, raw, base)["n_events"] == 0
    assert stats.noise_sd(raw) == pytest.approx(1.0, rel=0.1)
    t2 = stats.d6_size_table([pd.DataFrame([{"location": f"L{i}", "territory": "in", **row} for i in range(3)])])
    assert t2["used"] == 3 and t2["transients"] == 15 and t2["in"]["demix"]["mean"] == row["tsize_demix"]


def test_one_recording(tmp_path: Path) -> None:
    write_inputs(tmp_path / "data")
    config = yaml.safe_load(D.CONFIG.read_text(encoding="utf-8"))
    config["datasets"]["D6"]["settings"] = {"max_iterations": 25}            # short fits
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "out"
    run = D.run_recording(tmp_path / "data", LOC, RECS[0], root, config=cfg, log=lambda s: None)
    assert run == root / f"{LOC}_{RECS[0]}" and not (run / "movies").exists()
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["cases"] == ["M-001", "M-002", "M-003"] and info["settings"]["max_iterations"] == 25
    assert (info["g"], info["b"]) == (5.5, 0.02) and info["movies"]["check_max_rel"] <= D.CHECK_TOL
    sc = toy()
    ch1, ch2 = sc["ch1"].astype(np.float64), sc["ch2"].astype(np.float64)
    r = (5.5 * (ch2 - 0.02 * ch1)).astype(np.float32)
    movies = {"M": (ch1 + r).astype(np.float32), "R": r}
    for m in D.MOVIES:
        index = json.loads((run / m / "compare" / "index.json").read_text())
        assert [c["spec"] for c in index] == info["cases"] and {c["movie"] for c in index} == {m}
        assert [c["territory"] for c in index][::2] == ["in", "out"] and index[0]["test"] == "d6"
        fits = pd.read_csv(run / m / "ours" / "fits.csv")
        assert list(fits.case) == info["cases"] and (fits.iterations <= 25).all()
        roi = D.location(D.read_location(tmp_path / "data" / LOC)).neurons == 1
        z = np.load(run / m / "compare" / "M-001.npz")
        assert not z["footprint"].any() and not z["amplitude"].any() and z["amplitude"].size == 1300
        expect = movies[m].transpose(0, 2, 1)[:, roi.T].mean(axis=1, dtype=np.float64)
        np.testing.assert_array_equal(z["roi_mean"], expect)                  # the field's column-major order
        movie, masks = peers.case_movie(run / m / "compare", "M-001")
        np.testing.assert_array_equal(movie[:, masks["target"]], movies[m][:, roi])      # the crop as it is
        with np.load(run / m / "ours" / "M-001.npz") as f:
            assert f["delivered"].shape == (1, 1300) and f["truth"].size == 0
    # the peers: the raw ROI average, as TUnCaT gives it
    for m in D.MOVIES:
        (run / m / "peers").mkdir()
        for name in info["cases"]:
            movie, masks = peers.case_movie(run / m / "compare", name)
            np.savez(run / m / "peers" / f"{name}.npz", tuncat_raw_roi=movie[:, masks["target"]].mean(axis=1))
    assert stats.main(["d6-scores", "--run", str(run)]) == 0
    assert stats.main(["d6-scores", "--run", str(run), "--baseline", "equal"]) == 0
    rep, eq = pd.read_csv(run / "scores.csv"), pd.read_csv(run / "scores_equal.csv")
    np.testing.assert_allclose(rep[rep.method == "roi"].leak_5hz, 1.0, atol=1e-5)   # 1 by construction (float32)
    for t in (rep, eq):
        ok = t[t.status == "ok"]
        assert set(ok.method) == {"demix", "ring", "roi"}
        assert (t[t.status != "ok"].leak_5hz == 1).all()
    pd.testing.assert_frame_equal(rep[rep.method.isin(["demix", "ring"])], eq[eq.method.isin(["demix", "ring"])])
    assert (run / "transient_size.csv").exists() and (run / "transient_size_equal.csv").exists()
    x = np.load(run / "M" / "peers" / "M-001.npz")["tuncat_raw_roi"]
    tr = stats.d6_traces(run, "M", "M-001", "equal")
    np.testing.assert_array_equal(tr["roi"], np.float64(x) - baseline(np.float64(x)[None], FPS)[0])
    with np.load(run / "M" / "ours" / "M-001.npz") as f:
        np.testing.assert_array_equal(tr["demix"], f["delivered"][0])
    y = np.load(run / "R" / "peers" / "M-001.npz")["tuncat_raw_roi"]
    astro = np.load(run / "M" / "compare" / "M-001.npz")["roi_mean"] - np.load(run / "R" / "compare" /
                                                                                "M-001.npz")["roi_mean"]
    expect = metrics.leakage(tr["roi"], stats.d6_traces(run, "R", "M-001", "equal")["roi"], astro, FPS, 5.0)
    assert eq.set_index(["case", "method"]).loc[("M-001", "roi"), "leak_5hz"] == pytest.approx(expect, abs=1e-15)
    assert expect != metrics.leakage(x, y, astro, FPS, 5.0)              # the baseline changed the raw ROI's
    with pytest.raises(ValueError, match="baseline"):
        stats.d6_traces(run, "M", "M-001", "other")
    for b, name in (("reported", "scores.csv"), ("equal", "scores_equal.csv")):
        out = tmp_path / f"table_{b}"
        assert stats.main(["d6", "--root", str(root), "--baseline", b, "--out", str(out)]) == 0
        rec = json.loads((out / "d6.json").read_text(encoding="utf-8"))
        assert rec["recordings"] == 1 and "transient_size" in rec and ("baseline" in rec) == (b == "equal")
        assert rec["cases_by_class"] == pd.read_csv(run / name).query("method == 'demix'").territory \
            .value_counts().to_dict()
