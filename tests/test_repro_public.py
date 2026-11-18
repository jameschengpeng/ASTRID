"""The D2-D5 reproduction code on small synthetic data: the readers, the ring, the cross-validation and the
tables' tests and Holm families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from astrid.patch import baseline as astrid_baseline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cv as C  # noqa: E402
from astrid_repro import metrics  # noqa: E402
from astrid_repro import public_data as D  # noqa: E402
from astrid_repro import ring as R  # noqa: E402
from astrid_repro import tables_public as T  # noqa: E402


def _cells(*arrays: np.ndarray) -> np.ndarray:
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def _v73(path: Path, key: str, data: np.ndarray) -> None:
    """An HDF5 file with MATLAB's v7.3 header."""
    with h5py.File(path, "w", userblock_size=512) as f:
        f[key] = data
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM")


def _masks() -> np.ndarray:
    m = np.zeros((3, 40, 30), dtype=bool)
    m[0, 5:9, 5:9] = True
    m[1, 20:25, 10:14] = True
    m[2, 22:30, 12:20] = True                      # overlaps the second
    return m


def test_masks_both_formats_and_labels(tmp_path: Path) -> None:
    m = _masks()
    scipy.io.savemat(tmp_path / "v5.mat", {"FinalMasks": m.transpose(2, 1, 0).astype(np.uint8)})
    np.testing.assert_array_equal(D.read_masks(tmp_path / "v5.mat"), m)
    _v73(tmp_path / "v73.mat", "FinalMasks", m.astype(np.uint8))
    np.testing.assert_array_equal(D.read_masks(tmp_path / "v73.mat"), m)
    lab = np.array([[1, 5, 1], [10, 12, 0]])
    scipy.io.savemat(tmp_path / "out.mat", {"output": _cells(lab, np.zeros((0, 0)), np.array([[3, 4, 1]]))})
    got = D.read_labels(tmp_path / "out.mat")
    assert len(got) == 3 and got[1] is None
    np.testing.assert_array_equal(got[0], lab)
    np.testing.assert_array_equal(got[2], [[3, 4, 1]])


def _peers(root: Path, video: str, n: int, t: int, value: float, cnmf_rows: int | None = None) -> None:
    for folder, key, shape in (("traces_FISSA_Raw/alpha= 0.010", "unmixed_traces", (n, t)),
                               ("traces_ours_Raw/alpha= 1.000", "traces_nmfdemix", (t, n))):
        (root / folder).mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(root / folder / f"{video}.mat", {key: np.full(shape, value)})
    (root / "traces_CNMF_Raw_p1").mkdir(parents=True, exist_ok=True)
    k = n if cnmf_rows is None else cnmf_rows
    scipy.io.savemat(root / "traces_CNMF_Raw_p1" / f"{video}.mat",
                     {"C_gt": np.ones((k, t)), "YrA_gt": np.full((k, t), value)})
    (root / "traces_ours_Raw/raw").mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(root / "traces_ours_Raw/raw" / f"{video}.mat",
                     {"traces": np.full((t, n), 3.0), "bgtraces": np.full((t, n), 1.0)})


def test_peer_and_ours_traces(tmp_path: Path) -> None:
    _peers(tmp_path / "a", "v1", 3, 6, 2.0, cnmf_rows=2)
    (tmp_path / "b/traces_FISSA_Raw/alpha=0.0005").mkdir(parents=True)
    scipy.io.savemat(tmp_path / "b/traces_FISSA_Raw/alpha=0.0005/v1.mat", {"unmixed_traces": np.zeros((3, 6))})
    fissa = D.peer_traces([tmp_path / "a", tmp_path / "b"], "FISSA", "v1")
    assert list(fissa) == ["a=0.01", "a=0.0005"]
    assert D.peer_traces(tmp_path / "a", "TUnCaT", "v1")["a=1"].shape == (3, 6)
    assert D.peer_traces(tmp_path / "a", "bgsubs", "v1")["bgsubs"][0, 0] == 2.0
    np.testing.assert_array_equal(D.peer_traces(tmp_path / "a", "CNMF", "v1")["p1"], np.full((2, 6), 3.0))
    masks = np.ones((3, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="cnmf_fill"):
        D.peer_traces(tmp_path / "a", "CNMF", "v1", masks=masks)
    filled = D.peer_traces(tmp_path / "a", "CNMF", "v1", masks=masks,
                           cnmf_fill=lambda m, k: np.vstack([m["C_gt"] + m["YrA_gt"], np.zeros((1, 6))]))
    assert filled["p1"].shape == (3, 6) and filled["p1"][2].sum() == 0
    with pytest.raises(ValueError, match="more than one root"):
        D.peer_traces([tmp_path / "a", tmp_path / "a"], "FISSA", "v1")
    for root, setting in ((tmp_path / "o1", "ks2_kb0.1"), (tmp_path / "o2", "ks32_kb0.1")):
        (root / "ABO" / setting / "v1").mkdir(parents=True)
        np.savez(root / "ABO" / setting / "v1" / "result.npz", **{"dff__v1": np.ones((3, 6))})
    assert list(D.ours_traces([tmp_path / "o1", tmp_path / "o2"], "ABO", "v1")) == ["ks2_kb0.1", "ks32_kb0.1"]


def _toy_scorer(x: np.ndarray, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    s = float(np.abs(x).sum())
    f = 1 / (1 + np.abs(s - thr))
    return f, f, f, x.shape[0], 7


def test_f1_grid_failure_rule_and_cv() -> None:
    seen = []

    def spy(x: np.ndarray, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        seen.append(x.copy())
        return _toy_scorer(x, thr)

    g = C.f1_grid({"a": np.array([[1.0, np.nan], [1.0, 1.0]]), "b": np.ones((2, 2))}, spy, [1, 2, 3], n_rois=2)
    np.testing.assert_array_equal(seen[0], [[0, 0], [1, 1]])           # the missing trace detects nothing
    assert g.f1.shape == (2, 3) and g.n_gt == 7
    with pytest.raises(ValueError, match="ROIs"):
        C.f1_grid({"a": np.ones((3, 2))}, spy, [1], n_rois=2)
    f = np.array([[[0.5, 0.9], [0.9, 0.1]], [[0.4, 0.8], [0.8, 0.2]], [[0.1, 0.2], [0.3, 0.95]]])
    grids = [C.Grid(["s1", "s2"], np.array([10.0, 20.0]), f[i], f[i], f[i], 1, 1) for i in range(3)]
    res = C.loo_cv(["v1", "v2", "v3"], grids)
    assert res.setting == ["s2", "s2", "s1"]                            # a tie: settings before thresholds
    np.testing.assert_array_equal(res.threshold, [20.0, 10.0, 20.0])
    np.testing.assert_allclose(res.f1, [0.1, 0.8, 0.2])
    assert res.mean_f1 == pytest.approx(1.1 / 3)
    np.testing.assert_array_equal(C.trim({"a": np.ones((2, 5)), "b": np.ones((2, 4))})["a"].shape, (2, 4))


def test_tuncat_scores_and_grid_files(tmp_path: Path) -> None:
    f = np.random.default_rng(0).random((4, 2, 3))
    scipy.io.savemat(tmp_path / "s.mat", {"list_F1": f, "list_recall": f, "list_precision": f,
                                          "list_thred_ratio": np.array([30, 35, 40]),
                                          "list_alpha": np.array([0.1, 1.0])})
    grids = C.read_tuncat_scores(tmp_path / "s.mat")
    assert len(grids) == 4 and grids[0].settings == ["a=0.1", "a=1"]
    np.testing.assert_array_equal(grids[2].f1, f[2])
    scipy.io.savemat(tmp_path / "one.mat", {"list_F1": f[:, 0], "list_recall": f[:, 0], "list_precision": f[:, 0],
                                            "list_thred_ratio": np.array([30, 35, 40])})
    assert C.read_tuncat_scores(tmp_path / "one.mat", "p1")[1].settings == ["p1"]
    C.save_grid(tmp_path / "g.npz", grids[1])
    back = C.load_grid(tmp_path / "g.npz")
    assert back.settings == grids[1].settings
    np.testing.assert_array_equal(back.f1, grids[1].f1)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_cell_cv() -> None:
    cell_of = {"c1_1": "c1", "c1_2": "c1", "c2_1": "c2", "c3_1": "c3"}
    per = {"s2": {"c1_1": 0.5, "c1_2": np.nan, "c2_1": 0.9, "c3_1": 0.6},
           "s1": {"c1_1": 0.7, "c1_2": 0.9, "c2_1": 0.2, "c3_1": 0.6}}
    res = C.cell_cv(per, cell_of, ["c1", "c2", "c3"])
    assert res is not None and res["settings"] == ["s1", "s2"] and res["n_recordings"] == 4
    assert res["chosen"] == ["s2", "s1", "s2"]
    np.testing.assert_allclose(res["r"], [0.5, 0.2, 0.6])
    assert C.cell_cv({"s": {"c1_1": np.nan}}, cell_of, ["c1"]) is None


def test_ring(tmp_path: Path) -> None:
    m = _masks()
    rings = R.ring_masks(m, 1.0)                                  # 4 to 16 px at 1 um/px
    assert not (rings & m.any(axis=0)).any()
    yy, xx = np.nonzero(rings[0])
    d = np.sqrt((yy[:, None] - np.argwhere(m[0])[:, 0]) ** 2 + (xx[:, None] - np.argwhere(m[0])[:, 1]) ** 2)
    assert d.min(axis=1).min() > 4 and d.min(axis=1).max() <= 16
    mov = np.random.default_rng(1).normal(100, 5, (25, 40, 30)).astype(np.float32)
    with h5py.File(tmp_path / "v.h5", "w") as f:
        f["mov"] = mov
    out = R.video_ring(tmp_path / "v.h5", m, 1.0)
    roi_mean = np.array([mov[:, k].astype(np.float64).mean(axis=1) for k in m])
    ring_mean = np.array([mov[:, k].astype(np.float64).mean(axis=1) for k in rings])
    np.testing.assert_allclose(out["roi_mean"], roi_mean, rtol=1e-12)
    np.testing.assert_allclose(out["trace"], roi_mean - 0.7 * ring_mean, rtol=1e-12)
    assert out["ring_px"].tolist() == rings.reshape(3, -1).sum(axis=1).tolist()
    with pytest.raises(ValueError, match="empty"):
        R.check_rings(m, np.zeros_like(m))


def _export(folder: Path, name: str, rng: np.random.Generator, t: int = 60) -> None:
    roi = np.zeros((20, 20), dtype=bool)
    roi[8:12, 8:12] = True
    with h5py.File(folder / f"{name}.h5", "w") as f:
        f["mov"] = rng.integers(0, 100, (t, 20, 20)).astype(np.int16)
        f["roi"] = roi
        f["valid"] = np.arange(t) > 2
        f["predicted"] = rng.random(t)
        f["fmean_roi"] = rng.random(t)
        f["fmean_neuropil"] = rng.random(t)


def test_cai1_ground_truth_and_reference_rows(tmp_path: Path) -> None:
    fps = 60.0
    k = D.single_spike_kernel("GCaMP6f", fps)
    assert k.max() == 1.0 and abs(np.argmax(k) / fps - 0.045) <= 1 / fps
    t = np.arange(600) / fps
    pred = D.predicted_calcium(np.array([1.0, 1.0]), t, "GCaMP6f")
    np.testing.assert_allclose(pred[60:60 + 50], 2 * k[:50])                  # two spikes in frame 60
    x = np.random.default_rng(2).random(100)
    ok = np.ones(100, dtype=bool)
    assert D.corr(x, 2 * x + 1, ok) == pytest.approx(1.0)
    assert np.isnan(D.corr(x, x, np.arange(100) < 9))
    _export(tmp_path, "r1", np.random.default_rng(3))
    rows = R.cai1_reference_rows(tmp_path / "r1.h5")
    with h5py.File(tmp_path / "r1.h5", "r") as f:
        mov, roi = np.asarray(f["mov"], dtype=np.float64), np.asarray(f["roi"])
        authors = np.asarray(f["fmean_roi"]) - 0.7 * np.asarray(f["fmean_neuropil"])
    np.testing.assert_allclose(rows["raw"], mov[:, roi].mean(axis=1), rtol=1e-12)
    np.testing.assert_allclose(rows["authors_ring"], authors, rtol=1e-12)
    assert rows["ring"].shape == (60,)


def _cv(f1: np.ndarray, setting: str = "s") -> C.CV:
    n = f1.size
    return C.CV([f"v{i}" for i in range(n)], [setting] * n, np.full(n, 50.0), f1, f1, f1)


def test_public_record_and_holm() -> None:
    rng = np.random.default_rng(4)
    ours = 0.85 + 0.05 * rng.random(10)
    cvs = {"ours": _cv(ours), "TUnCaT": _cv(ours - 0.01 * rng.random(10)), "FISSA": _cv(ours - 0.03),
           "CNMF": _cv(ours - 0.05 * rng.random(10)), "raw": _cv(ours - 0.1), "ring": _cv(ours - 0.04),
           "suite2p": _cv(ours - 0.02 * rng.random(10))}
    record, s2p, family = T.public_record("ABO", cvs)
    assert list(record) == ["ours", "TUnCaT", "FISSA", "CNMF", "raw", "ring", "_meta"]
    stats = record["_meta"]["stats"]
    assert family is not None and family["family"] == ["TUnCaT", "FISSA", "CNMF", "Ring", "Suite2p"]
    assert s2p is not None and s2p["dataset"] == "D2" and s2p["stats"]["Suite2p"]["added"] is False
    raw = {m: stats[m]["p"] for m in ("TUnCaT", "FISSA", "CNMF")}
    p_s2p, p_ring = s2p["stats"]["Suite2p"]["p"], stats["Ring"]["p"]
    three, four = metrics.holm(raw), metrics.holm({**raw, "Suite2p": p_s2p})
    five = metrics.holm({**raw, "Suite2p": p_s2p, "Ring": p_ring})
    for m in raw:
        assert (stats[m]["p_holm_declared"], stats[m]["p_holm_suite2p"], stats[m]["p_holm"]) == \
            (three[m], four[m], five[m])
    assert stats["Ring"]["p_holm"] == five["Ring"] and s2p["stats"]["Suite2p"]["p_holm"] == five["Suite2p"]
    assert stats["FISSA"]["wins"] == 10 and stats["FISSA"]["mean_diff"] == pytest.approx(0.03)
    rows = T.holm_rows("D2", "F1", family)
    assert [r["comparison"] for r in rows] == ["TUnCaT", "FISSA", "CNMF", "Suite2p", "Ring"]
    stats["TUnCaT"]["p_holm_declared"] = 0.5                         # a stored value that is not Holm's
    with pytest.raises(ValueError, match="not Holm"):
        T.amend(stats, dict(s2p["stats"]["Suite2p"]), ring_key="Ring")
    assert T.added_test(np.array([1.0, 2.0]), np.array([2.0, 1.0]), "lower")["wins"] == 1
    assert "Suite2p" in T.public_markdown("ABO", record, s2p)


def test_public_grids_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    arch = tmp_path / "archive"
    m = _masks()
    for v in ("v1", "v2", "v3"):
        (arch / "data/ABO/GT Masks").mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(D.masks_path(arch, "ABO", v), {"FinalMasks": m.transpose(2, 1, 0).astype(np.uint8)})
        with h5py.File(D.movie_path(arch, "ABO", v), "w") as f:
            f["mov"] = rng.normal(100, 5, (30, 40, 30)).astype(np.float32)
        _peers(D.released_root(arch, "ABO"), v, 3, 30, rng.random())
        (tmp_path / "ours/ABO/ks2_kb0.1" / v).mkdir(parents=True)
        np.savez(tmp_path / "ours/ABO/ks2_kb0.1" / v / "result.npz", **{f"dff__{v}": rng.random((3, 30))})
    grids, rings = T.public_grids("ABO", arch, lambda ds, v, a: _toy_scorer, ours=[tmp_path / "ours"],
                                  videos=["v1", "v2", "v3"], log=lambda s: None)
    assert set(grids) == {"ours", "TUnCaT", "FISSA", "CNMF", "raw", "bgsubs", "ring"}
    assert rings["v1"]["trace"].shape == (3, 30)
    for meth, gs in grids.items():
        for v, g in zip(("v1", "v2", "v3"), gs, strict=True):
            C.save_grid(tmp_path / f"grid_{meth}_{v}.npz", g)
    assert T.main(["public", "--dataset", "ABO", "--grids", str(tmp_path), "--videos", "v1", "v2", "v3",
                   "--out", str(tmp_path / "tab")]) == 0
    out = json.loads((tmp_path / "tab/cv_ABO.json").read_text(encoding="utf-8"))
    assert out["suite2p"] is None and "Ring" in out["table"]["_meta"]["stats"]


def test_settings_option_keeps_only_astrid_settings(tmp_path: Path) -> None:
    assert len(T.SHARED_SETTINGS) == len(set(T.SHARED_SETTINGS)) == 14
    traces = {"ks4_kb0.1": 1, "ks2_kb0.1": 2, "ks8_kb0.03": 3}
    assert list(T.keep_settings(traces, ["ks8_kb0.03", "ks4_kb0.1"], "x")) == ["ks4_kb0.1", "ks8_kb0.03"]  # own order
    assert T.keep_settings(traces, None, "x") is traces
    with pytest.raises(ValueError, match="ks1_kb0.3"):
        T.keep_settings(traces, ["ks1_kb0.3"], "x")
    rng = np.random.default_rng(7)
    arch = tmp_path / "archive"
    for v in ("v1", "v2"):
        (arch / "data/ABO/GT Masks").mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(D.masks_path(arch, "ABO", v), {"FinalMasks": _masks().transpose(2, 1, 0).astype(np.uint8)})
        with h5py.File(D.movie_path(arch, "ABO", v), "w") as f:
            f["mov"] = rng.normal(100, 5, (30, 40, 30)).astype(np.float32)
        _peers(D.released_root(arch, "ABO"), v, 3, 30, rng.random())
        for s in ("ks2_kb0.1", "ks4_kb0.1"):
            (tmp_path / "ours/ABO" / s / v).mkdir(parents=True)
            np.savez(tmp_path / "ours/ABO" / s / v / "result.npz", **{f"dff__{v}": rng.random((3, 30))})
    full, _ = T.public_grids("ABO", arch, lambda ds, v, a: _toy_scorer, ours=[tmp_path / "ours"], videos=["v1", "v2"],
                             ring=False, log=lambda s: None)
    kept, _ = T.public_grids("ABO", arch, lambda ds, v, a: _toy_scorer, ours=[tmp_path / "ours"], videos=["v1", "v2"],
                             ring=False, log=lambda s: None, settings=["ks4_kb0.1"])
    assert full["ours"][0].settings == ["ks2_kb0.1", "ks4_kb0.1"] and kept["ours"][0].settings == ["ks4_kb0.1"]
    assert np.array_equal(kept["ours"][0].f1[0], full["ours"][0].f1[1])
    assert all(np.array_equal(kept[m][0].f1, full[m][0].f1) for m in full if m != "ours")    # the others unchanged
    index = [{"name": f"c{c}_1", "cell": f"c{c}", "rec": "1"} for c in range(3)]
    (tmp_path / "d3").mkdir()
    for row in index:
        _export(tmp_path / "d3", row["name"], rng)
        for s in ("ks1_kb0.03", "ks2_kb0.1"):
            d = tmp_path / "d3ours/cai1" / s / row["name"]
            d.mkdir(parents=True)
            np.savez(d / "result.npz", **{f"dff__{row['name']}": rng.random((1, 60))})
    (tmp_path / "d3/index.json").write_text(json.dumps(index), encoding="utf-8")
    _, scores = T.cai1_scores(tmp_path / "d3", [tmp_path / "d3ours"], settings=["ks2_kb0.1"])
    assert list(scores["ours"]) == ["ks2_kb0.1"]


def test_cai1_record(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    index = [{"name": f"c{c}_{r}", "cell": f"c{c}", "rec": str(r)} for c in range(6) for r in (1, 2)]
    for row in index:
        _export(tmp_path, row["name"], rng)
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    peers = tmp_path / "peers"
    peers.mkdir()
    for row in index:
        np.savez(peers / f"{row['name']}.npz", **{"tuncat_a=1": rng.random(60), "fissa_a=0.1": rng.random(60),
                                                  "cnmf": rng.random(60)})
        d = tmp_path / "ours/cai1/ks1_kb0.03" / row["name"]
        d.mkdir(parents=True)
        np.savez(d / "result.npz", **{f"dff__{row['name']}": rng.random((1, 60))})
        s = tmp_path / "s2p/cai1" / row["name"] / "suite2p"
        s.mkdir(parents=True)
        np.savez(s / f"{row['name']}.npz", trace=rng.random((1, 60)).astype(np.float32))
    idx, scores = T.cai1_scores(tmp_path, [tmp_path / "ours"], peers)
    assert list(scores) == ["raw", "ring", "authors_ring", "ours", "TUnCaT", "FISSA", "CNMF"]
    record, s2p, family = T.cai1_record(idx, scores, T.cai1_suite2p_scores(tmp_path, tmp_path / "s2p", idx))
    assert family is not None and family["family"] == ["TUnCaT", "FISSA", "CNMF", "Ring", "Suite2p"]
    assert len(record["results"]["ring"]["per_recording"]) == 12 and record["stats"]["Ring"]["n"] == 6
    assert s2p is not None and s2p["n_recordings"] == 12 and len(s2p["suite2p"]["r"]) == 6
    del scores["CNMF"]["cnmf"]["c0_1"]
    with pytest.raises(ValueError, match="incomplete"):
        T.cai1_record(idx, scores)


def _cnmf_grid(grid: Path, dataset: str, video: str, n: int, t: int, values: dict[str, float],
               rows: int | None = None) -> None:
    """CNMF's grid as ``tuncat_peers`` writes it, one folder per setting (each run with its own --out)."""
    for g, p in C.CNMF_GRID:
        name = C.cnmf_setting(g, p)
        if name in values:
            d = grid / name / dataset / f"traces_CNMF_Raw_p{p}"
            d.mkdir(parents=True, exist_ok=True)
            k = n if rows is None else rows
            scipy.io.savemat(d / f"{video}.mat", {"C_gt": np.zeros((k, t)), "YrA_gt": np.full((k, t), values[name])})


def test_cnmf_grid_order_and_readers(tmp_path: Path) -> None:
    """The declared order breaks ties (the first maximum): the grid's rows come in it, and sorted names (D3's
    cell_cv) give it too. Settings without the video are left out; dropped components need cnmf_fill."""
    names = [C.cnmf_setting(g, p) for g, p in C.CNMF_GRID]
    assert names == ["gnb1_p1", "gnb1_p2", "gnb2_p1", "gnb2_p2", "gnb3_p1", "gnb3_p2"] == sorted(names)
    _cnmf_grid(tmp_path / "g", "NAOMi", "Video_0", 3, 5, {n: float(i) for i, n in enumerate(names) if n != "gnb2_p1"})
    got = D.cnmf_grid_traces(tmp_path / "g", "NAOMi", "Video_0")
    assert list(got) == [n for n in names if n != "gnb2_p1"]
    np.testing.assert_array_equal(got["gnb3_p2"], np.full((3, 5), 5.0))
    _cnmf_grid(tmp_path / "h", "NAOMi", "Video_0", 3, 5, {"gnb1_p1": 1.0}, rows=2)
    with pytest.raises(ValueError, match="cnmf_fill"):
        D.cnmf_grid_traces(tmp_path / "h", "NAOMi", "Video_0", masks=np.ones((3, 4, 4), dtype=bool))
    for name, v in (("gnb1_p2", 2.0), ("gnb3_p1", 3.0)):
        (tmp_path / "c" / name / "cai1").mkdir(parents=True)
        np.savez(tmp_path / "c" / name / "cai1" / "r1.npz", cnmf=np.full(4, v), cnmf_C=np.zeros(4))
    assert {k: float(x[0]) for k, x in D.cai1_cnmf_grid(tmp_path / "c", "r1").items()} == {"gnb1_p2": 2.0,
                                                                                         "gnb3_p1": 3.0}
    tie = [C.Grid(["gnb1_p1", "gnb2_p2"], np.array([1.0]), *(np.array([[0.5], [0.5]]),) * 3, 1, 1)] * 3
    assert C.loo_cv(["a", "b", "c"], tie).setting == ["gnb1_p1"] * 3            # a tie: the published setting
    assert T.setting_label("CNMF", "gnb3_p2") == "3, 2" and T.setting_label("CNMF", "p1") is None


def test_public_grids_with_the_cnmf_grid(tmp_path: Path) -> None:
    """--cnmf-grid: CNMF over its six settings, CNMF_fixed (the published setting, from the peers) a reference row
    without a test; the tables list CNMF's chosen settings as the paper's table of them does."""
    rng = np.random.default_rng(7)
    arch = tmp_path / "archive"
    m = _masks()
    videos = ["v1", "v2", "v3"]
    for v in videos:
        (arch / "data/ABO/GT Masks").mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(D.masks_path(arch, "ABO", v), {"FinalMasks": m.transpose(2, 1, 0).astype(np.uint8)})
        with h5py.File(D.movie_path(arch, "ABO", v), "w") as f:
            f["mov"] = rng.normal(100, 5, (30, 40, 30)).astype(np.float32)
        _peers(D.released_root(arch, "ABO"), v, 3, 30, rng.random())
        _peers(tmp_path / "peers", v, 3, 30, 0.5)
        _cnmf_grid(tmp_path / "grid", "ABO", v, 3, 30, {C.cnmf_setting(g, p): 0.1 * g + 0.01 * p
                                                       for g, p in C.CNMF_GRID})
        (tmp_path / "ours/ABO/ks2_kb0.1" / v).mkdir(parents=True)
        np.savez(tmp_path / "ours/ABO/ks2_kb0.1" / v / "result.npz", **{f"dff__{v}": rng.random((3, 30))})

    def scorer(ds: str, v: str, a: str) -> C.Scorer:
        def score(x: np.ndarray, thr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
            f = np.full(thr.size, float(x.mean()))          # the larger the trace, the better: gnb3_p2 wins
            return f, f, f, 3, 4
        return score

    grids, _ = T.public_grids("ABO", arch, scorer, ours=[tmp_path / "ours"], peers=[tmp_path / "peers"],
                              videos=videos, ring=False, log=lambda s: None, cnmf_grid=tmp_path / "grid")
    assert grids["CNMF"][0].settings == [C.cnmf_setting(g, p) for g, p in C.CNMF_GRID]
    assert grids[T.CNMF_FIXED][0].settings == ["p1"]
    record, _, _ = T.public_record("ABO", T.cv_by_method(grids, videos))
    assert record["CNMF"]["setting"] == ["gnb3_p2"] * 3 and record[T.CNMF_FIXED]["setting"] == ["p1"] * 3
    assert "CNMF" in record["_meta"]["stats"] and T.CNMF_FIXED not in record["_meta"]["stats"]
    text = T.public_markdown("ABO", record, None)
    assert "| CNMF (n_b, p) | 3, 2 (3) |" in text and "CNMF_fixed: CNMF at the published setting" in text


def test_cai1_scores_with_the_cnmf_grid(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    index = [{"name": f"c{c}_{r}", "cell": f"c{c}", "rec": str(r)} for c in range(4) for r in (1, 2)]
    peers = _mkdir(tmp_path / "peers")
    for row in index:
        _export(tmp_path, row["name"], rng)
        np.savez(peers / f"{row['name']}.npz", cnmf=rng.random(60))
        for g, p in C.CNMF_GRID:
            d = _mkdir(tmp_path / "grid" / C.cnmf_setting(g, p) / "cai1")
            np.savez(d / f"{row['name']}.npz", cnmf=rng.random(60))
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    idx, scores = T.cai1_scores(tmp_path, None, tmp_path / "peers", tmp_path / "grid")
    assert sorted(scores["CNMF"]) == [C.cnmf_setting(g, p) for g, p in C.CNMF_GRID]
    assert list(scores[T.CNMF_FIXED]) == ["cnmf"]
    record, _, _ = T.cai1_record(idx, scores)
    assert record["results"]["CNMF"]["settings"] == [C.cnmf_setting(g, p) for g, p in C.CNMF_GRID]
    assert "CNMF (n_b, p)" in T.cai1_markdown(record, None)


def test_cai1_equal_baseline(tmp_path: Path) -> None:
    """``--baseline equal``: every trace but ASTRID's is scored after its own running baseline is removed by
    ASTRID's rule; ASTRID's r are the reported ones."""
    rng = np.random.default_rng(9)
    t, fps = 600, 5.0                                         # 120 s: the rule's 60-s window moves
    index = [{"name": f"c{c}_{r}", "cell": f"c{c}", "rec": str(r)} for c in range(4) for r in (1, 2)]
    drift = np.linspace(0.0, 3.0, t)                          # slow, as a baseline drifts
    peers = _mkdir(tmp_path / "peers")
    for row in index:
        name = row["name"]
        _export(tmp_path, name, rng, t=t)
        with h5py.File(tmp_path / f"{name}.h5", "a") as f:
            f.attrs["fps"] = fps
        np.savez(peers / f"{name}.npz", **{"tuncat_a=1": rng.random(t) + drift, "fissa_a=0.1": rng.random(t),
                                           "cnmf": rng.random(t)})
        np.savez(_mkdir(tmp_path / "ours/cai1/ks1_kb0.03" / name) / "result.npz",
                 **{f"dff__{name}": rng.random((1, t))})
        np.savez(_mkdir(tmp_path / "s2p/cai1" / name / "suite2p") / f"{name}.npz", trace=rng.random((1, t)))
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    _, rep = T.cai1_scores(tmp_path, [tmp_path / "ours"], peers)
    idx, eq = T.cai1_scores(tmp_path, [tmp_path / "ours"], peers, baseline="equal")
    assert eq["ours"] == rep["ours"]
    for row in index:
        name = row["name"]
        pred, ok = D.read_cai1_scored(tmp_path, name)
        with np.load(peers / f"{name}.npz") as z:
            x = np.asarray(z["tuncat_a=1"], dtype=np.float64)
        assert eq["TUnCaT"]["a=1"][name] == D.corr(x - astrid_baseline(x[None], fps)[0], pred, ok)
        assert rep["TUnCaT"]["a=1"][name] == D.corr(x, pred, ok)
        raw = R.cai1_reference_rows(tmp_path / f"{name}.h5")["raw"]
        assert eq["raw"]["raw"][name] == D.corr(D.minus_baseline(raw, fps), pred, ok)
    s2p_eq = T.cai1_suite2p_scores(tmp_path, tmp_path / "s2p", idx, "equal")
    _, _, family = T.cai1_record(idx, eq, s2p_eq)
    assert family is not None and family["family"] == ["TUnCaT", "FISSA", "CNMF", "Ring", "Suite2p"]
    assert T.main(["cai1", "--export", str(tmp_path), "--ours", str(tmp_path / "ours"), "--peers", str(peers),
                   "--suite2p", str(tmp_path / "s2p"), "--baseline", "equal", "--out", str(tmp_path / "tab")]) == 0
    out = json.loads((tmp_path / "tab/cv_cai1_equal_baseline.json").read_text(encoding="utf-8"))
    assert out["table"]["baseline"] == T.EQUAL_BASELINE and out["suite2p"]["suite2p"]["per_recording"] == s2p_eq
    assert "Equal baseline:" in (tmp_path / "tab/cv_cai1_equal_baseline.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="baseline must be"):
        T.cai1_scores(tmp_path, None, peers, baseline="detrended")
    with pytest.raises(ValueError, match="non-finite"):
        D.minus_baseline(np.array([1.0, np.nan, 2.0]), fps)


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
