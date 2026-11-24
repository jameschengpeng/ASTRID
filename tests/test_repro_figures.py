"""The figure code on small synthetic data: the shared plot of the scores, Figure 5's panels (D3) from a table,
an export and traces made here, and one row of the supplementary figure of D2, D4 and D5 with a toy scoring in
place of TUnCaT's evaluation (which is GPL-2 and tested in its own folder)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import scipy.io

matplotlib = pytest.importorskip("matplotlib")         # the extra "reproduce" (or "dev")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cv as C  # noqa: E402
from astrid_repro import public_data as D  # noqa: E402
from astrid_repro import ring as R  # noqa: E402
from astrid_repro import tables_public as T  # noqa: E402
from astrid_repro.figures import cai1, common, public  # noqa: E402
from astrid_repro.figures.__main__ import main as figures_main  # noqa: E402


def test_common() -> None:
    assert [common.stars(p) for p in (0.2, 0.04, 0.009, 0.0004)] == ["n.s.", "*", "**", "***"]
    tests = {"a": {"p": 0.01, "p_holm": 0.03}, "b": {"p": 0.04, "p_holm": 0.06}, "c": {"p": 0.03, "p_holm": 0.06}}
    common.check_holm(tests, "toy")
    tests["b"]["p_holm"] = 0.04
    with pytest.raises(ValueError, match="Holm"):
        common.check_holm(tests, "toy")
    fig, ax = plt.subplots()
    values = {"ours": [0.9, 0.8, 0.85], "TUnCaT": [0.7, 0.75, 0.8]}
    shown = common.paired_dots(ax, values, ("ours", "TUnCaT"), {"TUnCaT": "*"}, "F1", "toy, 3 videos")
    assert shown["means"]["ours"] == pytest.approx(0.85) and shown["marks"] == {"TUnCaT": "*"}
    assert [t.get_text() for t in ax.texts] == ["*"] and [t.get_text() for t in ax.get_xticklabels()] == \
        ["ASTRID", "TUnCaT"]
    assert len(ax.collections) == 2 and len(ax.get_lines()) == 3 + 2           # 3 units, 2 mean bars
    plt.close(fig)
    assert common.callable_from("json:dumps") is json.dumps and common.callable_from(None) is None
    assert common.jsonable({"a": np.arange(2), "b": np.float64(1.5), "c": (np.int64(3),)}) == \
        {"a": [0, 1], "b": 1.5, "c": [3]}


# ---- D3 ----

def _cai1_export(folder: Path, name: str, indicator: str, rng: np.random.Generator, t: int = 90) -> None:
    roi = np.zeros((20, 20), dtype=bool)
    roi[8:12, 8:12] = True
    spikes = np.sort(rng.uniform(5, 85, 6))
    tf = np.arange(t, dtype=float)                                    # 1 frame per second
    with h5py.File(folder / f"{name}.h5", "w") as f:
        f["mov"] = rng.integers(0, 100, (t, 20, 20)).astype(np.int16)
        f["roi"] = roi
        f["valid"] = (tf >= 3) & (tf <= 85)
        f["frame_present"] = np.ones(t, dtype=bool)
        f["t_frame"] = tf
        f["spike_times"] = spikes
        f["predicted"] = D.predicted_calcium(spikes, tf, indicator)
        f["fmean_roi"] = rng.random(t)
        f["fmean_neuropil"] = rng.random(t)
        f.attrs.update(ephys_start_s=2.0, ephys_end_s=86.0, indicator=indicator, fps=1.0)


def _cai1(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    index = [{"name": f"2012010{c}_cell{c}_00{r}", "cell": f"2012010{c}_cell{c}", "rec": f"00{r}",
              "indicator": "GCaMP6f" if c < 3 else "GCaMP6s"} for c in range(5) for r in (1, 2)]
    for row in index:
        _cai1_export(tmp_path, row["name"], row["indicator"], rng)
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    peers, ours, s2p = tmp_path / "peers", tmp_path / "ours", tmp_path / "s2p"
    peers.mkdir()
    for row in index:
        pred, _ = D.read_cai1_scored(tmp_path, row["name"])
        noisy = {k: pred + w * rng.normal(size=pred.size) for k, w in
                 (("tuncat_a=1", 0.3), ("fissa_a=0.1", 0.5), ("cnmf", 0.8), ("ours", 0.1), ("s2p", 0.4))}
        np.savez(peers / f"{row['name']}.npz", **{k: noisy[k] for k in ("tuncat_a=1", "fissa_a=0.1", "cnmf")})
        for setting, extra in (("ks1_kb0.03", 0.0), ("ks2_kb0.1", 0.2)):
            d = ours / "cai1" / setting / row["name"]
            d.mkdir(parents=True)
            np.savez(d / "result.npz", **{f"dff__{row['name']}": (noisy["ours"] + extra * rng.normal(size=90))[None]})
        s = s2p / "cai1" / row["name"] / "suite2p"
        s.mkdir(parents=True)
        np.savez(s / f"{row['name']}.npz", trace=noisy["s2p"][None].astype(np.float32))
    idx, scores = T.cai1_scores(tmp_path, [ours], peers)
    record, s2p_entry, family = T.cai1_record(idx, scores, T.cai1_suite2p_scores(tmp_path, s2p, idx))
    table = tmp_path / "cv_cai1.json"
    table.write_text(json.dumps({"table": record, "suite2p": s2p_entry, "holm": family}), encoding="utf-8")
    return {"table": table, "export": tmp_path, "ours": [ours], "peers": peers, "suite2p": s2p, "index": index}


def test_cai1_panels(tmp_path: Path) -> None:
    p = _cai1(tmp_path)
    made = cai1.panels(p["table"], export=p["export"], ours=p["ours"], peers=p["peers"], suite2p=p["suite2p"])
    assert [stem for stem, _, _ in made] == ["fig5b_D3_scores", "fig5a_D3_image", "fig5a_D3_traces"]
    scores, traces = made[0][2], made[2][2]
    table = cai1.read_table(p["table"])
    margin = np.asarray(table["values"]["ours"]) - np.max([table["values"][m] for m in cai1.ORDER[1:]], axis=0)
    assert traces["cell"] == table["cells"][int(np.argmax(margin))]
    assert traces["ASTRID's r above the best other method's (cell)"] == pytest.approx(margin.max())
    s, e = traces["window (frames)"]
    assert e - s == 30 and 2.0 <= traces["window (s)"][0] and traces["window (s)"][1] <= 86.0
    assert set(scores["indicators"].values()) == {"GCaMP6f", "GCaMP6s"} and len(scores["cells"]) == 5
    assert [r["method"] for r in traces["rows"]] == list(cai1.ORDER)
    for _, fig, _ in made:
        plt.close(fig)
    out = tmp_path / "figs"
    assert figures_main(["cai1", "--table", str(p["table"]), "--export", str(p["export"]), "--ours",
                         str(p["ours"][0]), "--peers", str(p["peers"]), "--suite2p", str(p["suite2p"]),
                         "--out", str(out), "--formats", "png"]) == 0
    assert sorted(f.name for f in out.iterdir()) == sorted(
        f"{s}.{x}" for s in ("fig5b_D3_scores", "fig5a_D3_image", "fig5a_D3_traces") for x in ("png", "json"))
    bad = json.loads(p["table"].read_text(encoding="utf-8"))
    bad["table"]["stats"]["FISSA"]["p_holm"] = 0.5                   # not Holm's correction of the raw p
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Holm"):
        cai1.read_table(tmp_path / "bad.json")


def test_cai1_equal_baseline(tmp_path: Path) -> None:
    """With ``tables_public cai1 --baseline equal``'s table, the figure scores and draws every trace but
    ASTRID's after its own running baseline is removed, so its recomputed correlations are that table's; CNMF
    over its cross-validation grid, as the table took it."""
    p = _cai1(tmp_path)
    rng, grid = np.random.default_rng(11), tmp_path / "grid"
    for row in p["index"]:                                          # two of CNMF's settings, as cai1_peers writes
        pred, _ = D.read_cai1_scored(p["export"], row["name"])
        for setting, w in (("gnb1_p1", 0.6), ("gnb1_p2", 0.3)):
            (grid / setting / "cai1").mkdir(parents=True, exist_ok=True)
            np.savez(grid / setting / "cai1" / f"{row['name']}.npz", cnmf=pred + w * rng.normal(size=pred.size))
    idx, scores = T.cai1_scores(p["export"], p["ours"], p["peers"], grid, baseline="equal")
    record, s2p_entry, family = T.cai1_record(
        idx, scores, T.cai1_suite2p_scores(p["export"], p["suite2p"], idx, "equal"))
    record["baseline"] = T.EQUAL_BASELINE                            # as tables_public writes it
    table = tmp_path / "cv_cai1_equal_baseline.json"
    table.write_text(json.dumps({"table": record, "suite2p": s2p_entry, "holm": family}), encoding="utf-8")
    eq, rep = cai1.read_table(table), cai1.read_table(p["table"])
    assert eq["baseline"] == "equal" and rep["baseline"] == "reported"
    assert set(eq["settings"]["CNMF"]) <= {"gnb1_p1", "gnb1_p2"}
    made = cai1.panels(table, export=p["export"], ours=p["ours"], peers=p["peers"], suite2p=p["suite2p"],
                       cnmf_grid=grid)
    scores_rec, traces = made[0][2], made[2][2]
    assert scores_rec["scoring"] == T.EQUAL_BASELINE and "minus its own running baseline" in traces["traces"]
    margin = np.asarray(eq["values"]["ours"]) - np.max([eq["values"][m] for m in cai1.ORDER[1:]], axis=0)
    assert traces["cell"] == eq["cells"][int(np.argmax(margin))]
    for _, fig, _ in made:
        plt.close(fig)
    # the traces drawn are the scored ones: every peer's minus its running baseline, ASTRID's as it is
    ex = cai1.example(eq, export=p["export"], ours=p["ours"], peers=p["peers"], suite2p=p["suite2p"],
                      cnmf_grid=grid)
    fps = D.read_cai1_fps(p["export"], ex["name"])
    for m in cai1.ORDER:
        x = cai1._trace(m, ex["name"], ex["settings"][m], Path(p["export"]), p["ours"], p["peers"], p["suite2p"],
                        cnmf_grid=grid)
        shown = x if m == "ours" else D.minus_baseline(x, fps)
        np.testing.assert_allclose(ex["z"][m], cai1._z(shown, ex["ok"]))
    # a note that is not tables_public's is refused; an equal table without its note does not reproduce
    bad = json.loads(table.read_text(encoding="utf-8"))
    bad["table"]["baseline"] = "another rule"
    (tmp_path / "bad_note.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline note"):
        cai1.read_table(tmp_path / "bad_note.json")
    del bad["table"]["baseline"]
    (tmp_path / "no_note.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the table"):
        cai1.example(cai1.read_table(tmp_path / "no_note.json"), export=p["export"], ours=p["ours"],
                     peers=p["peers"], suite2p=p["suite2p"], cnmf_grid=grid)
    out = tmp_path / "figs_equal"
    assert figures_main(["cai1", "--table", str(table), "--export", str(p["export"]), "--ours",
                         str(p["ours"][0]), "--peers", str(p["peers"]), "--cnmf-grid", str(grid), "--suite2p",
                         str(p["suite2p"]), "--out", str(out), "--formats", "png"]) == 0
    assert json.loads((out / "fig5b_D3_scores.json").read_text(encoding="utf-8"))["scoring"] == T.EQUAL_BASELINE


# ---- D2, D4, D5, with a toy scoring ----

def _runs(x: np.ndarray, thr: float) -> np.ndarray:
    d = np.diff(np.concatenate([[0], (x > thr).astype(int), [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1], axis=1)


class ToyView:
    """A toy in place of TUnCaT's evaluation: the trace minus its median over its SD, detections are the runs
    above the threshold, a detection matches a labelled transient it overlaps."""

    def __init__(self, labels: list[np.ndarray | None]) -> None:
        self.labels = labels

    def __call__(self, x: np.ndarray, threshold: float) -> dict[str, Any]:
        z = (x - np.median(x, axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-12)
        out: dict[str, Any] = {"z": z, "truth": [], "detections": [], "truth_matched": [],
                               "detection_matched": [], "roi_f1": np.full(len(self.labels), np.nan)}
        hits = [0, 0, 0, 0]                                       # matched truth, truth, matched detections, detections
        for i, lab in enumerate(self.labels):
            det = _runs(z[i], threshold)
            out["detections"].append(det)
            if lab is None:
                out["truth"].append(None)
                out["truth_matched"].append(np.zeros(0, dtype=bool))
                out["detection_matched"].append(np.zeros(len(det), dtype=bool))
                continue
            gt = lab[lab[:, 2] == 1, :2] - 1
            over = (np.minimum(gt[:, None, 1], det[None, :, 1]) >= np.maximum(gt[:, None, 0], det[None, :, 0]))
            g, d = over.any(axis=1), over.any(axis=0)
            out["truth"].append(gt)
            out["truth_matched"].append(g)
            out["detection_matched"].append(d)
            rec, pre = (g.mean() if g.size else 0.0), (d.mean() if d.size else 0.0)
            out["roi_f1"][i] = 2 * rec * pre / (rec + pre) if rec and pre else 0.0
            hits = [hits[0] + g.sum(), hits[1] + g.size, hits[2] + d.sum(), hits[3] + d.size]
        rec, pre = (hits[0] / hits[1] if hits[1] else 0.0), (hits[2] / hits[3] if hits[3] else 0.0)
        out["video_f1"] = float(2 * rec * pre / (rec + pre)) if rec and pre else 0.0
        return out


def _public(tmp_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(8)
    arch, ours, s2p = tmp_path / "archive", tmp_path / "ours", tmp_path / "s2p"
    t, videos = 4000, ("501484643", "501574836", "501729039")
    masks = np.zeros((3, 40, 30), dtype=bool)
    masks[0, 5:9, 5:9] = masks[1, 20:25, 10:14] = masks[2, 30:36, 18:24] = True
    starts = np.arange(200, 3800, 360)                               # 10 transients per ROI
    shape = np.exp(-np.arange(12) / 3.0)
    traces: dict[str, dict[str, np.ndarray]] = {}
    for k, v in enumerate(videos):
        (arch / "data/ABO/GT Masks").mkdir(parents=True, exist_ok=True)
        (arch / "data/ABO/GT transients").mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(D.masks_path(arch, "ABO", v), {"FinalMasks": masks.transpose(2, 1, 0).astype(np.uint8)})
        labels = np.empty(3, dtype=object)
        for i in range(3):
            labels[i] = np.stack([starts + 1, starts + 12, np.ones_like(starts)], axis=1)
        scipy.io.savemat(D.labels_path(arch, "ABO", v), {"output": labels})
        with h5py.File(D.movie_path(arch, "ABO", v), "w") as f:
            f["mov"] = rng.normal(100, 5, (t, 40, 30)).astype(np.float32)
        clean = np.zeros((3, t))
        for s in starts:
            clean[:, s:s + 12] += shape
        x = {"ours": clean + (0.02 + 0.01 * k) * rng.normal(size=(3, t)),
             "a=1": clean * 0.3 + 0.05 * rng.normal(size=(3, t)), "a=0.01": 0.05 * rng.normal(size=(3, t)),
             "p1": clean * 0.2 + 0.08 * rng.normal(size=(3, t)),
             "suite2p": clean * 0.25 + 0.06 * rng.normal(size=(3, t))}
        x["ours"][1, starts[3]:starts[3] + 12] = 0.0                     # ROI 2 misses one transient
        traces[v] = x
        (ours / "ABO/ks2_kb0.1" / v).mkdir(parents=True)
        np.savez(ours / "ABO/ks2_kb0.1" / v / "result.npz", **{f"dff__{v}": x["ours"]})
        rel = D.released_root(arch, "ABO")
        for folder, key, arr in (("traces_ours_Raw/alpha= 1.000", "traces_nmfdemix", x["a=1"].T),
                                 ("traces_FISSA_Raw/alpha= 0.010", "unmixed_traces", x["a=0.01"])):
            (rel / folder).mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(rel / folder / f"{v}.mat", {key: arr})
        (rel / "traces_CNMF_Raw_p1").mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(rel / "traces_CNMF_Raw_p1" / f"{v}.mat", {"C_gt": x["p1"], "YrA_gt": np.zeros((3, t))})
        (s2p / "ABO" / v / "suite2p").mkdir(parents=True)
        np.savez(D.suite2p_path(s2p, "ABO", v), trace=x["suite2p"])
    thr = 3.0
    cvs = {}
    for m, key in (("ours", "ours"), ("TUnCaT", "a=1"), ("FISSA", "a=0.01"), ("CNMF", "p1"), ("suite2p", "suite2p"),
                   ("ring", None)):
        f1 = []
        for v in videos:
            labels = D.read_labels(D.labels_path(arch, "ABO", v))
            x = R.video_ring(D.movie_path(arch, "ABO", v), masks, D.UM_PER_PX["ABO"])["trace"] if key is None \
                else traces[v][key]
            f1.append(ToyView(labels)(x, thr)["video_f1"])
        setting = {"ours": "ks2_kb0.1", "TUnCaT": "a=1", "FISSA": "a=0.01", "CNMF": "p1", "ring": "ring",
                   "suite2p": "suite2p"}[m]
        f = np.asarray(f1)
        cvs[m] = C.CV(list(videos), [setting] * 3, np.full(3, thr), f, f, f)
    record, s2p_entry, family = T.public_record("ABO", cvs)
    table = tmp_path / "cv_ABO.json"
    table.write_text(json.dumps({"table": record, "suite2p": s2p_entry, "holm": family}), encoding="utf-8")
    return {"table": table, "archive": arch, "ours": [ours], "suite2p": s2p, "videos": videos}


def toy_view(dataset: str, video: str, archive: str) -> ToyView:
    return ToyView(D.read_labels(D.labels_path(archive, dataset, video)))


def test_public_panels(tmp_path: Path) -> None:
    p = _public(tmp_path)
    made = public.panels("ABO", p["table"], archive=p["archive"], ours=p["ours"], peers=None, suite2p=p["suite2p"],
                         evaluator=toy_view)
    assert [stem for stem, _, _ in made] == ["figS6a_D2_scores", "figS6b_D2_image", "figS6b_D2_traces"]
    scores, image, traces = (rec for _, _, rec in made)
    table = public.read_table(p["table"], "ABO")
    f1 = np.asarray(table["values"]["ours"])
    assert traces["video"] == p["videos"][int(np.argmin(np.abs(f1 - np.median(f1))))]
    rows = {r["method"]: r for r in traces["rows"]}
    assert traces["example ROI"] in (1, 3)                          # ROI 2 has a transient ASTRID misses
    assert rows["ours"]["roi_f1"] == 1.0 and traces["transients of the ROI"] == 10
    best = max(public.ORDER[1:], key=lambda m: rows[m]["roi_f1"])
    assert traces["best other method"] == best
    assert traces["ASTRID's F1 above the best other method's"] == pytest.approx(1.0 - rows[best]["roi_f1"])
    s, e = traces["window (frames)"]
    assert e - s == 3600 and traces["whole recording"] is False       # 120 s at 30 frames per second
    assert image["scale bar (um)"] == 20 and image["ROIs"] == 3        # 30 px at 0.785 um: under 120 um
    assert scores["videos"] == list(p["videos"]) and set(scores["marks"]) == set(public.ORDER[1:])
    for _, fig, _ in made:
        plt.close(fig)
    bad = json.loads(p["table"].read_text(encoding="utf-8"))
    for entry in (bad["table"]["ours"], bad["suite2p"]["ours"]):    # no longer the scoring's F1 of any video
        entry["f1"] = [f - 0.25 for f in entry["f1"]]
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="in the table"):
        public.example("ABO", public.read_table(tmp_path / "bad.json", "ABO"), archive=p["archive"],
                       ours=p["ours"], peers=None, suite2p=p["suite2p"], evaluator=toy_view)


def test_public_example_with_the_cnmf_grid(tmp_path: Path) -> None:
    """CNMF's setting chosen on its cross-validation grid: its traces come from the grid (``cnmf_grid``, as
    tables_public reads it), not from the fixed setting's folder."""
    p = _public(tmp_path)
    tab = json.loads(p["table"].read_text(encoding="utf-8"))
    tab["table"]["CNMF"]["setting"] = ["gnb2_p1"] * len(p["videos"])
    p["table"].write_text(json.dumps(tab), encoding="utf-8")
    grid, rel = tmp_path / "grid", D.released_root(p["archive"], "ABO")
    folder = grid / "gnb2_p1" / "ABO" / "traces_CNMF_Raw_p1"
    folder.mkdir(parents=True)
    for v in p["videos"]:
        m = scipy.io.loadmat(rel / "traces_CNMF_Raw_p1" / f"{v}.mat")
        scipy.io.savemat(folder / f"{v}.mat", {"C_gt": m["C_gt"], "YrA_gt": m["YrA_gt"]})
    kw: dict[str, Any] = {"archive": p["archive"], "ours": p["ours"], "peers": None, "suite2p": p["suite2p"],
                          "evaluator": toy_view}
    table = public.read_table(p["table"], "ABO")
    with pytest.raises(KeyError):
        public.example("ABO", table, **kw)                           # not a setting of the fixed folder
    ex = public.example("ABO", table, cnmf_grid=grid, **kw)
    assert ex["methods"]["CNMF"]["setting"] == "gnb2_p1"
