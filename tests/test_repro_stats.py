"""The statistics and tables of D1, D6, D7 and the ablations (``reproduce/astrid_repro/stats.py``) on small
synthetic score tables: each unit's median, the paired tests, Holm's families, the failure rules and the
commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import metrics, stats  # noqa: E402

METHODS = ("demix", "tuncat", "tuncat_astro", "fissa", "cnmf", "cnmf_astro", "ring", "suite2p", "roi")
WORSE = {"demix": 0.0, "tuncat": 0.02, "tuncat_astro": 0.1, "fissa": 0.2, "cnmf": 0.3, "cnmf_astro": 0.25,
         "ring": 0.15, "suite2p": 0.12, "roi": 0.5}


def d1_tables(n_loc: int = 8, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Pooled per-case scores of every method: ``n_loc`` locations (three mice), two recordings each, three
    injection sites at x1 and x3 and two leakage targets; every comparator is worse than ASTRID on average."""
    rng = np.random.default_rng(seed)
    inj, leak = [], []
    for i in range(n_loc):
        loc = f"m{i % 3}_loc{i:02d}"
        for rec in ("r1", "r2"):
            for j in range(3):
                site = "astro" if j % 2 == 0 else "control"
                for scale in (1.0, 3.0):
                    base = rng.uniform(0.3, 0.8)
                    for m in METHODS:
                        err = base + WORSE[m] + rng.normal(0, 0.05)
                        inj.append({"case": f"inj{j:02d}-{site}-x{scale:g}", "site": site, "scale": scale,
                                    "method": m, "corr": 0.9, "gain": 1 + WORSE[m] + rng.normal(0, 0.05),
                                    "err_raw": 2 * err, "err_2hz": 1.5 * err, "err_05hz": err,
                                    "location": loc, "recording": rec})
            for target in ("M-001", "M-002"):
                for scale in (1.0, 3.0):
                    base = rng.uniform(0.2, 0.5)
                    for m in METHODS:
                        lk = 1.0 if m == "roi" else base + WORSE[m] + rng.normal(0, 0.05)
                        leak.append({"target": target, "scale": scale, "method": m, "leak": lk, "leak_05hz": lk,
                                     "kept_corr": 0.95 - WORSE[m] + rng.normal(0, 0.02), "location": loc,
                                     "recording": rec})
    return {"injection": pd.DataFrame(inj), "leakage": pd.DataFrame(leak)}


def location_medians(t: pd.DataFrame, value: str, scale: float, method: str) -> pd.Series:
    g = t[(t.scale == scale) & (t.method == method)]
    return g.groupby("location")[value].median()


def test_d1_units_tests_and_holm_with_suite2p() -> None:
    tables = d1_tables()
    res = stats.d1(tables)
    rec, s2p, sec = res["d1_stats"], res["suite2p"], res["secondary"]
    assert list(rec) == ["error x1", "error x3", "leakage x1", "leakage x3"]
    r = rec["error x1"]
    assert r["locations"] == sorted({f"m{i % 3}_loc{i:02d}" for i in range(8)})
    assert r["mouse"] == [s.split("_")[0] for s in r["locations"]]
    assert r["recordings"] == [2] * 8 and r["cases"] == [6] * 8         # 3 sites x 2 recordings at x1
    ours = location_medians(tables["injection"], "err_05hz", 1.0, "demix")
    np.testing.assert_array_equal(r["scores"]["demix"], ours.to_numpy())
    raw = {}
    for m in stats.D1_DECLARED:
        d = ours.to_numpy() - location_medians(tables["injection"], "err_05hz", 1.0, m).to_numpy()
        s = r["stats"][m]
        assert s["p"] == wilcoxon(d).pvalue and s["mean_diff"] == d.mean() and s["wins"] == int((d < 0).sum())
        raw[m] = s["p"]
    raw["suite2p"] = s2p["error x1"]["stats"]["suite2p"]["p"]
    seven, six = metrics.holm(raw), metrics.holm({k: v for k, v in raw.items() if k != "suite2p"})
    for m in stats.D1_DECLARED:
        assert r["stats"][m]["p_holm"] == seven[m] and r["stats"][m]["p_holm_declared"] == six[m]
        assert r["stats"][m]["p_holm"] >= r["stats"][m]["p_holm_declared"]          # a comparison more
    assert s2p["error x1"]["stats"]["suite2p"]["p_holm"] == seven["suite2p"]
    assert r["holm"]["family"] == [*stats.D1_DECLARED, "suite2p"]
    assert "suite2p" not in r["scores"] and r["scores"]["roi"]                    # roi: a reference, no test
    # Suite2p on every measure, the secondary ones in families of seven
    assert list(s2p) == [m[0] for m in stats.D1_MEASURES]
    assert list(sec) == [m[0] for m in stats.D1_SECONDARY]
    g = sec["gain deviation x1"]
    gd = tables["injection"].assign(gain_dev=(tables["injection"].gain - 1).abs())
    d = (location_medians(gd, "gain_dev", 1.0, "demix") - location_medians(gd, "gain_dev", 1.0, "cnmf")).to_numpy()
    assert g["stats"]["cnmf"]["p"] == wilcoxon(d).pvalue and g["stats"]["cnmf"]["wins"] == int((d < 0).sum())
    k = sec["kept correlation x1"]["stats"]["fissa"]                        # higher is better
    assert k["wins"] == k["n"] and k["mean_diff"] > 0
    assert sec["error x1, astrocyte sites"]["stats"]["suite2p"]["p_holm"] == \
        s2p["error x1, astrocyte sites"]["stats"]["suite2p"]["p_holm"]
    assert all(len(sec[n]["family"]) == 7 for n in sec)
    text = stats.d1_markdown(res)
    assert "| Suite2p |" in text and "TUnCaT + terr." in text


def test_d1_refuses_a_location_without_a_method_and_suite2p_without_astrids_cases() -> None:
    tables = d1_tables()
    t = tables["injection"]
    gap = {**tables, "injection": t[~((t.location == "m0_loc00") & (t.method == "fissa") & (t.scale == 1.0))]}
    with pytest.raises(SystemExit, match="a location lacks a method"):
        stats.d1(gap)
    t = tables["leakage"]
    one = t.index[(t.method == "suite2p")][0]
    with pytest.raises(SystemExit, match="Suite2p's cases are not ASTRID's"):
        stats.d1({**tables, "leakage": t.drop(index=one)})


def test_one_wilcoxon_with_its_method() -> None:
    d = np.array([0.0, 0.0, 0.3, -0.1, 0.2, 0.25, 0.4, 0.05, -0.02, 0.15, 0.3, 0.12, 0.07, 0.22, 0.31, 0.09])
    assert metrics.wilcoxon_p(d) == wilcoxon(d).pvalue                      # SciPy's default: zeros, n > 13
    assert metrics.wilcoxon_p(d, "exact") == wilcoxon(d, method="exact").pvalue
    assert metrics.wilcoxon_p(d) != metrics.wilcoxon_p(d, "exact")          # the two differ with zeros
    assert metrics.wilcoxon_p(np.zeros(5), "exact") == 1.0
    with pytest.raises(ValueError, match="method"):
        metrics.wilcoxon_p(d, "asymptotic")


def test_ablations_default_test_holm_over_three_and_the_same_cases() -> None:
    full = d1_tables(seed=1)
    ours = {k: v[v.method == "demix"] for k, v in full.items()}
    worse = {k: v.assign(err_05hz=v.get("err_05hz", 0) + 0.1, leak=v.get("leak", 0) + 0.1) for k, v in ours.items()}
    low = ours["injection"].copy()
    low.loc[low.index[:5], "gain"] = 0.1                                   # five x1 or x3 gains below 0.5
    variants = {"I": worse, "II": ours, "III": {**ours, "injection": low}}
    rec = stats.ablations(full, variants)
    s = rec["error x1"]["stats"]
    assert s["II"]["p"] == 1.0 and s["II"]["n_nonzero"] == 0 and s["II"]["wins"] == 0  # identical: no test
    d = np.asarray(rec["error x1"]["scores"]["full"]) - np.asarray(rec["error x1"]["scores"]["I"])
    assert s["I"]["p"] == wilcoxon(d).pvalue and s["I"]["wins"] == s["I"]["n"] == 8
    assert {v: s[v]["p_holm"] for v in stats.VARIANTS} == metrics.holm({v: s[v]["p"] for v in stats.VARIANTS})
    assert rec["kept correlation x1"]["better"] == "higher"
    n3 = int((low.scale == 3).sum())
    assert rec["_descriptive"]["n_cases"] == n3
    assert rec["_descriptive"]["counts"]["III"] == int((low[low.scale == 3].gain < 0.5).sum())
    assert "I, no astrocyte components" in stats.ablation_markdown(rec)
    with pytest.raises(SystemExit, match="cases differ"):
        stats.ablations(full, {**variants, "III": {**ours, "injection": low.iloc[1:]}})
    with pytest.raises(ValueError):
        stats.ablations(full, {"I": worse, "II": ours})


def d6_recording(root: Path, name: str, location: str, T: int = 240) -> Path:
    """A D6 recording folder (``stats.d6_scores``' layout) with two cases: ASTRID removes the astrocyte
    channel, the ring half of it, TUnCaT none (it is the raw ROI); CNMF's trace is constant on M (a failure)
    and Suite2p's is missing for the second case."""
    rng = np.random.default_rng(len(name))
    run = root / name
    for movie in ("M", "R"):
        for sub in ("compare", "ours", "peers", "suite2p"):
            (run / movie / sub).mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"location": location, "recording": name}))
    index = [{"spec": f"M-00{k}", "label": k, "territory": cls, "territory_share": share, "neighbours": 1,
              "atoms": 1} for k, cls, share in ((1, "in", 0.6), (2, "out", 0.0))]
    (run / "M" / "compare" / "index.json").write_text(json.dumps(index))
    t = np.arange(T)
    for c in index:
        neuron = 50 + rng.normal(size=T)
        astro = 3 * np.sin(2 * np.pi * t / 80)                              # slow, inside both bands
        roi = {"M": neuron + astro, "R": neuron}
        for movie in ("M", "R"):
            np.savez(run / movie / "compare" / f"{c['spec']}.npz", roi_mean=roi[movie])
            ring = roi[movie] - (0.5 * astro if movie == "M" else 0)
            if movie == "R":                                                # the per-recording layout
                np.savez(run / movie / "ours" / f"{c['spec']}.npz", delivered__n=1, delivered__r0=neuron - 50,
                         ring_trace__n=1, ring_trace__r0=ring)
            else:
                np.savez(run / movie / "ours" / f"{c['spec']}.npz", delivered=(neuron - 50)[None],
                         ring_trace=ring[None])
            cnmf = np.zeros(T) if movie == "M" else neuron
            np.savez(run / movie / "peers" / f"{c['spec']}.npz", tuncat=roi[movie], tuncat_astro=roi[movie],
                     tuncat_raw_roi=roi[movie], fissa=roi[movie], cnmf=cnmf, cnmf_astro=neuron)
            if c["spec"] == "M-001":
                np.savez(run / movie / "suite2p" / f"{c['spec']}.npz", suite2p=neuron)
    return run


def test_d6_scores_measures_and_failures(tmp_path: Path) -> None:
    run = d6_recording(tmp_path, "rec1", "A-M1_loc01")
    s = stats.d6_scores(run).set_index(["case", "method"])
    assert list(s.columns[:7]) == ["location", "recording", "label", "territory", "territory_share",
                                   "neighbours", "atoms"]
    for case in ("M-001", "M-002"):
        assert s.loc[(case, "demix"), "leak_5hz"] == 0.0 and s.loc[(case, "demix"), "status"] == "ok"
        assert s.loc[(case, "roi"), "leak_5hz"] == 1.0 and s.loc[(case, "tuncat"), "leak_05hz"] == 1.0
        assert abs(s.loc[(case, "ring"), "leak_5hz"] - 0.5) < 1e-9
        assert s.loc[(case, "demix"), "r_5hz"] == pytest.approx(1.0)
        c = s.loc[(case, "cnmf")]                                    # constant on M: leakage 1, r 0
        assert (c["status"], c["leak_5hz"], c["leak_05hz"], c["r_5hz"], c["r_05hz"]) == ("unusable", 1, 1, 0, 0)
    assert s.loc[("M-001", "suite2p"), "status"] == "ok"
    assert tuple(s.loc[("M-002", "suite2p"), ["status", "leak_5hz", "r_5hz"]]) == ("missing", 1.0, 0.0)
    x, y = (np.load(run / m / "peers" / "M-001.npz")["fissa"] for m in ("M", "R"))
    astro = np.load(run / "M" / "compare" / "M-001.npz")["roi_mean"] - np.load(run / "R" / "compare" /
                                                                                "M-001.npz")["roi_mean"]
    assert s.loc[("M-001", "fissa"), "leak_05hz"] == metrics.leakage(x, y, astro, stats.D6_FPS, 0.5)


def d6_score_rows(seed: int = 2) -> list[pd.DataFrame]:
    """Score tables of five locations (one recording each; two location names of one mouse)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, loc in enumerate(("A-M1_loc01", "A-M2_loc02", "B_loc03", "B_loc04", "C_loc05")):
        rows = []
        for k, cls in enumerate(("in", "in", "out", "between")):
            for m in stats.METHODS:
                lk = 0.5 + WORSE[m] + rng.normal(0, 0.05)
                rows.append({"location": loc, "recording": f"rec{i}", "case": f"M-{k:03d}", "territory": cls,
                             "method": m, "status": "failure" if (m == "cnmf" and k == 0) else "ok",
                             "r_5hz": 0.9 - WORSE[m], "r_05hz": 0.8, "leak_5hz": lk, "leak_05hz": lk / 2})
        out.append(pd.DataFrame(rows))
    return out


def test_d6_tables() -> None:
    parts = d6_score_rows()
    rec = stats.d6_tables(parts)
    assert rec["recordings"] == 5 and rec["primary"] == "leakage in territory"
    col = rec["columns"]["leakage in territory"]
    assert col["mouse"] == ["A", "A", "B", "B", "C"] and col["cases"] == [2] * 5 and col["recordings"] == [1] * 5
    t = pd.concat(parts)
    med = t[(t.territory == "in") & (t.method == "demix")].groupby("location").leak_5hz.median()
    np.testing.assert_array_equal(col["scores"]["demix"], med.to_numpy())
    raw = {m: col["stats"][m]["p"] for m in stats.COMPARED}
    assert {m: col["stats"][m]["p_holm"] for m in stats.COMPARED} == metrics.holm(raw)
    r = rec["columns"]["r in territory (trace kept)"]
    assert not r["lower_is_better"] and r["stats"]["fissa"]["wins"] == 5           # higher r is better
    assert len(rec["columns"]["leakage, all cases"]["locations"]) == 5
    assert rec["failures"]["cnmf"] == 5 and rec["failures"]["demix"] == 0
    assert rec["cases_by_class"] == {"in": 10, "out": 5, "between": 5}
    assert "2 mice" not in stats.d6_markdown(rec) and "3 mice" in stats.d6_markdown(rec)


def d7_recording(root: Path, name: str, seed: int, scored: bool = True) -> None:
    """A D7 recording folder (``stats.d7_tables``' layout): two injected sites at x1 and x3, one leakage
    target at x1 and x3 (with its x0 case, not scored)."""
    rng = np.random.default_rng(seed)
    d = root / name
    (d / "compare").mkdir(parents=True)
    (d / "scores").mkdir()
    index = [{"spec": f"inj0{j}-astro-x{s}", "base": f"inj0{j}-astro", "test": "injection", "scale": float(s),
              "site": "astro"} for j in range(2) for s in (1, 3)]
    index += [{"spec": f"M-007-astro-x{s}", "base": "M-007", "test": "leakage", "scale": float(s), "site": ""}
              for s in (0, 1, 3)]
    (d / "compare" / "index.json").write_text(json.dumps(index))
    inj, leak = [], []
    for c in index[:4]:
        for m in stats.METHODS:
            err = 0.3 + WORSE[m] + rng.normal(0, 0.02)
            inj.append({"case": c["spec"], "site": "astro", "scale": c["scale"], "method": m, "corr": 0.9,
                        "gain": 1.0, "err_raw": err, "err_2hz": err, "err_05hz": err})
    for s in (1.0, 3.0):
        for m in stats.METHODS:
            leak.append({"target": "M-007", "scale": s, "method": m, "leak": 0.2 + WORSE[m], "leak_05hz": 0.1,
                         "kept_corr": 0.9})
    inj_t, leak_t = pd.DataFrame(inj), pd.DataFrame(leak)
    if seed == 0:                       # failures: CNMF missing on one case, FISSA's NaN correlation, a NaN leak
        inj_t = inj_t[~((inj_t.case == "inj00-astro-x1") & (inj_t.method == "cnmf"))].copy()
        inj_t.loc[(inj_t.case == "inj01-astro-x1") & (inj_t.method == "fissa"), "corr"] = np.nan
        leak_t.loc[(leak_t.scale == 1.0) & (leak_t.method == "tuncat"), "leak"] = np.nan
    inj_t.to_csv(d / "scores" / "injection_scores.csv", index=False)
    leak_t.to_csv(d / "scores" / "leakage_scores.csv", index=False)
    if scored:
        (d / "scores" / "summary.txt").write_text("scored\n")


def test_d7_failure_rule_units_and_locations(tmp_path: Path) -> None:
    for name, seed in (("loc1_data1", 0), ("loc1_data2", 1), ("loc2_data1", 2), ("loc2_data3", 3)):
        d7_recording(tmp_path, name, seed)
    d7_recording(tmp_path, "loc2_data9", 9, scored=False)                  # no summary.txt: not counted
    rec = stats.d7_tables(tmp_path)
    assert rec["recordings"] == 4
    col = rec["columns"]["error x1"]
    assert col["recordings"] == ["loc1/data1", "loc1/data2", "loc2/data1", "loc2/data3"]
    assert col["location"] == ["loc1", "loc1", "loc2", "loc2"] and col["cases"] == [2] * 4
    s = pd.read_csv(tmp_path / "loc1_data1" / "scores" / "injection_scores.csv").set_index(["case", "method"])
    roi = s.loc[("inj00-astro-x1", "roi"), "err_05hz"]
    other = s.loc[("inj01-astro-x1", "cnmf"), "err_05hz"]
    assert col["scores"]["cnmf"][0] == np.median([roi, other])               # the missing row: the raw ROI's
    assert rec["failures"]["injection"]["cnmf"] == 1 and rec["failures"]["injection"]["fissa"] == 1
    assert rec["failures"]["leakage"]["tuncat"] == 1 and rec["failures"]["leakage"]["cases"] == 8
    assert rec["columns"]["leakage x1"]["scores"]["tuncat"][0] == 1.0       # a failed leakage scores 1
    assert set(rec["by_location"]["error x1"]) == {"loc1", "loc2"}
    assert rec["by_location"]["error x1"]["loc1"]["n"] == 2
    assert "err_2hz" in col["other_bands"] and "other_bands" not in rec["columns"]["leakage x3"]
    raw = {m: col["stats"][m]["p"] for m in stats.COMPARED}
    assert {m: col["stats"][m]["p_holm"] for m in stats.COMPARED} == metrics.holm(raw)
    assert rec["failures"]["injection"]["cases"] == 16                     # 4 recordings x 4 injected cases
    assert "Failures (injection, 16 cases" in stats.d7_markdown(rec)


def test_commands(tmp_path: Path) -> None:
    tables = d1_tables()
    scores = tmp_path / "pooled"
    scores.mkdir()
    for k, t in tables.items():
        t.to_csv(scores / f"{k}_scores_8loc.csv", index=False)
    out = tmp_path / "out"
    assert stats.main(["d1", "--scores", str(scores), "--suffix", "8loc", "--out", str(out)]) == 0
    for f in ("d1_stats.json", "d1_suite2p.json", "d1_secondary.json", "d1_stats.md"):
        assert (out / f).exists()
    assert json.loads((out / "d1_stats.json").read_text())["error x1"]["holm"]["family"][-1] == "suite2p"
    with pytest.raises(SystemExit):
        stats.main(["ablations", "--full", str(scores), "--variant", f"I={scores}", f"II={scores}",
                    "--out", str(out)])
    assert stats.main(["ablations", "--full", str(scores), "--suffix", "8loc", "--variant", f"I={scores}",
                       f"II={scores}", f"III={scores}", "--out", str(out)]) == 0
    assert json.loads((out / "ablation_stats.json").read_text())["leakage x3"]["stats"]["I"]["p"] == 1.0
    run = d6_recording(tmp_path / "d6", "rec1", "A_loc01")
    assert stats.main(["d6-scores", "--run", str(run)]) == 0 and (run / "scores.csv").exists()
    for k in range(3):
        d7_recording(tmp_path / "d7", f"loc1_data{k}", k + 1)
    assert stats.main(["d7", "--root", str(tmp_path / "d7"), "--out", str(out)]) == 0
    assert json.loads((out / "d7.json").read_text())["recordings"] == 3


def test_error_split_parts_add_up() -> None:
    rng = np.random.default_rng(5)
    t = np.arange(600) / 20.0
    q = np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.sin(2 * np.pi * 0.03 * t)
    y = 0.8 * q + 0.2 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.1, t.size)
    s = stats.error_split(y, q, 20.0)
    assert s["err05"] ** 2 == pytest.approx(s["gain_part"] + s["resid_part"], rel=1e-12)
    assert s["err05"] == metrics.error(y, q, 20.0) and 0.7 < s["g05"] < 0.9
    assert s["g_raw"] == pytest.approx(metrics.gain(y, q), rel=1e-12)
    exact = stats.error_split(2 * q, q, 20.0)
    assert exact["g05"] == pytest.approx(2.0) and exact["resid_part"] == pytest.approx(0.0, abs=1e-20)
