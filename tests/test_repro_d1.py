"""D1's input files and runner (``reproduce/astrid_repro/d1.py``) on a small synthetic location: the reader and
its checks, the settings of each test and of the ablations, and one recording run end to end (the cases, the
fits, the export, the scores, a rescoring with a peer's results and the pooled tables)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cases as C  # noqa: E402
from astrid_repro import d1 as D  # noqa: E402
from astrid_repro import peers  # noqa: E402
from test_repro_cases import FPS, toy, toy_movie  # noqa: E402

REC = "rec1"


def write_inputs(folder: Path, t: int = 200, fps: float = FPS) -> Path:
    """The toy location in D1's input files (``d1``'s docstring)."""
    loc, dff, astro_dff = toy(t)
    fld = loc.field
    d = folder / fld.name
    d.mkdir(parents=True)
    np.savez(d / "location.npz", neurons=fld.neurons, candidates=fld.candidates, events=loc.events, tau=fld.tau,
             max_proj=loc.max_proj, astro_label=loc.astro_label, fps=np.float64(fps), um_per_px=np.float64(2.0),
             border_px=np.int64(5), recordings=np.array([REC]))
    with h5py.File(d / f"{REC}.h5", "w") as f:
        f["mov"] = toy_movie(loc, dff, astro_dff)
        f["dff"] = dff
        f["astro_dff"] = astro_dff
    return d


def test_inputs_and_settings(tmp_path: Path) -> None:
    d = write_inputs(tmp_path / "data")
    loc = D.read_location(d)
    rec = D.read_recording(d, REC, loc)
    assert loc.field.name == "toy_loc01" and rec.frames == 200 and rec.dff.shape == (6, 200)
    assert rec.astro_dff.shape == (2, 200) and rec.movie().shape() == (200, 90, 100)
    s, astro = D.d1_settings(loc)
    assert astro and (s["injection"].max_iterations, s["leakage"].max_iterations) == (600, 2000)
    assert (s["injection"].tau_r, s["injection"].tau_d) == (0.05, 0.4)
    assert s["leakage"].kappa_a == s["injection"].kappa_a == 10.0
    assert D.d1_settings(loc, "I")[1] is False and D.d1_settings(loc, "I")[0]["injection"] == s["injection"]
    b, c = D.d1_settings(loc, "II")[0], D.d1_settings(loc, "III")[0]
    assert (b["injection"].kappa_a, b["leakage"].kappa_a, b["leakage"].max_iterations) == (0.0, 0.0, 2000)
    assert (c["injection"].kappa_rho, c["leakage"].kappa_rho, c["injection"].kappa_a) == (0.0, 0.0, 10.0)
    other = write_inputs(tmp_path / "other", fps=30.0)
    with pytest.raises(ValueError, match="not D1's"):
        D.d1_settings(D.read_location(other), None)
    with h5py.File(d / f"{REC}.h5", "a") as f:
        del f["dff"]
        f["dff"] = np.zeros((5, 200))
    with pytest.raises(ValueError, match="ROIs x frames"):
        D.read_recording(d, REC, loc)


def test_one_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_inputs(tmp_path / "data")
    monkeypatch.setattr(C, "SITES", 2)                         # 2 sites of each kind, 1 leakage target
    monkeypatch.setattr(C, "LEAK_TARGETS", 1)
    config = yaml.safe_load(D.CONFIG.read_text(encoding="utf-8"))
    entry = config["datasets"]["D1"]
    entry["settings"] = {"max_iterations": 30}                 # short fits
    entry["leakage_test_settings"] = {"max_iterations": 40}
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = D.run_recording(tmp_path / "data", "toy_loc01", REC, tmp_path / "out", config=cfg, log=lambda s: None)
    assert run == tmp_path / "out" / "toy_loc01_rec1"
    info = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert info["tests"] == ["injection", "leakage"] and info["fps"] == FPS and info["astro_components"]
    assert info["choices"]["injection"]["sources"] == [2, 1] and info["choices"]["leakage"]["targets"] == [3]
    assert info["settings"]["leakage"]["max_iterations"] == 40
    names = info["cases"]["injection"] + info["cases"]["leakage"]
    assert names[:2] == ["inj00-astro-x1", "inj00-astro-x3"] and info["cases"]["leakage"][0] == "M-003-astro-x0"
    fits = pd.read_csv(run / "injection" / "fits.csv")
    assert list(fits.case) == info["cases"]["injection"] and (fits.iterations <= 30).all()
    assert (pd.read_csv(run / "leakage" / "fits.csv").iterations <= 40).all()
    every, todo = peers.pending(run / "compare")                # the peers' view of the export
    assert every == names and todo == names
    inj = pd.read_csv(run / "scores" / "injection_scores.csv")
    leak = pd.read_csv(run / "scores" / "leakage_scores.csv")
    assert set(inj.method) == {"demix", "ring"} and len(inj) == 2 * len(info["cases"]["injection"])
    assert set(leak.scale) == {1.0, 3.0} and set(leak.target) == {"M-003"}
    results = tmp_path / "peer_results"                        # a peer's traces: scored where present
    results.mkdir()
    for name in names:
        with np.load(run / ("injection" if name.startswith("inj") else "leakage") / f"{name}.npz") as z:
            np.savez(results / f"{name}.npz", fissa=z["ring_trace"][0] * 0.5)
    assert D.main(["score", "--run", str(run), "--results", str(results)]) == 0
    inj2 = pd.read_csv(run / "scores" / "injection_scores.csv")
    assert list(inj2.method.unique()) == ["demix", "ring", "fissa"]
    pd.testing.assert_frame_equal(inj2[inj2.method != "fissa"].reset_index(drop=True), inj)
    assert D.main(["pool", "--root", str(tmp_path / "out"), "--out", str(tmp_path / "pooled")]) == 0
    pooled = pd.read_csv(tmp_path / "pooled" / "injection_scores.csv")
    need = {"location", "recording", "case", "site", "scale", "method", "err_05hz", "gain"}
    assert need <= set(pooled.columns) and set(pooled.location) == {"toy_loc01"} and set(pooled.recording) == {REC}
    pooled = pd.read_csv(tmp_path / "pooled" / "leakage_scores.csv")
    assert {"location", "recording", "target", "scale", "method", "leak", "kept_corr"} <= set(pooled.columns)
    # an ablation scored against the full model's export
    again = D.run_recording(tmp_path / "data", "toy_loc01", REC, tmp_path / "out_A", tests=["leakage"],
                            variant="I", cases_dir=run / "compare", config=cfg, log=lambda s: None)
    assert not (again / "compare").exists()
    with np.load(again / "leakage" / "M-003-astro-x1.npz") as z:
        assert "astro" not in list(z["kinds"]) and "A-inj" not in list(z["names"])
    assert set(pd.read_csv(again / "scores" / "leakage_scores.csv").method) == {"demix", "ring"}
