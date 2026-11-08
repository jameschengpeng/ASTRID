"""The runner of ASTRID on the public datasets (``reproduce/astrid_repro/run.py``) and the crop of D2's movies
(``abo_crop.py``) on small synthetic data: the grids and settings read from ``datasets.yaml``, the units, the
layout of the results, a fit, the resumption, and the crop's window and file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import abo_crop  # noqa: E402
from astrid_repro import public_data as D  # noqa: E402
from astrid_repro import run as R  # noqa: E402


def test_grids_and_names() -> None:
    for d in ("D2", "D3", "D4", "D5"):
        g = R.grid(d)
        assert len(g) == 19 and g[:2] == [(1.0, 0.03), (1.0, 0.1)]
    assert R.grid("D4")[-3:] == [(1.0, 1.0), (2.0, 1.0), (4.0, 1.0)]
    assert R.setting_name(2.0, 0.1) == "ks2_kb0.1" and R.setting_name(0.5, 0.01) == "ks0.5_kb0.01"
    assert R.setting_name(32.0, 0.003) == "ks32_kb0.003"


def _archive(tmp_path: Path) -> Path:
    """One NAOMi-like video in TUnCaT's layout: two cells, one active, and overlapping masks."""
    arch = tmp_path / "archive"
    (arch / "data" / "NAOMi" / "GT Masks").mkdir(parents=True)
    rng = np.random.default_rng(0)
    t, h, w = 240, 40, 40
    yy, xx = np.mgrid[:h, :w]
    cells = [((yy - 15) ** 2 + (xx - 15) ** 2 <= 16), ((yy - 18) ** 2 + (xx - 18) ** 2 <= 16)]
    act = np.zeros(t)
    act[[40, 120, 200]] = 1.0
    act = np.convolve(act, np.exp(-np.arange(30) / 5.0))[:t]
    mov = 100 + rng.normal(0, 2, (t, h, w)) + 60 * act[:, None, None] * cells[0]
    with h5py.File(D.movie_path(arch, "NAOMi", "Video_0"), "w") as f:
        f["mov"] = mov.astype(np.float32)
    masks = np.stack(cells).astype(np.uint8)
    scipy.io.savemat(D.masks_path(arch, "NAOMi", "Video_0"), {"FinalMasks": masks.transpose(2, 1, 0)})
    return arch


def test_units_and_a_fit(tmp_path: Path) -> None:
    arch = _archive(tmp_path)
    units = R.units("D4", arch)
    assert len(units) == 10 * 19 and units[0]["fps"] == 30.0 and units[0]["um_per_px"] == 1.0
    assert (units[0]["tau_r"], units[0]["tau_d"]) == (0.045, 0.142)
    out = tmp_path / "results"
    done = R.run("D4", arch, out, videos=["Video_0"], settings=["ks2_kb0.1"])
    assert done == [out / "NAOMi" / "ks2_kb0.1" / "Video_0" / "result.npz"]
    with np.load(done[0]) as z:
        assert z["dff__Video_0"].shape == (2, 240) and bool(z["fitted"].all())
    assert D.ours_traces(out, "NAOMi", "Video_0")["ks2_kb0.1"].shape == (2, 240)    # what the tables read
    assert R.run("D4", arch, out, videos=["Video_0"], settings=["ks2_kb0.1"]) == []   # done already: skipped


def test_cai1_units(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(json.dumps([
        {"name": "c_001", "cell": "c", "rec": "001", "indicator": "GCaMP6s", "fps": 60.060060060065474}]),
        encoding="utf-8")
    u = R.units("D3", tmp_path)
    assert len(u) == 19 and u[0]["fps"] == 60.0601 and (u[0]["tau_r"], u[0]["tau_d"]) == (0.179, 0.55)
    assert u[0]["rois"] is None and u[0]["um_per_px"] == 0.47


def test_crop(tmp_path: Path) -> None:
    assert abo_crop.window((115770, 512, 512)) == (23154, 156, 156)
    full = np.arange(26 * 512 * 512, dtype=np.uint32).reshape(26, 512, 512) % 65521
    with h5py.File(tmp_path / "ophys_experiment_1.h5", "w") as f:
        f["data"] = full.astype(np.uint16)
    out = tmp_path / "ABO"
    out.mkdir()
    assert abo_crop.main(["--full", str(tmp_path), "--out", str(out), "--experiments", "1"]) == 0
    with h5py.File(out / "1.h5", "r") as f:
        mov = f["mov"]
        assert mov.shape == (5, 200, 200) and mov.dtype == np.uint16 and mov.chunks == (1, 200, 200)
        assert np.array_equal(mov[()], full[:5, 156:356, 156:356].astype(np.uint16))
    rec = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert rec["frames"] == [0, 5] and rec["rows"] == [156, 356] and rec["source_shape"] == [26, 512, 512]
    assert abo_crop.URL.format("501484643").endswith("ophys_movies/ophys_experiment_501484643.h5")
    with pytest.raises(SystemExit):
        abo_crop.main(["--out", str(out)])
