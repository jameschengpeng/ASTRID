"""The peers' runner on D3 (``reproduce/astrid_repro/cai1_peers.py``) on a synthetic exported recording, with
stand-ins for FISSA, TUnCaT and CaImAn (none is installed with ASTRID): the alpha grids, the settings, CaImAn's
data layout, the saved keys, the merging of reruns and the timing file."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cai1_peers as P  # noqa: E402

FAKES = {
    "fissa.py": """
        import numpy as np

        class Experiment:
            def __init__(self, images, rois, alpha, ncores_preparation, ncores_separation, verbosity):
                self.movie, self.roi, self.alpha = images[0], rois[0][0], alpha

            def separate(self):
                trace = self.movie[:, self.roi].mean(axis=1)
                self.raw = np.empty((1, 1), dtype=object)
                self.raw[0, 0] = trace[None]
                self.result = np.empty((1, 1), dtype=object)
                self.result[0, 0] = (self.alpha * trace)[None]
        """,
    "tuncat/__init__.py": "",
    "tuncat/run_TUnCaT.py": """
        import numpy as np

        def run_TUnCaT(name, movie, masks, tmp, **kw):
            raw = np.stack([movie[:, m].mean(axis=1) for m in masks], axis=1)
            return raw * kw["list_alpha"][0], None, raw, raw + kw["tol"]
        """,
    "caiman/__init__.py": """
        import os

        import h5py
        import numpy as np
        from . import cluster

        SEEN = []

        def save_memmap(files, base_name, order, var_name_hdf5):
            # CaImAn 1.13: the memmap goes to its temporary folder, $CAIMAN_TEMP (else ~/caiman_data/temp, which
            # this fake refuses), whatever folder base_name names
            temp = os.environ.get("CAIMAN_TEMP")
            assert temp and os.path.isdir(temp), "CAIMAN_TEMP must name an existing folder"
            with h5py.File(files[0], "r") as f:
                mov = np.asarray(f[var_name_hdf5], dtype=np.float32)
            fname = os.path.join(temp, os.path.basename(base_name) + "x.npy")
            np.save(fname, mov)
            SEEN.append(fname)
            return fname

        def load_memmap(fname):
            mov = np.load(fname)
            return np.reshape(mov, (mov.shape[0], -1), order="F").T, mov.shape[1:], mov.shape[0]

        def stop_server(dview=None):
            pass
        """,
    "caiman/cluster.py": """
        def setup_cluster(backend, n_processes, single_thread):
            return None, None, n_processes
        """,
    "caiman/source_extraction/__init__.py": "",
    "caiman/source_extraction/cnmf/__init__.py": "",
    "caiman/source_extraction/cnmf/cnmf.py": """
        import numpy as np

        CALLS = []

        class CNMF:
            def __init__(self, **kw):
                self.kw = kw
                CALLS.append(kw)

            def fit(self, images):
                T = images.shape[0]
                Yr = np.reshape(images, (T, -1), order="F").T
                A = self.kw["Ain"].astype(float)
                A /= np.linalg.norm(A, axis=0)
                self.estimates = type("E", (), {"C": A.T @ Yr, "YrA": np.ones((1, T))})()
        """,
}


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "fake_tools"
    for rel, text in FAKES.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(textwrap.dedent(text))
    monkeypatch.syspath_prepend(str(d))
    for mod in [m for m in sys.modules if m.split(".")[0] in ("fissa", "tuncat", "caiman")]:
        monkeypatch.delitem(sys.modules, mod)


def _recording(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    folder.mkdir(parents=True, exist_ok=True)
    mov = np.random.default_rng(3).integers(100, 200, (40, 8, 6)).astype(np.int16)
    roi = np.zeros((8, 6), dtype=bool)
    roi[2:6, 1:3] = True                                    # not symmetric: a row/column mix-up shows
    with h5py.File(folder / "r1.h5", "w") as f:
        f["mov"], f["roi"] = mov, roi
    return mov.astype(np.float32), roi


def test_settings_of_the_paper() -> None:
    assert len(P.FISSA_ALPHAS) == 9 and len(P.TUNCAT_ALPHAS) == 13
    assert P.TUNCAT["tol"] == 1e-4 and P.TUNCAT["max_iter"] == 20000 and P.TUNCAT["flexible_alpha"] is True


def test_fissa_and_tuncat(tmp_path: Path, tools: None) -> None:
    mov, roi = _recording(tmp_path / "data")
    out = tmp_path / "peers"
    assert P.main(["r1", "--data", str(tmp_path / "data"), "--out", str(out)]) == 0
    roi_mean = mov[:, roi].astype(np.float64).mean(axis=1)
    with np.load(out / "r1.npz") as z:
        keys = set(z.files)
        assert keys == {f"fissa_a={a:g}" for a in P.FISSA_ALPHAS} | {f"tuncat_a={a:g}" for a in P.TUNCAT_ALPHAS} \
            | {"fissa_raw", "tuncat_raw", "tuncat_bg"}
        np.testing.assert_allclose(z["fissa_a=0.2"], 0.2 * roi_mean, rtol=1e-6)
        np.testing.assert_allclose(z["tuncat_a=20"], 20 * roi_mean, rtol=1e-6)
        np.testing.assert_allclose(z["tuncat_bg"], roi_mean + 1e-4, rtol=1e-6)
    seconds = json.loads((out / "r1_seconds.json").read_text(encoding="utf-8"))
    assert set(seconds) == {k for k in keys if "_a=" in k}


def test_cnmf_merges_into_the_file(tmp_path: Path, tools: None) -> None:
    mov, roi = _recording(tmp_path / "data")
    out = tmp_path / "peers"
    P.main(["r1", "--data", str(tmp_path / "data"), "--out", str(out), "--methods", "fissa"])
    assert P.main(["r1", "--data", str(tmp_path / "data"), "--out", str(out), "--methods", "cnmf",
                   "--cpus", "2"]) == 0
    from caiman.source_extraction.cnmf import cnmf

    kw = cnmf.CALLS[-1]
    assert (kw["k"], kw["p"], kw["gnb"], kw["merge_thresh"], kw["n_processes"]) == (1, 1, 1, 1, 2)
    np.testing.assert_array_equal(kw["Ain"][:, 0], roi.ravel(order="F"))      # CaImAn's column-major pixels
    with np.load(out / "r1.npz") as z:
        assert "fissa_a=0.01" in z.files and {"cnmf", "cnmf_C"} <= set(z.files)   # merged, not replaced
        expected = mov[:, roi].astype(np.float64).sum(axis=1) / np.sqrt(roi.sum())
        np.testing.assert_allclose(z["cnmf_C"], expected, rtol=1e-5)
        np.testing.assert_allclose(z["cnmf"], expected + 1, rtol=1e-5)
    with pytest.raises(SystemExit):
        P.main(["r1", "--data", str(tmp_path / "data"), "--out", str(out), "--methods", "suite2p"])


def test_cnmf_settings_and_its_memmap(tmp_path: Path, tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """--cnmf-gnb and --cnmf-p reach CaImAn (the cross-validation grid); the memmap is written to a temporary
    CAIMAN_TEMP that is gone after the fit, and the caller's CAIMAN_TEMP is left as it was."""
    _recording(tmp_path / "data")
    monkeypatch.setenv("CAIMAN_TEMP", "the caller's")
    assert P.main(["r1", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "grid" / "gnb3_p2" / "cai1"),
                   "--methods", "cnmf", "--cnmf-gnb", "3", "--cnmf-p", "2"]) == 0
    import caiman
    from caiman.source_extraction.cnmf import cnmf

    kw = cnmf.CALLS[-1]
    assert (kw["p"], kw["gnb"], kw["k"], kw["merge_thresh"]) == (2, 3, 1, 1)
    memmap = Path(caiman.SEEN[-1])
    assert memmap.name.startswith("r1_memmap_") and not memmap.parent.exists()
    assert os.environ["CAIMAN_TEMP"] == "the caller's"
    with np.load(tmp_path / "grid" / "gnb3_p2" / "cai1" / "r1.npz") as z:
        assert {"cnmf", "cnmf_C"} <= set(z.files)
