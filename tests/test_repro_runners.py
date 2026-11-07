"""The peers' runners (``reproduce/astrid_repro/peers.py``, ``suite2p_runner.py``) on synthetic cases, with
stand-ins for FISSA, TUnCaT, CaImAn, Suite2p and PyTorch (none is installed with ASTRID): the case format, the
settings, the plumbing, the component matching and the argument handling."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import peers, suite2p_runner  # noqa: E402

FAKE_FISSA = """
import numpy as np

class Experiment:
    def __init__(self, images, rois, ncores_preparation, ncores_separation, verbosity):
        self.movie, self.roi = images[0], rois[0][0]

    def separate(self):
        trace = self.movie[:, self.roi].mean(axis=1)
        self.raw = np.empty((1, 1), dtype=object)
        self.raw[0, 0] = trace[None]
        self.result = np.empty((1, 1), dtype=object)
        self.result[0, 0] = (2 * trace)[None]
"""
FAKE_TUNCAT = """
import numpy as np

def run_TUnCaT(name, movie, masks, tmp, **kw):
    raw = np.stack([movie[:, m].mean(axis=1) for m in masks], axis=1)
    return raw * kw["list_alpha"][0], None, raw, raw + masks.shape[0] + kw["tol"]
"""


def fake_tools(tmp: Path) -> Path:
    """Stand-in fissa and tuncat packages: FISSA returns twice the ROI mean, TUnCaT the ROI means, its
    background the ROI mean plus the number of masks plus the tolerance."""
    d = tmp / "fake_tools"
    (d / "tuncat").mkdir(parents=True)
    (d / "fissa.py").write_text(textwrap.dedent(FAKE_FISSA))
    (d / "tuncat" / "__init__.py").write_text("")
    (d / "tuncat" / "run_TUnCaT.py").write_text(textwrap.dedent(FAKE_TUNCAT))
    return d


def use_fake_tools(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(fake_tools(tmp)))
    for mod in ("fissa", "tuncat", "tuncat.run_TUnCaT"):
        monkeypatch.delitem(sys.modules, mod, raising=False)


def fake_cases(folder: Path, cases: list[str], T: int = 50) -> None:
    """A case folder in the documented format: one base (2 neighbours, 1 atom) and the given cases of it."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    np.save(folder / "inj00-control_movie.npy", rng.normal(size=(T, 6, 5)).astype(np.float32))
    target = np.zeros((6, 5), dtype=bool)
    target[2:4, 2:4] = True
    neighbours = np.zeros((2, 6, 5), dtype=bool)
    neighbours[0, 0, 0] = neighbours[1, 5, 4] = True
    np.savez(folder / "inj00-control_masks.npz", target=target, neighbours=neighbours,
             atoms=np.ones((1, 6, 5), dtype=bool), box=np.array([0, 6, 0, 5]))
    for c in cases:
        np.savez(folder / f"{c}.npz", footprint=np.full((6, 5), 0.5, dtype=np.float32), amplitude=np.arange(T * 1.0),
                 truth=np.arange(T * 1.0))
    (folder / "index.json").write_text(json.dumps([{"spec": c, "base": peers.base_of(c), "test": "injection",
                                                    "scale": 1.0, "site": "control"} for c in cases]))


def test_settings_of_the_paper() -> None:
    assert peers.TUNCAT["tol"] == 1e-4 and peers.TUNCAT["max_iter"] == 20000 and peers.TUNCAT["list_alpha"] == [1]
    assert peers.TUNCAT["flexible_alpha"] and peers.TUNCAT["multi_alpha"]
    assert peers.CNMF == {"p": 1, "merge_thresh": 1, "gnb": 1, "method_deconvolution": "oasis", "border_pix": 0,
                          "low_rank_background": True, "n_pixels_per_process": 1000, "rf": None, "stride": None}
    assert peers.DEFAULT_METHODS == ("fissa", "tuncat") and peers.TUNCAT_CPUS == 3


def test_case_format() -> None:
    assert peers.base_of("inj00-astro-x3") == "inj00-astro"
    assert peers.base_of("M-039-astro-x1") == peers.base_of("M-039-astro-x0") == "M-039"
    assert peers.base_of("M-002") == "M-002"


def test_case_movie_adds_the_signal(tmp_path: Path) -> None:
    fake_cases(tmp_path, ["inj00-control-x1"])
    movie, masks = peers.case_movie(tmp_path, "inj00-control-x1")
    base = np.load(tmp_path / "inj00-control_movie.npy")
    assert movie.dtype == np.float32 and movie.shape == base.shape
    np.testing.assert_allclose(movie - base, np.broadcast_to(0.5 * np.arange(50.0)[:, None, None], base.shape),
                               atol=1e-4)
    assert set(masks) == {"target", "neighbours", "atoms", "box"}


def test_run_case_saves_merges_and_replaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each method's arrays saved; TUnCaT gets the target, the neighbours, then the atoms; a rerun of one
    method merges into the results file; an unreadable file is replaced."""
    use_fake_tools(tmp_path, monkeypatch)
    cases = tmp_path / "compare"
    fake_cases(cases, ["inj00-control-x1"])
    res = peers.run_case(cases, "inj00-control-x1")
    with np.load(cases / "results" / "inj00-control-x1.npz") as z:
        saved = {k: z[k] for k in z.files}
    assert set(saved) == {k for m in peers.DEFAULT_METHODS for k in peers.RESULT_KEYS[m]}
    np.testing.assert_allclose(saved["fissa"], 2 * saved["fissa_raw_roi"])
    np.testing.assert_allclose(saved["tuncat"], saved["tuncat_raw_roi"])
    np.testing.assert_allclose(saved["tuncat_bg"], saved["tuncat_raw_roi"] + 3 + 1e-4)   # 3 masks, tol 1e-4
    np.testing.assert_allclose(saved["tuncat_astro"], res["tuncat_astro"])
    only = peers.run_case(cases, "inj00-control-x1", ("fissa",), tuncat_tol=1e-3)
    assert set(only) == {"fissa", "fissa_raw_roi"}
    with np.load(cases / "results" / "inj00-control-x1.npz") as z:
        assert set(z.files) == set(saved)
        for k in saved:
            np.testing.assert_array_equal(z[k], saved[k])
    assert not list((cases / "results").glob("*.tmp"))
    (cases / "results" / "inj00-control-x1.npz").write_bytes(b"truncated")
    peers.run_case(cases, "inj00-control-x1", ("fissa",))
    with np.load(cases / "results" / "inj00-control-x1.npz") as z:
        assert set(z.files) == {"fissa", "fissa_raw_roi"}
    other = tmp_path / "elsewhere"
    peers.run_case(cases, "inj00-control-x1", ("tuncat",), other)
    assert (other / "inj00-control-x1.npz").exists()


def test_pending_and_shards(tmp_path: Path) -> None:
    cases = [f"inj{i:02d}-control-x1" for i in range(5)]
    fake_cases(tmp_path, cases)
    (tmp_path / "results").mkdir()
    np.savez(tmp_path / "results" / f"{cases[1]}.npz",
             **{k: np.zeros(3) for ks in peers.RESULT_KEYS.values() for k in ks})
    np.savez(tmp_path / "results" / f"{cases[2]}.npz", fissa=np.zeros(3), fissa_raw_roi=np.zeros(3))
    (tmp_path / "results" / f"{cases[3]}.npz").write_bytes(b"truncated")
    every, todo = peers.pending(tmp_path)
    assert every == cases and todo == [cases[0], cases[2], cases[3], cases[4]]
    assert peers.pending(tmp_path, ("fissa",))[1] == [cases[0], cases[3], cases[4]]
    assert peers.shards(todo, 3) == [[todo[0], todo[3]], [todo[1]], [todo[2]]]


def test_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cases(tmp_path, ["inj00-control-x1"])
    with pytest.raises(SystemExit):
        peers.main(["run", "--cases", str(tmp_path), "--methods", "fissa,caiman", "inj00-control-x1"])
    with pytest.raises(SystemExit):
        peers.main(["run", "--cases", str(tmp_path)])                     # no case named
    with pytest.raises(SystemExit):
        peers.main(["launch", "--cases", str(tmp_path), "--runners", "0"])
    started: list[tuple[list[str], dict[str, Any]]] = []

    class Record:                                                   # a runner that writes nothing, exits 0
        def __init__(self, cmd: list[str], **kw: Any) -> None:
            started.append((cmd, kw))

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(peers.subprocess, "Popen", Record)
    new = tmp_path / "rerun" / "results"
    assert peers.main(["launch", "--cases", str(tmp_path), "--runners", "1", "--methods", "tuncat",
                       "--results", str(new), "--tuncat-tol", "1e-3"]) == 1      # still pending: exit 1
    cmd, kw = started[0]
    assert cmd[1:5] == ["-m", "astrid_repro.peers", "run", "--cases"]
    assert Path(kw["cwd"]) == ROOT / "reproduce"
    assert cmd[cmd.index("--results") + 1] == str(new.resolve()) and cmd[cmd.index("--tuncat-tol") + 1] == "1e-3"
    assert cmd[cmd.index("--methods") + 1] == "tuncat" and cmd[-1] == "inj00-control-x1"
    assert "== launch" in (tmp_path / "rerun" / "run_0.log").read_text()


def test_launch_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runner processes of ``python -m astrid_repro.peers run`` with the stand-in tools."""
    monkeypatch.setenv("PYTHONPATH", str(fake_tools(tmp_path)))
    cases = tmp_path / "compare"
    names = ["inj00-control-x1", "inj00-control-x3", "inj00-control-x9"]
    fake_cases(cases, names)
    assert peers.launch(cases, 2, sys.executable) == 0
    assert sorted(p.stem for p in (cases / "results").glob("*.npz")) == sorted(names)
    assert (cases / "run_0.log").exists() and (cases / "run_1.log").exists()


class FakeCNMF:
    """CaImAn's CNMF with its data layout (``Ain`` (pixels, k), pixels in column-major order): each seed's
    footprint is its mask at unit norm, its trace the movie's projection on it; ``drop`` removes components."""
    drop: tuple[int, ...] = ()
    widen: dict[int, np.ndarray] = {}
    calls: list[dict[str, Any]] = []

    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        FakeCNMF.calls.append(kw)

    def fit(self, images: np.ndarray) -> None:
        T = images.shape[0]
        Yr = np.reshape(images, (T, -1), order="F").T
        A = self.kw["Ain"].astype(float)
        for j, extra in self.widen.items():
            A[extra, j] = 1.0
        A /= np.linalg.norm(A, axis=0)
        keep = [j for j in range(A.shape[1]) if j not in self.drop]
        A = A[:, keep]
        self.estimates = type("Estimates", (), {"A": A, "C": A.T @ Yr, "YrA": np.zeros((len(keep), T))})()


def test_cnmf_target_units_settings_and_matching() -> None:
    """The target's trace in the ROI's raw units, the seeds in CaImAn's layout with the paper's settings; a
    removed neighbour leaves the target matched, a removed target gives a zero trace; an atom lying over the
    target never takes its place."""
    rng = np.random.default_rng(4)
    movie = rng.normal(size=(40, 6, 5)).astype(np.float32)
    seeds = np.zeros((3, 6, 5), dtype=bool)
    seeds[0, 1:4, 2:4] = True                                         # not symmetric: a C/F mix-up shows
    seeds[1, 0, 0:2] = True
    seeds[2, 5, 3:5] = True
    mod = type("M", (), {"CNMF": FakeCNMF})
    roi_mean = movie[:, seeds[0]].mean(axis=1)
    try:
        FakeCNMF.calls = []
        trace, info = peers.cnmf_target(movie, seeds, cnmf_mod=mod)
        np.testing.assert_allclose(trace, roi_mean, rtol=1e-5, atol=1e-6)
        assert info[:4].tolist() == [3, 3, 0, 1.0]
        kw = FakeCNMF.calls[0]
        np.testing.assert_array_equal(kw["Ain"][:, 0], seeds[0].ravel(order="F"))
        assert kw["k"] == 3 and kw["gSig"] == [1, 1] and kw["n_processes"] == 1 and kw["check_nan"] is True
        assert {k: kw[k] for k in peers.CNMF} == peers.CNMF
        FakeCNMF.drop = (1,)
        trace, info = peers.cnmf_target(movie, seeds, cnmf_mod=mod)
        np.testing.assert_allclose(trace, roi_mean, rtol=1e-5, atol=1e-6)
        assert info[:3].tolist() == [3, 2, 0]
        FakeCNMF.drop = (0,)
        trace, info = peers.cnmf_target(movie, seeds, cnmf_mod=mod)
        assert not trace.any() and info[:3].tolist() == [3, 2, -1]
        seeds[2] = seeds[0]                                           # an atom exactly over the target
        spread = np.flatnonzero(~seeds[0].ravel(order="F"))[:8]
        FakeCNMF.drop, FakeCNMF.widen = (1,), {0: spread}             # the target's footprint spreads out
        trace, info = peers.cnmf_target(movie, seeds, cnmf_mod=mod)
        assert info[:3].tolist() == [3, 2, 0]
        assert info[3] < peers.jaccard(np.where(seeds[2].ravel(order="F"), 1.0, 0.0), seeds[0].ravel(order="F"))
        A = np.stack([s.ravel(order="F") for s in seeds], axis=1).astype(float)
        assert peers.seeds_of(A[:, [0, 2]], seeds) == [0, 2] and peers.seeds_of(A[:, :0], seeds) == []
    finally:
        FakeCNMF.drop, FakeCNMF.widen = (), {}


# --------------------------------------------------------------------------------------------- Suite2p

SETTINGS = {"extraction": {"neuropil_coefficient": 0.7}, "diameter": [12.0, 12.0]}


def fake_suite2p(monkeypatch: pytest.MonkeyPatch, roi_stats: Any, extraction: Any) -> None:
    torch = ModuleType("torch")
    torch.set_num_threads = lambda n: None  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]
    stats = ModuleType("suite2p.detection.stats")
    stats.roi_stats = roi_stats  # type: ignore[attr-defined]
    stats.assign_overlaps = lambda stat, ly, lx: stat  # type: ignore[attr-defined]
    extract = ModuleType("suite2p.extraction.extract")
    extract.extraction_wrapper = extraction  # type: ignore[attr-defined]
    modules = {"torch": torch, "suite2p": ModuleType("suite2p"),
               "suite2p.detection": ModuleType("suite2p.detection"), "suite2p.detection.stats": stats,
               "suite2p.extraction": ModuleType("suite2p.extraction"), "suite2p.extraction.extract": extract}
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)


def movie_and_masks(n: int = 2) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((n, 8, 8), bool)
    for k in range(n):
        masks[k, 1 + 3 * k:3 + 3 * k, 1:3] = True
    return np.ones((5, 8, 8), np.float32), masks


def raise_footprint(*args: Any, **kw: Any) -> Any:
    raise RuntimeError("footprint.ndim (2) must match len(axes) (2)")


def test_neuropil_subtraction_and_its_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    movie, masks = movie_and_masks()
    f, fneu = np.full((2, 5), 3.0), np.full((2, 5), 1.0)
    seen: dict[str, Any] = {}

    def keep_all(rois: np.ndarray, ly: int, lx: int, **kw: Any) -> np.ndarray:
        seen.update(kw, n=len(rois), lam=rois[0]["lam"])
        return rois

    fake_suite2p(monkeypatch, keep_all, lambda stat, movie, **kw: (f, fneu, None, None))
    _, _, trace, times = suite2p_runner.neuropil_subtraction(movie, masks, SETTINGS, 1)
    np.testing.assert_allclose(trace, 3.0 - 0.7)
    assert times["threads"] == 1 and seen["max_overlap"] is None and seen["n"] == 2 and (seen["lam"] == 1).all()
    fake_suite2p(monkeypatch, keep_all, raise_footprint)
    with pytest.raises(suite2p_runner.Suite2pError, match=r"extraction: RuntimeError: footprint\.ndim"):
        suite2p_runner.neuropil_subtraction(movie, masks, SETTINGS, 1)
    fake_suite2p(monkeypatch, raise_footprint, raise_footprint)
    with pytest.raises(suite2p_runner.Suite2pError, match="roi_stats: RuntimeError"):
        suite2p_runner.neuropil_subtraction(movie, masks, SETTINGS, 1)
    fake_suite2p(monkeypatch, lambda rois, ly, lx, **kw: rois[:1], raise_footprint)
    with pytest.raises(RuntimeError, match="roi_stats kept 1 of 2 ROIs") as err:
        suite2p_runner.neuropil_subtraction(movie, masks, SETTINGS, 1)
    assert not isinstance(err.value, suite2p_runner.Suite2pError)
    masks[1] = False
    with pytest.raises(ValueError, match="an empty mask"):
        suite2p_runner.neuropil_subtraction(movie, masks, SETTINGS, 1)


def test_cases_mode_records_a_failed_case_and_goes_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cases = tmp_path / "rec1" / "compare"
    names = ["inj00-control-x1", "inj00-control-x3", "inj00-control-x9"]
    fake_cases(cases, names, T=5)
    calls: list[int] = []

    def subtraction(frames: np.ndarray, m: np.ndarray, settings: Any, threads: int) -> Any:
        calls.append(len(m))
        if len(calls) == 2:                                         # the second case: Suite2p fails on it
            raise suite2p_runner.Suite2pError("extraction: RuntimeError: footprint")
        return None, None, np.full((len(m), frames.shape[0]), float(len(calls))), {"wall_s": 0.5, "cpu_s": 0.5}

    monkeypatch.setattr(suite2p_runner, "neuropil_subtraction", subtraction)
    monkeypatch.setattr(suite2p_runner, "suite2p_settings", lambda: SETTINGS)
    monkeypatch.setattr(suite2p_runner, "versions", lambda: {})
    results, meta = tmp_path / "results", tmp_path / "meta"
    argv = ["cases", "--cases", str(cases), "--results", str(results), "--threads", "1", "--out", str(meta)]
    assert suite2p_runner.main(argv) == 0
    assert calls == [3, 3, 3]                                       # the target and its two neighbours
    assert not (results / f"{names[1]}.npz").exists()
    np.testing.assert_array_equal(np.load(results / f"{names[0]}.npz")["suite2p"], np.full(5, 1.0))
    np.testing.assert_array_equal(np.load(results / f"{names[2]}.npz")["suite2p"], np.full(5, 3.0))
    prov = json.loads((meta / "rec1.json").read_text(encoding="utf-8"))
    assert list(prov["failed"]) == [names[1]] and prov["cases"] == 3 and list(prov["bases"]) == ["inj00-control"]
    seconds = json.loads((meta / "rec1_seconds.json").read_text(encoding="utf-8"))
    assert seconds["cases"] == 3 and seconds["wall_s"] == 1.0
    with pytest.raises(SystemExit):
        suite2p_runner.main(["cases", "--cases", str(cases), "--results", str(results), "--out", str(meta)])


def test_suite2p_inputs(tmp_path: Path) -> None:
    assert suite2p_runner.split_key(r"C:\data\movie.h5:mov") == (r"C:\data\movie.h5", "mov")
    assert suite2p_runner.split_key(r"C:\data\movie.h5") == (r"C:\data\movie.h5", None)
    assert suite2p_runner.split_key("movie.h5") == ("movie.h5", None)
    labels = np.zeros((4, 6), np.int32)
    labels[0, :2], labels[3, 4:] = 1, 2
    np.savez(tmp_path / "labels.npz", rois=labels)
    masks = suite2p_runner.load_masks(str(tmp_path / "labels.npz"))
    assert masks.shape == (2, 4, 6) and masks[0, 0, :2].all() and masks[1, 3, 4:].all()
    from scipy.io import savemat
    stack = np.zeros((4, 6, 2), bool)                               # MATLAB's (H, W, n)
    stack[0, :2, 0] = stack[3, 4:, 1] = True
    savemat(tmp_path / "FinalMasks_V.mat", {"FinalMasks": stack})
    masks = suite2p_runner.load_masks(str(tmp_path / "FinalMasks_V.mat"))
    assert masks.shape == (2, 6, 4) and masks[0, :2, 0].all()     # transposed as TUnCaT reads them
    a = np.ones((2, 3), bool)
    assert suite2p_runner.array_sha256(a) == suite2p_runner.array_sha256(a.copy(order="F"))
