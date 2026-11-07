"""The readers (Suite2p, CaImAn, AQuA2), the endpoints that end in demix, and the command line, on files made
here in each program's format."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from test_package import FPS, scene

import astrid
from astrid import aqua2
from astrid.cli import load_masks, main
from astrid.readers import CaimanMemmap, footprint_masks, read_caiman, read_suite2p

KW = dict(tau_r=0.05, tau_d=0.4)


@pytest.fixture(scope="module")
def sc() -> dict:
    return scene(t=300)


# --------------------------------------------------------------------------------------------- Suite2p
def write_suite2p(folder, sc: dict) -> None:
    folder.mkdir(parents=True)
    mov = np.round(sc["movie"]).astype(np.int16)
    t, ly, lx = mov.shape
    mov.tofile(folder / "data.bin")
    np.save(folder / "ops.npy", {"Ly": ly, "Lx": lx, "nframes": t, "fs": FPS}, allow_pickle=True)
    stat = []
    for m in sc["rois"]:
        yp, xp = np.nonzero(m)
        stat.append({"ypix": yp, "xpix": xp, "overlap": np.zeros(yp.size, dtype=bool)})
    stat.append({"ypix": np.array([1, 1]), "xpix": np.array([1, 2]), "overlap": np.zeros(2, dtype=bool)})
    np.save(folder / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(folder / "iscell.npy", np.array([[1, 0.9], [1, 0.8], [0, 0.1]]))


def test_suite2p(tmp_path, sc) -> None:
    folder = tmp_path / "suite2p" / "plane0"
    write_suite2p(folder, sc)
    inp = read_suite2p(folder)
    assert inp.rois.shape == (2, 40, 48) and np.array_equal(inp.rois, sc["rois"]) and inp.fps == FPS
    assert list(inp.index) == [0, 1]
    res = astrid.from_suite2p(folder, um_per_px=2.0, **KW)
    assert res.arrays()["fitted"].all() and res.provenance["roi_index"] == [0, 1]
    assert np.corrcoef(res.dff[0], sc["neurons"][0])[0, 1] > 0.98


# --------------------------------------------------------------------------------------------- CaImAn
def write_caiman(tmp_path, sc: dict):
    import h5py
    t, h, w = sc["movie"].shape
    mmap = tmp_path / f"movie_d1_{h}_d2_{w}_d3_1_order_C_frames_{t}.mmap"
    np.ascontiguousarray(sc["movie"].reshape((t, h * w), order="F").T, dtype=np.float32).tofile(mmap)
    cols = []
    for m in sc["rois"]:
        cy, cx = np.argwhere(m).mean(axis=0)
        bump = np.exp(-((np.mgrid[:h, :w][0] - cy) ** 2 + (np.mgrid[:h, :w][1] - cx) ** 2) / 8.0) * m
        cols.append(bump.ravel(order="F"))
    a = sparse.csc_matrix(np.stack(cols, axis=1))
    path = tmp_path / "cnm.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("estimates/A")
        g["data"], g["indices"], g["indptr"], g["shape"] = a.data, a.indices, a.indptr, np.array(a.shape)
        f["dims"] = np.array([h, w])
        f["params/data/fr"] = FPS
        f["params/data/dxy"] = np.array([2.0, 2.0])
        f["estimates/idx_components"] = np.array([0, 1])
        f["mmap_file"] = str(mmap)
    return path, mmap


def test_caiman(tmp_path, sc) -> None:
    path, mmap = write_caiman(tmp_path, sc)
    mov = CaimanMemmap(mmap)
    assert np.array_equal(mov[0:5], sc["movie"][0:5])
    inp = read_caiman(path)
    assert inp.rois.shape == (2, 40, 48) and inp.um_per_px == 2.0 and inp.fps == FPS
    assert all((r & ~m).sum() == 0 for r, m in zip(inp.rois, sc["rois"], strict=True))
    res = astrid.from_caiman(path, **KW)
    assert res.arrays()["fitted"].all()
    assert np.corrcoef(res.dff[1], sc["neurons"][1])[0, 1] > 0.98


def test_footprint_masks_energy() -> None:
    col = np.zeros(16)
    col[[0, 5, 6]] = [3.0, 1.0, 0.1]                      # energy shares 0.9, 0.0999, 0.001
    masks = footprint_masks(col[:, None], (4, 4), threshold=0.95)
    assert masks.sum() == 1 and masks[0, 0, 0]


# --------------------------------------------------------------------------------------------- AQuA2
def aqua_events(sc: dict, gap: int) -> tuple[list[np.ndarray], int, int, int]:
    """Two events on the territory and one elsewhere, as 1-based linear indices into the cropped volume."""
    h, w = sc["territory"].shape
    x, y, t = h - 2 * gap, w - 2 * gap, 30
    crop = sc["territory"][gap:h - gap, gap:w - gap]
    rows, cols = np.nonzero(crop)
    evts = []
    for frame in (3, 12):
        evts.append(rows + cols * x + frame * x * y + 1)
    evts.append(np.array([2 + 2 * x + 20 * x * y + 1]))
    return evts, x, y, t


def test_aqua2_v5(tmp_path, sc) -> None:
    from scipy.io import savemat
    evts, x, y, t = aqua_events(sc, 5)
    cell = np.empty(len(evts), dtype=object)
    for i, e in enumerate(evts):
        cell[i] = e.astype(float)
    path = tmp_path / "res_v5.mat"
    savemat(path, {"res_dbg": {"datOrg1": np.zeros((x, y, 1, t)), "evt1": cell, "opts": {"regMaskGap": 5}}})
    ev = aqua2.read_events(path)
    assert (ev.x, ev.y, ev.n_frames, ev.gap, len(ev.events)) == (x, y, t, 5, 3)
    terr = aqua2.territory(path)
    assert np.array_equal(terr, sc["territory"])                      # two events: territory; one: not
    assert not aqua2.territory(path, min_events=3).any()


def test_aqua2_v73(tmp_path, sc) -> None:
    import h5py
    evts, x, y, t = aqua_events(sc, 5)
    path = tmp_path / "res_v73.mat"
    with h5py.File(path, "w") as f:
        r = f.create_group("res")
        r["datOrg1"] = np.zeros((t, 1, y, x))                         # MATLAB [x, y, 1, T] as h5py reads it
        refs = f.create_group("#refs#")
        handles = []
        for i, e in enumerate(evts):
            refs[f"e{i}"] = e.astype(float)[:, None]
            handles.append(refs[f"e{i}"].ref)
        r.create_dataset("evt1", data=np.array(handles, dtype=h5py.ref_dtype)[:, None])
        r.create_group("opts")["regMaskGap"] = np.array([[5.0]])
    terr = aqua2.territory([path])
    assert np.array_equal(terr, sc["territory"])


def test_territory_from_aqua2(tmp_path, sc) -> None:
    from scipy.io import savemat
    evts, x, y, t = aqua_events(sc, 5)
    cell = np.empty(len(evts), dtype=object)
    for i, e in enumerate(evts):
        cell[i] = e.astype(float)
    path = tmp_path / "res.mat"
    savemat(path, {"res_dbg": {"datOrg1": np.zeros((x, y, 1, t)), "evt1": cell, "opts": {"regMaskGap": 5}}})
    res = astrid.territory_from_aqua2(sc["movie"], sc["rois"], path, fps=FPS, um_per_px=2.0, **KW)
    direct = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, territory=sc["territory"], **KW)
    assert np.array_equal(res.dff, direct.dff) and res.provenance["territory_px"] == int(sc["territory"].sum())
    piped = astrid.run_pipeline(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, aqua2_results=[path],
                                marker=[np.ones((40, 48)) + sc["territory"]], out=tmp_path / "out", **KW)
    assert (tmp_path / "out" / "result.npz").exists() and piped.provenance["evidence"] is True


# --------------------------------------------------------------------------------------------- CLI
def test_cli_demix(tmp_path, sc) -> None:
    np.save(tmp_path / "movie.npy", sc["movie"])
    np.savez(tmp_path / "rois.npz", rois=sc["rois"])
    labels = sc["rois"][0] * 1 + sc["rois"][1] * 2
    np.savez(tmp_path / "labels.npz", labels=labels)
    assert np.array_equal(load_masks(str(tmp_path / "labels.npz")), sc["rois"])
    np.savez(tmp_path / "terr.npz", territory=sc["territory"])
    out = tmp_path / "out"
    assert main(["demix", "--movie", str(tmp_path / "movie.npy"), "--rois", str(tmp_path / "rois.npz:rois"),
                 "--fps", str(FPS), "--um-per-px", "2", "--tau-r", "0.05", "--tau-d", "0.4", "--territory",
                 str(tmp_path / "terr.npz"), "--out", str(out)]) == 0
    direct = astrid.demix(sc["movie"], sc["rois"], fps=FPS, um_per_px=2.0, territory=sc["territory"], **KW)
    with np.load(out / "result.npz") as z:
        assert np.array_equal(z["dff__movie"], direct.dff)


def test_cli_pipeline(tmp_path, sc) -> None:
    np.save(tmp_path / "movie.npy", sc["movie"])
    np.savez(tmp_path / "rois.npz", rois=sc["rois"])
    (tmp_path / "loc.yaml").write_text("movies: [movie.npy]\nrois: rois.npz\nfps: 30\num_per_px: 2\n"
                                       "tau_r: 0.05\ntau_d: 0.4\nout: res\n", encoding="utf-8")
    assert main(["pipeline", str(tmp_path / "loc.yaml")]) == 0
    assert (tmp_path / "res" / "result.npz").exists()
