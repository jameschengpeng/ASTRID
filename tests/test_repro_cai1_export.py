"""The D3 export on a small synthetic dataset laid out as CRCNS distributes cai-1: TIFF chunks whose last frame
is missing, a processed file in the dataset's format, a hand-drawn ROI, and a known motion step to register."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

tifffile = pytest.importorskip("tifffile")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cai1_export as X  # noqa: E402
from astrid_repro import public_data as D  # noqa: E402

CHUNK = 50                       # frames per TIFF file here (the dataset logs 2,400)
SIZE = 96                        # the field, enough room for the 40-px shift search
FPS = 60.0
SHIFT = (3, -2)                  # the motion of frames 60 onwards


def _field(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:SIZE, :SIZE]
    soma = np.exp(-((yy - 48) ** 2 + (xx - 40) ** 2) / (2 * 6.0 ** 2))
    texture = np.clip(rng.normal(0, 1, (SIZE, SIZE)), -2, 2)
    return 300 + 200 * soma + 40 * texture, ((yy - 48) ** 2 + (xx - 40) ** 2) <= 36


def _processed(path: Path, fmean_roi: np.ndarray, t_frame: np.ndarray, spikes: np.ndarray,
               indicator: str) -> None:
    """A processed file in the dataset's format (``obj.timeSeriesArrayHash``)."""
    t_ephys = np.arange(0.1, 1.5, 1e-3)
    detected = np.zeros(t_ephys.size, dtype=np.uint8)
    detected[np.searchsorted(t_ephys, spikes)] = 1
    series = np.empty(5, dtype=object)
    for i, (value, time) in enumerate(((fmean_roi, t_frame), (fmean_roi * 0.5, t_frame),
                                       (np.zeros(t_ephys.size), t_ephys), (np.zeros(t_ephys.size), t_ephys),
                                       (detected, t_ephys))):
        series[i] = {"valueMatrix": value, "time": time}
    names = np.array(["fmean_roi", "fmean_neuropil", "raw_ephys", "filtered ephys", "detected_spikes"],
                     dtype=object)
    scipy.io.savemat(path, {"obj": {"timeSeriesArrayHash": {"value": series, "keyNames": names},
                                    "indicator": indicator}})


def _dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int = 2 * CHUNK,
             samples: int = 2 * CHUNK) -> dict[str, object]:
    monkeypatch.setattr(X, "FRAMES_PER_CHUNK", CHUNK)
    rng = np.random.default_rng(0)
    field, disk = _field(rng)
    activity = 1 + 0.5 * np.sin(np.arange(frames) / 5.0) ** 2
    movie = np.stack([field * (1 + (a - 1) * disk) for a in activity]) + rng.normal(0, 2, (frames, SIZE, SIZE))
    movie[60:] = np.roll(movie[60:], SHIFT, axis=(1, 2))
    movie = np.clip(movie, 0, 65535).astype(np.uint16)
    crcns = tmp_path / "crcns"
    raw = crcns / D.CAI1_FOLDERS["GCaMP6f"] / "raw_data" / "20120101_cell1"
    raw.mkdir(parents=True)
    for c in range(frames // CHUNK):                         # the last frame of each chunk is missing
        tifffile.imwrite(raw / f"cell1_001_{c + 1}.tif", movie[c * CHUNK:(c + 1) * CHUNK - 1])
    tifffile.imwrite(raw / "cell1_009_1.tif", movie[:5])     # no processed file: not a recording
    (raw / "notes.tif").write_bytes(b"")                    # not a chunk's name
    (crcns / D.CAI1_FOLDERS["GCaMP6f"] / "processed_data").mkdir(parents=True)
    (crcns / D.CAI1_FOLDERS["GCaMP6s"] / "raw_data").mkdir(parents=True)
    t_frame = np.arange(samples) / FPS
    roi_mean = movie[:samples, disk].astype(np.float64).mean(axis=1)
    spikes = np.array([0.3, 0.8, 1.2])
    _processed(D.cai1_processed_path(crcns, "GCaMP6f", "20120101_cell1", "001"), roi_mean, t_frame, spikes,
               "GCaMP6f")
    rois = tmp_path / "rois"
    rois.mkdir()
    np.savez(rois / "20120101_cell1.npz", roi=disk)
    return {"crcns": crcns, "rois": rois, "disk": disk, "movie": movie, "t_frame": t_frame, "spikes": spikes}


def test_inventory_and_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dataset(tmp_path, monkeypatch)
    recs = X.inventory(d["crcns"])
    assert [r.name for r in recs] == ["20120101_cell1_001"] and len(recs[0].tifs) == 2
    movie, present = X.read_movie(recs[0], 2 * CHUNK)
    assert not present[CHUNK - 1] and not present[2 * CHUNK - 1] and present.sum() == 2 * CHUNK - 2
    assert np.isnan(movie[CHUNK - 1]).all() and np.array_equal(movie[CHUNK], d["movie"][CHUNK])
    short, _ = X.read_movie(recs[0], 70)                    # processed traces shorter than the frames
    assert short.shape[0] == 70
    monkeypatch.setattr(X, "FRAMES_PER_CHUNK", 10)
    with pytest.raises(ValueError, match="more than"):
        X.read_movie(recs[0], 100)


def test_register_bin_and_hold() -> None:
    rng = np.random.default_rng(1)
    field, _ = _field(rng)
    movie = np.repeat(field[None], 120, axis=0).astype(np.float32) + rng.normal(0, 1, (120, SIZE, SIZE))
    movie[60:] = np.roll(movie[60:], SHIFT, axis=(1, 2))
    present = np.ones(120, dtype=bool)
    present[[10, 70]] = False
    movie[~present] = np.nan
    reg, shifts = X.register(movie.astype(np.float32), present)
    assert tuple(shifts[90]) == (-SHIFT[0], -SHIFT[1]) and tuple(shifts[20]) == (0, 0)
    assert np.isnan(reg[70]).all()
    inner = (slice(None), slice(10, -10), slice(10, -10))
    assert np.abs(reg[90][inner[1:]] - reg[20][inner[1:]]).mean() < 3.0     # aligned again
    m = np.zeros((8, 8), dtype=bool)                        # four blocks of 4 x 4 pixels
    m[0:4, 0:2] = True                                      # half of the first block: kept
    m[0:4, 4:8] = True                                      # all of the second
    m[4:8, 4] = True                                        # 7 of 16 pixels of the last: not kept
    m[4:7, 5] = True
    assert X.bin_mask(m).tolist() == [[True, True], [False, False]]
    b = X.bin_movie(np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8))
    assert b.shape == (2, 2, 2) and b[0, 0, 0] == np.arange(64).reshape(8, 8)[:4, :4].mean()
    held = X.hold_missing(np.arange(5.0)[:, None, None] * np.ones((5, 1, 1)), np.array([0, 1, 0, 1, 0], bool))
    assert held[:, 0, 0].tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]


def test_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _dataset(tmp_path, monkeypatch)
    out = tmp_path / "export"
    assert X.main(["--crcns", str(d["crcns"]), "--rois", str(d["rois"]), "--out", str(out)]) == 0
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [r["name"] for r in index] == ["20120101_cell1_001"]
    row = index[0]
    assert row["roi_r"] == pytest.approx(1.0) and row["frames"] == 2 * CHUNK and row["shape"] == [24, 24]
    with h5py.File(out / "20120101_cell1_001.h5", "r") as f:
        mov, valid, present = f["mov"][()], f["valid"][()], f["frame_present"][()]
        assert mov.dtype == np.int16 and mov.shape == (2 * CHUNK, 24, 24)
        assert np.array_equal(mov[CHUNK - 1], mov[CHUNK - 2])           # a missing frame holds the previous one
        t = d["t_frame"]
        assert np.array_equal(valid, present & (t >= 0.1) & (t <= 1.499))
        assert tuple(f["shifts"][()][80]) == (-SHIFT[0], -SHIFT[1])
        assert np.array_equal(f["roi"][()], X.bin_mask(d["disk"]))
        np.testing.assert_array_equal(f["predicted"][()], D.predicted_calcium(f["spike_times"][()], t, "GCaMP6f"))
        assert f.attrs["indicator"] == "GCaMP6f" and f.attrs["bin"] == 4 and f.attrs["max_shift_px"] == 3
    with np.load(out / "20120101_cell1_001_roi.npz") as z:
        assert np.array_equal(z["roi"], X.bin_mask(d["disk"]))
    np.savez(d["rois"] / "20120101_cell1_001.npz", roi=np.roll(d["disk"], 30, axis=1))   # the recording's own
    with pytest.raises(ValueError, match="not their cell"):
        X.export(d["crcns"], d["rois"], tmp_path / "again")
