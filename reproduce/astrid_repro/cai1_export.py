"""D3, the CRCNS cai-1 dataset (Chen et al. 2013), exported as every method read it: per recording, the movie
registered and binned 4 x 4, the cell's hand-drawn ROI, the scored frames, the recorded spikes and the calcium
predicted from them, and the dataset authors' own ROI and neuropil traces.

**Inputs.** The dataset as CRCNS distributes it (https://crcns.org, cai-1; a free account), unpacked under one
root as ``GCaMP6f_11cells_Chen2013`` and ``GCaMP6s_9cells_Chen2013``, each with
``raw_data/<date>_cell<k>/cell<k>_<nnn>_<c>.tif`` (ScanImage stacks, 256 x 256, logged 2,400 frames per file)
and ``processed_data/data_<date>_cell<k>_<nnn>.mat`` (the authors' ROI and neuropil traces, the frame times and
the electrophysiology). The ROIs are this repository's ``reproduce/cai1_rois`` (``<cell>.npz``, key ``roi``):
the dataset ships none, so the ROI of each cell was drawn by hand once, on the cell's mean image, blind to
every method.

**Readings of the data**, fixed before any fit:
1. *Frame alignment.* Frame ``i`` of chunk ``c`` (0-based) is sample ``2400 c + i`` of the processed traces.
   Most chunks hold 2,399 frames, the missing one being the last of each chunk. The movie is assembled on the
   traces' grid, the missing frames left as NaN (``frame_present`` False); in the exported movie they hold the
   previous frame.
2. *Truncation.* A recording whose processed traces are shorter than its frames is cut to them; a chunk with
   more than 2,400 frames is refused.
3. *The ROI.* The cell's mask (or the recording's own, ``<cell>_<rec>.npz``, if one exists), checked on the
   unregistered movie: its mean must correlate at least ``ROI_IDENTITY_R`` with the authors' ``fmean_roi``,
   else the recording is refused.
4. *Registration*, rigid: per block of ``REGISTER_BLOCK`` frames, the integer shift (within ``REGISTER_MAX_PX``)
   that best aligns the block's mean with the reference (FFT cross-correlation), the reference being the mean of
   the first ``REGISTER_REF_FRAMES`` frames; the block shifts median-filtered over ``REGISTER_MEDIAN_BLOCKS``
   blocks; a second pass against the registered mean. Pixels shifted in from outside the field take the
   reference's value.
5. *Scored frames* (``valid``): present, and within the electrophysiology.
6. *Binning* 4 x 4 (256 to 64 px, about 0.47 um per binned pixel, an estimate: the pixel size is not in the
   files); the ROI keeps a binned pixel if at least half of its block is in the mask.
The ground truth is ``public_data.predicted_calcium`` of the spikes in the processed files.

**Output**, per recording ``<out>/<cell>_<rec>.h5``: ``mov`` (T, 64, 64) int16, ``valid``, ``frame_present``,
``shifts`` (T, 2), ``roi``, ``t_frame``, ``spike_times``, ``predicted``, ``fmean_roi``, ``fmean_neuropil`` and
the attributes ``fps``, ``indicator``, ``cell``, ``rec``, ``roi_r`` (reading 3's check), ``max_shift_px``,
``ephys_start_s``, ``ephys_end_s``, ``bin``; ``<cell>_<rec>_roi.npz`` holds the binned ROI again; ``index.json``
lists the recordings.

    python -m astrid_repro.cai1_export --crcns <cai-1 root> --rois cai1_rois --out <export> [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from astrid_repro import public_data as D

FRAMES_PER_CHUNK = 2400                # ScanImage loggingFramesPerFile
ROI_IDENTITY_R = 0.95
REGISTER_BLOCK = 30                    # frames per shift estimate (0.5 s at 60 Hz)
REGISTER_MAX_PX = 40                   # the largest shift searched
REGISTER_REF_FRAMES = 600              # the reference: the mean of the first 10 s
REGISTER_MEDIAN_BLOCKS = 5             # the block shifts are median-filtered over this many blocks (2.5 s)
BIN = 4


@dataclass(frozen=True)
class Recording:
    indicator: str
    cell: str                          # <date>_cell<k>
    rec: str                           # <nnn>
    mat: Path
    tifs: tuple[Path, ...]             # the chunks in order

    @property
    def name(self) -> str:
        return f"{self.cell}_{self.rec}"


def inventory(root: str | Path) -> list[Recording]:
    """Every recording with a processed file, GCaMP6f's cells first, in name order."""
    root = Path(root)
    out = []
    for ind, folder in D.CAI1_FOLDERS.items():
        for cell in sorted((root / folder / "raw_data").iterdir()):
            if not cell.is_dir():
                continue
            chunks: dict[str, list[Path]] = {}
            for f in sorted(cell.glob("*.tif")):
                m = re.match(r"cell(\d+)_(\d+)_(\d+)\.tif$", f.name)
                if m is not None:
                    chunks.setdefault(m.group(2), []).append(f)
            for rec, files in sorted(chunks.items()):
                mat = D.cai1_processed_path(root, ind, cell.name, rec)
                if mat.exists():
                    out.append(Recording(ind, cell.name, rec, mat, tuple(files)))
    return out


def read_chunk(path: Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as t:
        return np.asarray(t.series[0].asarray())


def read_movie(r: Recording, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """(movie (T, H, W) float32 on the processed traces' grid, frame_present (T,)): readings 1 and 2."""
    import tifffile

    counts = []
    for f in r.tifs:
        with tifffile.TiffFile(f) as t:
            counts.append(len(t.pages))
    with tifffile.TiffFile(r.tifs[0]) as t:
        h, w = t.pages[0].shape
    movie = np.full((n_samples, h, w), np.nan, dtype=np.float32)
    present = np.zeros(n_samples, dtype=bool)
    for c, (f, n) in enumerate(zip(r.tifs, counts, strict=True)):
        start = FRAMES_PER_CHUNK * c
        if start >= n_samples:
            break
        if n > FRAMES_PER_CHUNK:
            raise ValueError(f"{f}: {n} frames in one chunk, more than {FRAMES_PER_CHUNK}")
        take = min(n, n_samples - start)
        movie[start:start + take] = read_chunk(f)[:take]
        present[start:start + take] = True
    return movie, present


def _xcorr_shift(block_mean: np.ndarray, ref: np.ndarray, max_px: int) -> tuple[int, int]:
    """The integer (dy, dx) that moves ``block_mean`` onto ``ref`` (FFT cross-correlation, within max_px)."""
    a = block_mean - block_mean.mean()
    b = ref - ref.mean()
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a)))))
    cy, cx = c.shape[0] // 2, c.shape[1] // 2
    win = c[cy - max_px:cy + max_px + 1, cx - max_px:cx + max_px + 1]
    dy, dx = np.unravel_index(int(win.argmax()), win.shape)
    return int(dy - max_px), int(dx - max_px)


def register(movie: np.ndarray, present: np.ndarray, block: int = REGISTER_BLOCK, max_px: int = REGISTER_MAX_PX,
             passes: int = 2, ref_frames: int = REGISTER_REF_FRAMES,
             median_blocks: int = REGISTER_MEDIAN_BLOCKS) -> tuple[np.ndarray, np.ndarray]:
    """Rigid registration (reading 4): returns (registered movie, shifts (T, 2) int). Missing frames stay NaN."""
    from scipy.ndimage import median_filter

    t = movie.shape[0]
    out = movie.copy()
    shifts = np.zeros((t, 2), dtype=int)
    first = slice(0, min(ref_frames, t))
    ref = np.nanmean(movie[first][present[first]], axis=0)
    starts = list(range(0, t, block))
    for _ in range(passes):
        est = np.zeros((len(starts), 2), dtype=int)
        has = np.zeros(len(starts), dtype=bool)
        for i, s in enumerate(starts):
            seg = slice(s, min(s + block, t))
            ok = present[seg]
            if ok.any():
                est[i] = _xcorr_shift(np.nanmean(movie[seg][ok], axis=0), ref, max_px)
                has[i] = True
        if median_blocks > 1 and has.sum() > median_blocks:
            est[has] = median_filter(est[has], size=(median_blocks, 1), mode="nearest")
        out = movie.copy()
        for i, s in enumerate(starts):
            seg = slice(s, min(s + block, t))
            ok = present[seg]
            if not has[i]:
                continue
            dy, dx = int(est[i, 0]), int(est[i, 1])
            shifts[seg] = (dy, dx)
            if dy == 0 and dx == 0:
                continue
            rolled = np.roll(movie[seg], (dy, dx), axis=(1, 2))
            fill = np.zeros(movie.shape[1:], dtype=bool)
            if dy > 0:
                fill[:dy] = True
            elif dy < 0:
                fill[dy:] = True
            if dx > 0:
                fill[:, :dx] = True
            elif dx < 0:
                fill[:, dx:] = True
            rolled[:, fill] = ref[fill].astype(movie.dtype)
            rolled[~ok] = np.nan                               # missing frames stay missing
            out[seg] = rolled
        ref = np.nanmean(out[present], axis=0)
    return out, shifts


def bin_movie(movie: np.ndarray, b: int = BIN) -> np.ndarray:
    """The mean over b x b pixel blocks, frame by frame (NaN frames stay NaN), the field cropped to a multiple
    of b."""
    t, h, w = movie.shape
    h2, w2 = h // b * b, w // b * b
    return movie[:, :h2, :w2].reshape(t, h2 // b, b, w2 // b, b).mean(axis=(2, 4))


def bin_mask(mask: np.ndarray, b: int = BIN, frac: float = 0.5) -> np.ndarray:
    """A binned pixel belongs to the mask if at least ``frac`` of its b x b block does."""
    h, w = mask.shape
    h2, w2 = h // b * b, w // b * b
    return mask[:h2, :w2].reshape(h2 // b, b, w2 // b, b).mean(axis=(1, 3)) >= frac


def hold_missing(movie: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Missing frames take the previous present frame (the first present one if none precedes)."""
    out = movie.copy()
    last = None
    first = int(np.flatnonzero(present)[0])
    for i in range(out.shape[0]):
        if present[i]:
            last = i
        else:
            out[i] = out[last if last is not None else first]
    return out


def roi_path(r: Recording, rois: str | Path) -> Path:
    """The recording's own ROI if there is one, else the cell's."""
    own = Path(rois) / f"{r.cell}_{r.rec}.npz"
    return own if own.exists() else Path(rois) / f"{r.cell}.npz"


def read_roi(r: Recording, rois: str | Path) -> np.ndarray:
    p = roi_path(r, rois)
    if not p.exists():
        raise FileNotFoundError(f"{r.name}: no ROI ({p})")
    with np.load(p) as z:
        return np.asarray(z["roi"], dtype=bool)


def roi_check(movie: np.ndarray, present: np.ndarray, roi: np.ndarray, fmean_roi: np.ndarray) -> float:
    """The correlation of the ROI's mean with the authors' ``fmean_roi`` on the present frames (reading 3)."""
    ours = movie[:, roi].mean(axis=1)
    ok = present & np.isfinite(ours)
    return float(np.corrcoef(ours[ok], fmean_roi[:len(ours)][ok])[0, 1])


def export_recording(r: Recording, rois: str | Path, out: str | Path) -> dict[str, Any]:
    """One recording's export (module docstring); returns its row of ``index.json``."""
    out = Path(out)
    s = D.read_cai1_processed(r.mat, r.indicator)
    if s["indicator"] != r.indicator:
        raise ValueError(f"{r.name}: the file says {s['indicator']}, the folder {r.indicator}")
    t_frame = s["t_frame"]
    movie, present = read_movie(r, len(t_frame))
    roi_full = read_roi(r, rois)
    roi_r = roi_check(movie, present, roi_full, s["fmean_roi"])      # on the unregistered movie, as theirs
    if roi_r < ROI_IDENTITY_R:
        raise ValueError(f"{r.name}: the ROI's mean correlates {roi_r:.3f} with the authors' fmean_roi "
                         f"(< {ROI_IDENTITY_R}): not their cell")
    movie, shifts = register(movie, present)
    fps = 1 / float(np.median(np.diff(t_frame)))
    predicted = D.predicted_calcium(s["spike_times"], t_frame, r.indicator)
    mov = hold_missing(bin_movie(movie), present)
    del movie
    roi = bin_mask(roi_full)
    te = s["t_ephys"]
    valid = present & (t_frame >= te[0]) & (t_frame <= te[-1])
    max_shift = int(np.abs(shifts).max())
    with h5py.File(out / f"{r.name}.h5", "w") as f:
        f.create_dataset("mov", data=np.rint(mov).astype(np.int16), chunks=(256, *mov.shape[1:]),
                         compression="gzip", compression_opts=1)
        f.create_dataset("valid", data=valid)
        f.create_dataset("frame_present", data=present)
        f.create_dataset("shifts", data=shifts)
        f.create_dataset("roi", data=roi)
        f.create_dataset("t_frame", data=t_frame)
        f.create_dataset("spike_times", data=s["spike_times"])
        f.create_dataset("predicted", data=predicted)
        f.create_dataset("fmean_roi", data=s["fmean_roi"])
        f.create_dataset("fmean_neuropil", data=s["fmean_neuropil"])
        f.attrs.update({"fps": fps, "indicator": r.indicator, "cell": r.cell, "rec": r.rec, "roi_r": roi_r,
                        "max_shift_px": max_shift, "ephys_start_s": float(te[0]), "ephys_end_s": float(te[-1]),
                        "bin": BIN})
    np.savez_compressed(out / f"{r.name}_roi.npz", roi=roi)
    return {"name": r.name, "cell": r.cell, "rec": r.rec, "indicator": r.indicator, "fps": fps,
            "frames": int(mov.shape[0]), "valid": int(valid.sum()), "ephys_end_s": float(te[-1]),
            "roi_r": roi_r, "roi_px": int(roi.sum()), "shape": list(mov.shape[1:]), "max_shift_px": max_shift,
            "spikes": int(s["spike_times"].size)}


def _task(args: tuple[Recording, str, str]) -> dict[str, Any]:
    return export_recording(*args)


def export(crcns: str | Path, rois: str | Path, out: str | Path, only: list[str] | None = None,
           workers: int = 1) -> list[dict[str, Any]]:
    """Every recording of the dataset (or ``only`` these names), ``workers`` at a time (each holds its movie,
    about 4 GB, a few times over); writes ``index.json`` in the inventory's order."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in inventory(crcns) if not only or r.name in only]
    tasks = [(r, str(rois), str(out)) for r in todo]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            index = list(ex.map(_task, tasks))
    else:
        index = [_task(t) for t in tasks]
    for row in index:
        print(row, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crcns", type=Path, required=True, help="the unpacked cai-1 dataset")
    ap.add_argument("--rois", type=Path, required=True, help="the hand-drawn ROIs (reproduce/cai1_rois)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="recording names (default: all)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    export(a.crcns, a.rois, a.out, a.only, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
