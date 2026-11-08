"""D2's movies: TUnCaT's crop of the ten Allen Brain Observatory experiments (Bao et al. 2022, its
``create_h5_video_ABO.m`` and ``prepareAllen.m``), the movies that every method of the paper read.

Each experiment's motion-corrected movie is public (the Allen Brain Observatory on AWS, ``URL``: dataset
``data``, (T, 512, 512) uint16, about 60 GB each). TUnCaT's archive holds the masks and transient labels of the
crop but not the movies, so they are made here: the first ``int(0.2 T)`` frames and the central 200 x 200 pixels
(rows and columns 156 to 355), written as ``<out>/<experiment>.h5`` with the dataset ``mov`` (uint16, one chunk
per frame), beside a record (``<experiment>.json``: the source's size and a checksum of its first and last MB,
its shape, the frames, rows and columns kept). Put them in ``<archive>/data/ABO/`` to read the dataset in
TUnCaT's archive layout.

    python -m astrid_repro.abo_crop --urls                          # the ten movies to download
    python -m astrid_repro.abo_crop --full <folder> --out <archive>/data/ABO [--experiments ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py

from astrid_repro.public_data import ABO_VIDEOS

URL = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-2p/ophys_movies/ophys_experiment_{}.h5"
FRACTION, PX = 0.2, 200                 # TUnCaT's rule: the first 20 % of frames, the central 200 x 200
BLOCK = 2000                            # frames copied at a time


def window(shape: tuple[int, int, int], fraction: float = FRACTION, px: int = PX) -> tuple[int, int, int]:
    """(frames kept, first row, first column) of a (T, H, W) movie."""
    t, h, w = shape
    return int(t * fraction), (h - px) // 2, (w - px) // 2


def crop(src: str | Path, dst: str | Path, experiment: str) -> dict[str, Any]:
    """One experiment's crop (module docstring); returns its record."""
    src, dst = Path(src), Path(dst)
    t0 = time.time()
    with h5py.File(src, "r") as f:
        d = f["data"]
        n, r0, c0 = window(d.shape)
        tmp = dst.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as g:
            mov = g.create_dataset("mov", shape=(n, PX, PX), dtype=d.dtype, chunks=(1, PX, PX))
            for s in range(0, n, BLOCK):
                b = d[s:min(s + BLOCK, n), r0:r0 + PX, c0:c0 + PX]
                mov[s:s + b.shape[0]] = b
        shape, dtype = [int(x) for x in d.shape], str(d.dtype)
    tmp.replace(dst)
    h = hashlib.sha256()
    with open(src, "rb") as fh:
        h.update(fh.read(1 << 20))
        fh.seek(-(1 << 20), 2)
        h.update(fh.read(1 << 20))
    rec = {"experiment": experiment, "source": src.name, "source_bytes": src.stat().st_size,
           "source_head_tail_sha256": h.hexdigest(), "source_shape": shape, "frames": [0, n],
           "rows": [r0, r0 + PX], "cols": [c0, c0 + PX],
           "rule": "TUnCaT create_h5_video_ABO.m: first 20 % of frames, central 200 x 200", "dtype": dtype,
           "seconds": time.time() - t0}
    dst.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--urls", action="store_true", help="print the URLs of the ten full movies and stop")
    ap.add_argument("--full", type=Path, help="the folder holding ophys_experiment_<id>.h5")
    ap.add_argument("--out", type=Path, help="where the crops go (<archive>/data/ABO)")
    ap.add_argument("--experiments", nargs="*", default=list(ABO_VIDEOS))
    a = ap.parse_args(argv)
    if a.urls:
        print("\n".join(URL.format(e) for e in a.experiments))
        return 0
    if a.full is None or a.out is None:
        ap.error("--full and --out are needed (or --urls)")
    a.out.mkdir(parents=True, exist_ok=True)
    for e in a.experiments:
        print(json.dumps(crop(a.full / f"ophys_experiment_{e}.h5", a.out / f"{e}.h5", e)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
