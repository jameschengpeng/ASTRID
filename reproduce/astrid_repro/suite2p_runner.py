"""Suite2p's neuropil subtraction on given ROIs, as it ran on every dataset of the paper.

Suite2p (1.1.0 in the paper, GPL-3) and PyTorch are not part of this repository. Install them in an environment
of their own and run this module there; it imports them only when it runs. Only Suite2p's own steps run, at
its shipped extraction defaults (``default_settings()``) but with ``allow_overlap`` True, so that a pixel
shared by ROIs is used for each and no given ROI is left without pixels:
1. each mask becomes a Suite2p ROI, a ``stat`` with its pixels (``ypix``, ``xpix``) and a uniform ``lam``, as
   for ROIs drawn in Suite2p's GUI;
2. Suite2p's ``roi_stats`` adds the statistics with its ROI removal off (``max_overlap=None``), so every given
   ROI is kept, and ``assign_overlaps`` marks the pixels shared by ROIs;
3. Suite2p's ``extraction_wrapper`` makes the cell and neuropil masks and extracts F and Fneu;
4. the trace is F - ``neuropil_coefficient`` (0.7) x Fneu, as Suite2p's pipeline subtracts it.
``torch_device`` is "cpu", with ``--threads`` threads for PyTorch. The frames are passed in memory as float32.

**Modes and inputs:**
- ``video``: one movie (``--movie file.h5[:key]``, (T, H, W), key ``mov`` by default) and its masks
  (``--masks``: ``file.mat[:key]``, (H, W, n) read as TUnCaT reads its ``FinalMasks``, transposed to (n, W, H);
  or ``file.npz[:key]``, (n, H, W) bool or an (H, W) label image);
- ``cai1``: single-ROI recordings (``--data <folder>``, ``<name>.h5`` with ``mov`` (T, H, W) and ``roi``
  (H, W)), as D3 was exported;
- ``cases``: one recording's exported cases (``--cases``: the folder of ``astrid_repro.peers``' case format),
  each case's movie built by ``peers.case_movie``, with the target and the neighbouring cells as plain TUnCaT
  and CNMF get them. A case on which Suite2p's own steps raise (``Suite2pError``) gets no results file, so it
  is scored as a failure, and is listed with its error under ``failed`` in the provenance. Such a case can
  occur with a compact ROI of about 11-13 px and no neighbouring cell: Suite2p 1.1.0's soma crop then leaves the
  ROI a radius of 0, and its percentile filter fails.

**Outputs** in ``--out``, per video (``video``, ``cai1``) or per recording (``cases``, named after the cases
folder's parent):
- ``<name>.npz``: F, Fneu and the trace, (n, T) float32, in the masks' order (``video``, ``cai1``);
  ``cases`` writes ``<results>/<case>.npz`` instead, with ``suite2p``, the target's trace (as
  ``astrid_repro.peers`` writes its results);
- ``<name>_seconds.json``: the wall-clock and CPU time of Suite2p's work (steps 2-4; reading the movie is not
  counted) and the threads;
- ``<name>.json``: the provenance (versions, the settings used, the hashes of the masks and the movie).

Run (in Suite2p's environment):
    python -I suite2p_runner.py video --movie M.h5:mov --masks FinalMasks_V.mat --name V --threads 8 --out O
    python -I suite2p_runner.py cai1 --data <folder> --threads 8 --out O <name> [<name> ...]
    python -I suite2p_runner.py cases --cases <recording>/compare --results O/results --threads 8 --out O
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def array_sha256(a: np.ndarray) -> str:
    """sha256 of an array's dtype, shape and bytes (masks as bool)."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}{a.shape}".encode() + a.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def split_key(spec: str) -> tuple[str, str | None]:
    """"path[:key]", a Windows drive letter allowed."""
    drive = len(spec) > 1 and spec[1] == ":"
    return tuple(spec.rsplit(":", 1)) if spec.count(":") > (1 if drive else 0) else (spec, None)  # type: ignore[return-value]


def load_masks(spec: str) -> np.ndarray:
    """(n, ., .) bool: a FinalMasks .mat (H, W, n) transposed as TUnCaT reads it, or an .npz array (or label
    image)."""
    path, key = split_key(spec)
    p = Path(path)
    if p.suffix == ".mat":
        try:
            from scipy.io import loadmat
            arr = loadmat(p)[key or "FinalMasks"]
            arr = np.asarray(arr).transpose(2, 1, 0) if arr.ndim == 3 else np.asarray(arr)
        except NotImplementedError:
            with h5py.File(p, "r") as f:
                arr = np.asarray(f[key or "FinalMasks"])
    else:
        with np.load(p) as z:
            arr = z[key or z.files[0]]
    arr = np.asarray(arr)
    if arr.ndim == 2:
        n = int(arr.max())
        return np.stack([arr == v for v in range(1, n + 1)]) if n else np.zeros((0, *arr.shape), bool)
    return arr.astype(bool)


def versions() -> dict[str, str]:
    import importlib.metadata as md
    return {p: md.version(p) for p in ("suite2p", "torch", "numpy", "scipy")}


def suite2p_settings() -> dict[str, Any]:
    from suite2p.parameters import default_settings  # type: ignore[import-not-found]
    s = default_settings()
    return {"extraction": {**s["extraction"], "allow_overlap": True}, "diameter": list(s["diameter"]),
            "torch_device": "cpu", "torch_device_default": s["torch_device"],
            "allow_overlap_default": s["extraction"]["allow_overlap"]}


class Suite2pError(RuntimeError):
    """Raised inside Suite2p's own steps (``roi_stats``, ``assign_overlaps``, ``extraction_wrapper``)."""


def neuropil_subtraction(movie: np.ndarray, masks: np.ndarray, settings: dict[str, Any],
                         threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Steps 2-4 of the module docstring on ``movie`` (T, H, W) float32 and ``masks`` (n, H, W) bool; returns
    F, Fneu, the trace and the times (wall and CPU, of these steps only). An error inside Suite2p's own steps
    is raised as ``Suite2pError``; this wrapper's own checks raise their own errors."""
    import torch  # type: ignore[import-not-found]
    from suite2p.detection.stats import assign_overlaps, roi_stats  # type: ignore[import-not-found]
    from suite2p.extraction.extract import extraction_wrapper  # type: ignore[import-not-found]

    torch.set_num_threads(threads)
    _, ly, lx = movie.shape
    if any(not m.any() for m in masks):
        raise ValueError("an empty mask")
    t0, c0 = time.perf_counter(), time.process_time()
    rois = []
    for m in masks:
        ypix, xpix = np.nonzero(m)
        rois.append({"ypix": ypix, "xpix": xpix, "lam": np.ones(ypix.size, np.float32)})
    try:
        stat = roi_stats(np.array(rois), ly, lx, diameter=settings["diameter"], max_overlap=None)
    except Exception as err:
        raise Suite2pError(f"roi_stats: {type(err).__name__}: {err}") from err
    if len(stat) != len(masks):
        raise RuntimeError(f"roi_stats kept {len(stat)} of {len(masks)} ROIs")
    try:
        stat = assign_overlaps(stat, ly, lx)
        f, fneu, _, _ = extraction_wrapper(stat, movie, settings=settings["extraction"],
                                           device=torch.device("cpu"))
    except Exception as err:
        raise Suite2pError(f"extraction: {type(err).__name__}: {err}") from err
    trace = f - settings["extraction"]["neuropil_coefficient"] * fneu
    times = {"wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0, "threads": threads}
    return f, fneu, trace, times


def write(out: Path, name: str, times: dict[str, Any], prov: dict[str, Any],
          arrays: dict[str, np.ndarray] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if arrays is not None:
        np.savez_compressed(out / f"{name}.npz", **arrays)
    (out / f"{name}_seconds.json").write_text(json.dumps(times, indent=1), encoding="utf-8")
    (out / f"{name}.json").write_text(json.dumps(prov, indent=1, default=str), encoding="utf-8")


def run_one(name: str, movie: np.ndarray, masks: np.ndarray, out: Path, threads: int,
            prov: dict[str, Any]) -> None:
    settings = suite2p_settings()
    f, fneu, trace, times = neuropil_subtraction(movie, masks, settings, threads)
    prov = {**prov, "versions": versions(), "settings": settings, "rois": int(len(masks)),
            "masks_sha256": array_sha256(masks), "frames": int(movie.shape[0]),
            "shape": list(movie.shape[1:])}
    write(out, name, times, prov, {"F": f, "Fneu": fneu, "trace": trace})
    print(f"{name}: {len(masks)} ROIs, {times['wall_s']:.2f} s wall, {times['cpu_s']:.2f} s CPU", flush=True)


def run_cases(cases: Path, results: Path, out: Path, threads: int) -> dict[str, str]:
    """The ``cases`` mode (module docstring); returns the failed cases with their errors."""
    if __package__ in (None, ""):                       # run as a script: reproduce/ holds astrid_repro
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from astrid_repro.peers import base_of, case_movie  # the peers' own case builder

    settings = suite2p_settings()
    index = json.loads((cases / "index.json").read_text(encoding="utf-8"))
    results.mkdir(parents=True, exist_ok=True)
    masks_by_base: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    wall = cpu = 0.0
    for e in index:
        case = e["spec"]
        movie, m = case_movie(cases, case)
        masks = np.concatenate([m["target"].astype(bool)[None], m["neighbours"].astype(bool)])
        frames = np.ascontiguousarray(movie, np.float32)
        try:
            _, _, trace, times = neuropil_subtraction(frames, masks, settings, threads)
        except Suite2pError as err:                 # no results file: scored as a failure
            failed[case] = str(err)
            print(f"{case}: Suite2p failed, no trace ({err})", flush=True)
            continue
        wall, cpu = wall + times["wall_s"], cpu + times["cpu_s"]
        np.savez_compressed(results / f"{case}.npz", suite2p=np.asarray(trace[0], dtype=np.float64))
        base = base_of(case)
        masks_by_base.setdefault(base, {"rois": int(len(masks)), "masks_sha256": array_sha256(masks),
                                        "masks_file_sha256": file_sha256(cases / f"{base}_masks.npz")})
    name = cases.parent.name
    write(out, name, {"wall_s": wall, "cpu_s": cpu, "threads": threads, "cases": len(index)},
          {"export": str(cases), "versions": versions(), "settings": settings, "cases": len(index),
           "bases": masks_by_base, "failed": failed}, None)
    print(f"{name}: {len(index)} cases ({len(failed)} failed), {wall:.1f} s wall", flush=True)
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("video", help="one movie with its masks")
    v.add_argument("--movie", required=True, help="file.h5[:key] (default key mov), (T, H, W)")
    v.add_argument("--masks", required=True, help="file.mat[:key] or file.npz[:key]")
    v.add_argument("--name", required=True)
    c = sub.add_parser("cai1", help="single-ROI recordings <data>/<name>.h5 (mov, roi)")
    c.add_argument("names", nargs="+")
    c.add_argument("--data", type=Path, required=True)
    d = sub.add_parser("cases", help="one recording's exported cases (astrid_repro.peers' format)")
    d.add_argument("--cases", type=Path, required=True, help="the cases folder (index.json)")
    d.add_argument("--results", type=Path, required=True, help="where <case>.npz go")
    for p in (v, c, d):
        p.add_argument("--threads", type=int, required=True)
        p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.mode == "video":
        path, key = split_key(a.movie)
        with h5py.File(path, "r") as f:
            movie = np.asarray(f[key or "mov"], dtype=np.float32)
        masks = load_masks(a.masks)
        mpath = Path(split_key(a.masks)[0])
        run_one(a.name, movie, masks, a.out, a.threads,
                {"movie": a.movie, "movie_sha256": file_sha256(Path(path)), "masks": a.masks,
                 "masks_file_sha256": file_sha256(mpath)})
    elif a.mode == "cai1":
        for name in a.names:
            h5 = a.data / f"{name}.h5"
            with h5py.File(h5, "r") as f:
                movie, roi = np.asarray(f["mov"], dtype=np.float32), np.asarray(f["roi"], dtype=bool)
            run_one(name, movie, roi[None], a.out, a.threads,
                    {"movie": str(h5), "movie_sha256": file_sha256(h5)})
    else:
        run_cases(a.cases, a.results, a.out, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
