"""The peer methods on D3 (CRCNS cai-1), with their authors' code: FISSA, TUnCaT and CNMF (CaImAn), on the
recordings of the export (``cai1_export``), as they ran for the paper.

FISSA, TUnCaT and CaImAn are GPL and are not part of this repository. Install them in an environment of their
own and run this module there (in the paper: FISSA 1.0.0 and TUnCaT, its v1.0 scripts packaged as 1.1.1, in one
environment; CaImAn 1.13.2 in its own). It imports none of them until a method needs it, and nothing from
``astrid``, so it can also be copied and run on its own.

Per recording ``<data>/<name>.h5`` (``mov`` (T, H, W) int16, registered; ``roi`` (H, W) bool), the methods:
- ``fissa``: FISSA on the ROI (4 neuropil sub-regions, its defaults) over the alpha grid of its authors' scripts
  on the Allen data; ``fissa_a=<alpha>`` is the separated signal matched to the ROI, ``fissa_raw`` FISSA's raw ROI
  trace;
- ``tuncat``: TUnCaT on the single mask over its 13-value alpha grid, one call per alpha (flexible alpha,
  tolerance 1e-4, at most 20,000 iterations, the values of its v1.0 scripts); ``tuncat_a=<alpha>``, with
  ``tuncat_raw`` and ``tuncat_bg``;
- ``cnmf``: CaImAn's CNMF seeded with the ROI, as its seeded script for the Allen data ran it (p = 1, one
  background component, merge_thresh 1, OASIS); ``cnmf`` = C + YrA of the seeded component, ``cnmf_C`` its C.
  ``--cnmf-gnb`` and ``--cnmf-p`` set the number of background components and the AR order instead, for the
  cross-validation of CNMF's settings (gnb 1, 2, 3 × p 1, 2; the defaults, 1 and 1, are the published setting).
  CaImAn writes its memory-mapped movie to ``$CAIMAN_TEMP``, else to ``~/caiman_data/temp``, whatever folder it is
  given: ``CAIMAN_TEMP`` is pointed to a temporary folder for each fit, deleted when the fit ends.
Saves ``<out>/<name>.npz`` (a rerun of some methods merges into it), the file ``tables_public cai1 --peers``
reads, and ``<out>/<name>_seconds.json``, each call's own time with the movie in memory. The ring subtraction
and the raw ROI mean need no run (``ring.cai1_reference_rows``).

    python -m astrid_repro.cai1_peers --data <export> --out <peers> --methods fissa,tuncat <name> [<name> ...]
    python -m astrid_repro.cai1_peers --data <export> --out <peers> --methods cnmf <name> ...   (CaImAn's env)
    python -m astrid_repro.cai1_peers --data <export> --out <grid>/gnb<g>_p<p>/cai1 --methods cnmf
        --cnmf-gnb <g> --cnmf-p <p> <name> ...      (CNMF's grid, read by ``tables_public cai1 --cnmf-grid <grid>``)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FISSA_ALPHAS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)                  # FISSA_ABO.py
TUNCAT_ALPHAS = (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100)            # TUnCaT_multi_ABO.py
TUNCAT: dict[str, Any] = dict(Qclip=0, th_pertmin=1, epsilon=0, th_residual=0, nbin=1,
                              bin_option="downsample", multi_alpha=True, flexible_alpha=True, tol=1e-4,
                              max_iter=20000)
METHODS = ("fissa", "tuncat", "cnmf")
SECONDS: dict[str, float] = {}       # this recording's call times (timing only)


def read(data: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(data / f"{name}.h5", "r") as f:
        return np.asarray(f["mov"], dtype=np.float32), np.asarray(f["roi"], dtype=bool)


def save(path: Path, res: dict[str, np.ndarray]) -> None:
    old: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as z:
            old = {k: z[k] for k in z.files}
    old.update(res)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **old)


def run_fissa(movie: np.ndarray, roi: np.ndarray, separation_cores: int | None = 1) -> dict[str, np.ndarray]:
    import fissa  # type: ignore[import-not-found]

    out: dict[str, np.ndarray] = {}
    for a in FISSA_ALPHAS:
        t = time.perf_counter()
        exp = fissa.Experiment([movie], [[roi]], alpha=a, ncores_preparation=1,
                               ncores_separation=separation_cores, verbosity=0)
        exp.separate()
        SECONDS[f"fissa_a={a:g}"] = time.perf_counter() - t
        out[f"fissa_a={a:g}"] = np.asarray(exp.result[0, 0][0], dtype=np.float64)
        if "fissa_raw" not in out:
            out["fissa_raw"] = np.asarray(exp.raw[0, 0][0], dtype=np.float64)
    return out


def run_tuncat(name: str, movie: np.ndarray, roi: np.ndarray) -> dict[str, np.ndarray]:
    from tuncat.run_TUnCaT import run_TUnCaT  # type: ignore[import-not-found]

    out: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for a in TUNCAT_ALPHAS:
            t = time.perf_counter()
            r = run_TUnCaT(f"{name}_a{a:g}", movie, roi[None], tmp, list_alpha=[a], **TUNCAT)
            SECONDS[f"tuncat_a={a:g}"] = time.perf_counter() - t
            out[f"tuncat_a={a:g}"] = np.asarray(r[0][:, 0], dtype=np.float64)
            if "tuncat_raw" not in out:
                out["tuncat_raw"] = np.asarray(r[2][:, 0], dtype=np.float64)
                out["tuncat_bg"] = np.asarray(r[3][:, 0], dtype=np.float64)
    return out


def run_cnmf(name: str, data: Path, roi: np.ndarray, n_processes: int, p: int = 1,
             gnb: int = 1) -> dict[str, np.ndarray]:
    """CaImAn seeded with the ROI, as CNMF_unmixing_seeded_ABO.py (p 1, gnb 1, merge_thresh 1, OASIS; ``p``
    and ``gnb`` as given for the cross-validation of CNMF's settings). The memmap lives in a temporary folder,
    ``CAIMAN_TEMP`` during the fit (module docstring), deleted with the folder after the fit."""
    import caiman as cm  # type: ignore[import-not-found]
    from caiman.source_extraction.cnmf import cnmf  # type: ignore[import-not-found]

    before = os.environ.get("CAIMAN_TEMP")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["CAIMAN_TEMP"] = tmp
        try:
            fname = cm.save_memmap([str(data / f"{name}.h5")], base_name=str(Path(tmp) / f"{name}_memmap_"),
                                   order="C", var_name_hdf5="mov")
            c, dview, n_proc = cm.cluster.setup_cluster(backend="multiprocessing", n_processes=n_processes,
                                                        single_thread=False)
            yr = images = cnm = None
            try:
                yr, dims, t = cm.load_memmap(fname)
                images = np.reshape(yr.T, [t] + list(dims), order="F")
                radius = int(np.median(np.sqrt(roi.sum() / np.pi)))
                a_in = np.reshape(roi[None].transpose([2, 1, 0]), (-1, 1), order="C")
                t0 = time.perf_counter()
                cnm = cnmf.CNMF(check_nan=t <= 10000, n_processes=n_proc, k=1, gSig=[radius, radius],
                                merge_thresh=1, p=p, Ain=a_in.astype(bool), dview=dview, rf=None, stride=None,
                                gnb=gnb, method_deconvolution="oasis", border_pix=0, low_rank_background=True,
                                n_pixels_per_process=1000)
                cnm.fit(images)
                c_, yra = np.asarray(cnm.estimates.C), np.asarray(cnm.estimates.YrA)
                SECONDS["cnmf"] = time.perf_counter() - t0
            finally:
                cm.stop_server(dview=dview)
                yr = images = cnm = None     # release the memmap before its folder is deleted (Windows)
        finally:
            if before is None:
                os.environ.pop("CAIMAN_TEMP", None)
            else:
                os.environ["CAIMAN_TEMP"] = before
    return {"cnmf": (c_[0] + yra[0]).astype(np.float64), "cnmf_C": c_[0].astype(np.float64)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--data", type=Path, required=True, help="the D3 export (cai1_export)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default="fissa,tuncat")
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                    help="CaImAn's processes")
    ap.add_argument("--fissa-separation-cores", default="1", help="an integer, or 'all' (FISSA's default)")
    ap.add_argument("--cnmf-p", type=int, default=1, help="CNMF's AR order (the published setting: 1)")
    ap.add_argument("--cnmf-gnb", type=int, default=1, help="CNMF's background components (published: 1)")
    a = ap.parse_args(argv)
    methods = tuple(a.methods.split(","))
    if any(m not in METHODS for m in methods):
        ap.error(f"--methods must be a subset of {','.join(METHODS)}")
    for name in a.recordings:
        t0 = time.time()
        movie, roi = read(a.data, name)
        res: dict[str, np.ndarray] = {}
        SECONDS.clear()
        if "fissa" in methods:
            cores = None if a.fissa_separation_cores == "all" else int(a.fissa_separation_cores)
            res.update(run_fissa(movie, roi, cores))
        if "tuncat" in methods:
            res.update(run_tuncat(name, movie, roi))
        if "cnmf" in methods:
            res.update(run_cnmf(name, a.data, roi, a.cpus, p=a.cnmf_p, gnb=a.cnmf_gnb))
        save(a.out / f"{name}.npz", res)
        sec = a.out / f"{name}_seconds.json"
        old = json.loads(sec.read_text(encoding="utf-8")) if sec.exists() else {}
        sec.write_text(json.dumps({**old, **SECONDS}, indent=1), encoding="utf-8")
        print(f"{name}: {','.join(methods)} in {time.time() - t0:.0f} s; {movie.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
