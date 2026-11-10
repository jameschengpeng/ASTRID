"""The peer methods on exported cases, with their authors' code: FISSA, TUnCaT and TUnCaT + territory, CNMF and
CNMF + territory (CaImAn), as they ran on D1, D6 and D7.

FISSA, TUnCaT and CaImAn are GPL and are not part of this repository. Install them in an environment of their
own and run this module there. It imports none of them until a method needs it, and nothing from ``astrid``.
In the paper, FISSA 1.0.0 and TUnCaT (its v1.0 scripts, packaged as 1.1.1) shared one environment and CaImAn
1.13.2 had its own.

**Inputs: the case format** (one folder per recording, ``--cases``):
- ``index.json``: the cases, one row each, with ``spec`` (the case's name), ``base``, ``test`` (``injection``,
  ``leakage``; ``d6`` on D6), ``scale`` (the added signal's strength), ``site`` (``astro`` or ``control`` for
  an injection) and the counts ``neighbours`` and ``atoms``;
- per base (an injection site or a leakage target; its cases share it):
  - ``<base>_movie.npy``: the raw movie cropped around the ROI, (T, h, w) float32;
  - ``<base>_masks.npz``: ``target`` (h, w) bool, the ROI; ``neighbours`` (k, h, w) bool, the neighbouring
    cells ASTRID had; ``atoms`` (m, h, w) bool, the pieces of astrocyte territory ASTRID had (in a leakage case
    with the added astrocyte's support); ``box``, the crop (r0, r1, c0, c1) in the field;
- per case, ``<case>.npz``: ``footprint`` (h, w) float32 and ``amplitude`` (T,), the added signal, so that the
  case's movie is the base's movie + footprint x amplitude (float32; ``case_movie``); and its reference,
  ``truth`` (an injection: the injected ROI average) or ``astro_roi`` (a leakage case: the added astrocyte
  signal averaged over the ROI).
A case's base is its name without the strength (``base_of``): ``inj00-astro-x3`` -> ``inj00-astro``,
``M-039-astro-x1`` -> ``M-039``; any other name is its own base.

**Methods** (``--methods``, comma-separated; ``fissa,tuncat`` by default):
- ``fissa``: FISSA on the target ROI at its default settings (4 neuropil sub-regions): ``fissa``, the separated
  signal matched to the ROI in raw units, and ``fissa_raw_roi``, FISSA's raw ROI trace;
- ``tuncat``: TUnCaT on the target and the neighbouring cells (alpha 1, flexible alpha, tolerance 1e-4 and at
  most 20,000 iterations; ``TUNCAT``): ``tuncat``, with ``tuncat_raw_roi`` and ``tuncat_bg`` (its raw ROI and
  background traces); ``tuncat_astro`` (TUnCaT + terr.): the same with the atoms as further masks;
- ``cnmf``: CaImAn's CNMF seeded with the target and the neighbouring cells, at the settings of TUnCaT's seeded
  CNMF (``CNMF``: AR(1), no merging, OASIS, a low-rank background of one component, no patches; gSig the
  median mask radius): the target's C + YrA in the ROI's raw units (times its footprint's mean over the ROI);
  ``cnmf_astro`` (CNMF + terr.): seeded with the atoms too; ``cnmf_info`` and ``cnmf_astro_info``: the seeds,
  the components kept, the target's component (-1 if CaImAn removed it; the trace is then zero), its match
  and the scale. CaImAn only removes components and keeps the seeds' order, so the target's component is
  found by the order-preserving assignment of kept components to seeds with the largest summed Jaccard index
  (``seeds_of``; a footprint above 0.2 of its peak against a seed mask, ``jaccard``). ``--cnmf-p`` and
  ``--cnmf-gnb`` set the AR order and the number of background components instead (the paper's in-sample
  sensitivity check of CNMF's settings on D1; unset, the fit is the published setting's, p 1 and gnb 1). The
  fit runs in memory: CaImAn writes no memmap.

**Outputs:** ``<results>/<case>.npz`` (``--results``, default ``<cases>/results``) with each method's arrays
for the target (``RESULT_KEYS``). A run of some methods merges into an existing file and keeps the others'.
FISSA is deterministic; TUnCaT is not (its NMF runs in a process pool), and reruns differ by up to about 0.002.

Run (from ``reproduce/``, in the methods' environment):
    python -m astrid_repro.peers run --cases <folder> [--results <dir>] [--methods fissa,tuncat] <case> ...
    python -m astrid_repro.peers launch --cases <folder> --runners 16 [--python <env python>] [--methods cnmf]
``launch`` runs every case whose results lack a requested method over N processes (case i to runner i mod
N), each appending to ``run_<i>.log`` beside the results folder.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

TUNCAT: dict[str, Any] = dict(list_alpha=[1], Qclip=0, th_pertmin=1, epsilon=0, th_residual=0, nbin=1,
                              bin_option="downsample", multi_alpha=True, flexible_alpha=True, tol=1e-4,
                              max_iter=20000)
# TUnCaT's seeded CNMF scripts (their params_movie and CNMF call), with one background component
CNMF: dict[str, Any] = dict(p=1, merge_thresh=1, gnb=1, method_deconvolution="oasis", border_pix=0,
                            low_rank_background=True, n_pixels_per_process=1000, rf=None, stride=None)
DEFAULT_METHODS = ("fissa", "tuncat")
METHODS = (*DEFAULT_METHODS, "cnmf")
RESULT_KEYS = {"fissa": ("fissa", "fissa_raw_roi"),                 # what ``run_case`` saves per method
               "tuncat": ("tuncat", "tuncat_raw_roi", "tuncat_bg", "tuncat_astro"),
               "cnmf": ("cnmf", "cnmf_info", "cnmf_astro", "cnmf_astro_info")}
TUNCAT_CPUS = 3                     # the pool TUnCaT's NMF sizes itself to (runners share the CPU)


def base_of(case: str) -> str:
    """The base of a case: inj00-astro-x3 -> inj00-astro, M-039-astro-x1 -> M-039."""
    return case.rsplit("-x", 1)[0] if case.startswith("inj") else case.rsplit("-astro-x", 1)[0]


def case_movie(cases: Path, case: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The case's movie (T, h, w) float32 and its base's masks."""
    spec = np.load(cases / f"{case}.npz")
    base = base_of(case)
    movie = np.load(cases / f"{base}_movie.npy")
    movie = (movie + spec["footprint"][None] * spec["amplitude"][:, None, None].astype(np.float32)
             ).astype(np.float32)
    with np.load(cases / f"{base}_masks.npz") as m:
        masks = {k: m[k] for k in m.files}
    return movie, masks


def save_results(path: Path, res: dict[str, Any]) -> None:
    """Merge ``res`` into the results npz at ``path`` (its other arrays kept), written atomically; an
    unreadable old file (an interrupted write) is replaced."""
    if path.exists():
        try:
            with np.load(path) as old:
                res = {**{k: old[k] for k in old.files}, **res}
        except (OSError, ValueError, zipfile.BadZipFile) as err:
            print(f"{path.name}: unreadable ({err}); replaced", flush=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **res)
    os.replace(tmp, path)


def jaccard(footprint: np.ndarray, mask: np.ndarray) -> float:
    """Jaccard index of a footprint above 0.2 of its peak with a mask."""
    peak = footprint.max()
    on = footprint > 0.2 * peak if peak > 0 else np.zeros_like(mask)
    union = (on | mask).sum()
    return float((on & mask).sum() / union) if union else 0.0


def seeds_of(A: np.ndarray, seeds: np.ndarray) -> list[int]:
    """The seed each kept component (a column of ``A``, pixels in column-major order) comes from: the
    order-preserving assignment with the largest summed ``jaccard`` (dynamic programming). With nothing
    removed it is the identity."""
    k, kept = len(seeds), A.shape[1]
    J = np.array([[jaccard(A[:, j], s.ravel(order="F")) for j in range(kept)]
                  for s in seeds]).reshape(k, kept)
    best = np.full((k + 1, kept + 1), -np.inf)
    best[:, 0] = 0.0
    for i in range(1, k + 1):
        for j in range(1, min(i, kept) + 1):
            best[i, j] = max(best[i - 1, j], best[i - 1, j - 1] + J[i - 1, j - 1])
    out, i, j = [0] * kept, k, kept
    while j > 0:                                    # back through the table: matched, or seed i-1 removed
        if best[i, j] == best[i - 1, j - 1] + J[i - 1, j - 1]:
            out[j - 1] = i - 1
            j -= 1
        i -= 1
    return out


def cnmf_target(movie: np.ndarray, seeds: np.ndarray, cnmf_mod: Any = None,
                params: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CNMF seeded with ``seeds`` (k, h, w) bool, the target first; returns the target's C + YrA in the ROI's
    raw units and [seeds, components kept, the target's component, its Jaccard match, the scale]. A target
    whose component CaImAn removed gets a zero trace (TUnCaT's convention for dropped components)."""
    if cnmf_mod is None:
        from caiman.source_extraction.cnmf import cnmf as cnmf_mod  # type: ignore[import-not-found]
    T, d1, d2 = movie.shape
    # CaImAn's layout: pixels in column-major order, Yr (pixels, T) C-contiguous
    Yr = np.ascontiguousarray(movie.reshape(T, d1 * d2, order="F").T, dtype=np.float32)
    images = np.reshape(Yr.T, (T, d1, d2), order="F")
    a_in = np.stack([s.ravel(order="F") for s in seeds], axis=1).astype(bool)
    radius = max(1, int(np.median(np.sqrt(seeds.sum(axis=(1, 2)) / np.pi))))
    cnm = cnmf_mod.CNMF(n_processes=1, dview=None, k=a_in.shape[1], gSig=[radius, radius], Ain=a_in,
                        check_nan=T <= 10000, **(CNMF if params is None else params))
    cnm.fit(images)                         # fit() works in place
    est = cnm.estimates
    A = est.A.toarray() if hasattr(est.A, "toarray") else np.asarray(est.A)
    trace = np.asarray(est.C, dtype=np.float64) + np.asarray(est.YrA, dtype=np.float64)
    target = seeds[0].ravel(order="F")
    origin = seeds_of(A, seeds)
    if 0 not in origin:                             # the target's component was removed
        return np.zeros(T), np.array([len(seeds), A.shape[1], -1, 0.0, 0.0])
    j = origin.index(0)                             # 0: the kept components keep the seeds' order
    scale = float(A[target, j].mean())
    return scale * trace[j], np.array([len(seeds), A.shape[1], j, jaccard(A[:, j], target), scale])


def run_case(cases: Path, case: str, methods: tuple[str, ...] = DEFAULT_METHODS, results: Path | None = None,
             tuncat_tol: float = TUNCAT["tol"], cnmf: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Run ``methods`` on one case and save (merge) its results npz in ``results`` (default
    ``<cases>/results``); returns this run's traces."""
    t0 = time.time()
    movie, m = case_movie(cases, case)
    target = m["target"].astype(bool)
    res: dict[str, Any] = {}
    timing = []
    if "fissa" in methods:
        import fissa  # type: ignore[import-not-found]

        exp = fissa.Experiment([movie], [[target]], ncores_preparation=1, ncores_separation=1, verbosity=0)
        exp.separate()
        res["fissa"] = np.asarray(exp.result[0, 0][0], dtype=np.float64)
        res["fissa_raw_roi"] = np.asarray(exp.raw[0, 0][0], dtype=np.float64)
        timing.append(f"fissa {time.time() - t0:.0f} s")
    if "tuncat" in methods:
        from tuncat.run_TUnCaT import run_TUnCaT  # type: ignore[import-not-found]

        t1 = time.time()
        kw = {**TUNCAT, "tol": tuncat_tol}
        with tempfile.TemporaryDirectory() as tmp:
            masks = np.concatenate([target[None], m["neighbours"].astype(bool)])
            out_t = run_TUnCaT(case, movie, masks, tmp, **kw)
            res["tuncat"], res["tuncat_raw_roi"], res["tuncat_bg"] = (out_t[0][:, 0], out_t[2][:, 0],
                                                                      out_t[3][:, 0])
            masks = np.concatenate([masks, m["atoms"].astype(bool)])
            out_t = run_TUnCaT(case + "_astro", movie, masks, tmp, **kw)
            res["tuncat_astro"] = out_t[0][:, 0]
        timing.append(f"tuncat x2 (tol {tuncat_tol:g}) {time.time() - t1:.0f} s; {masks.shape[0]} masks")
    if "cnmf" in methods:
        t1 = time.time()
        kw = {} if cnmf is None else {"params": cnmf}          # CNMF's settings, when not ``CNMF``'s
        seeds = np.concatenate([target[None], m["neighbours"].astype(bool)])
        res["cnmf"], res["cnmf_info"] = cnmf_target(movie, seeds, **kw)
        seeds = np.concatenate([seeds, m["atoms"].astype(bool)])
        res["cnmf_astro"], res["cnmf_astro_info"] = cnmf_target(movie, seeds, **kw)
        timing.append(f"cnmf x2 {time.time() - t1:.0f} s; {seeds.shape[0]} seeds")
    results = cases / "results" if results is None else results
    results.mkdir(parents=True, exist_ok=True)
    save_results(results / f"{case}.npz", res)
    print(f"{case}: " + ", ".join(timing), flush=True)
    return res


def saved_keys(path: Path) -> set[str]:
    """The arrays in a results npz (none if it is missing or unreadable), read without numpy."""
    try:
        with zipfile.ZipFile(path) as z:
            return {n.removesuffix(".npy") for n in z.namelist()}
    except (OSError, zipfile.BadZipFile):
        return set()


def pending(cases: Path, methods: tuple[str, ...] = DEFAULT_METHODS,
            results: Path | None = None) -> tuple[list[str], list[str]]:
    """(every case of the index, those whose results (default ``<cases>/results``) lack an array of
    ``methods``), in index order."""
    every = [c["spec"] for c in json.loads((cases / "index.json").read_text())]
    need = {k for m in methods for k in RESULT_KEYS[m]}
    results = cases / "results" if results is None else results
    return every, [c for c in every if not need <= saved_keys(results / f"{c}.npz")]


def shards(todo: list[str], n: int) -> list[list[str]]:
    """Case i to runner i mod n."""
    return [todo[i::n] for i in range(n)]


def parse_methods(text: str | None) -> tuple[str, ...]:
    """``--methods`` as a tuple (the default methods if not given); ValueError for an unknown method."""
    methods = tuple(x for x in (text or ",".join(DEFAULT_METHODS)).split(",") if x)
    if not set(methods) <= set(METHODS):
        raise ValueError(f"--methods must be a subset of {','.join(METHODS)}")
    return methods


def launch(cases: Path, runners: int, python: str, methods_arg: str | None = None, results_arg: Path | None = None,
           tuncat_tol: str | None = None, cnmf_p: str | None = None, cnmf_gnb: str | None = None) -> int:
    """Run every pending case in ``runners`` processes of this module's ``run``; returns the largest exit code,
    or 1 if a case is still pending afterwards."""
    methods = parse_methods(methods_arg)
    cases = cases.resolve()
    results = (cases / "results" if results_arg is None else results_arg).resolve()
    logs = cases if results_arg is None else results.parent
    results.mkdir(parents=True, exist_ok=True)
    every, todo = pending(cases, methods, results)
    extra = ["--methods", methods_arg] if methods_arg else []
    extra += ["--results", str(results)] if results_arg is not None else []
    extra += ["--tuncat-tol", tuncat_tol] if tuncat_tol is not None else []
    extra += ["--cnmf-p", cnmf_p] if cnmf_p is not None else []
    extra += ["--cnmf-gnb", cnmf_gnb] if cnmf_gnb is not None else []
    here = Path(__file__).resolve().parents[1]                    # reproduce/, holding astrid_repro
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    procs = []
    for i, part in enumerate(shards(todo, runners)):
        with open(logs / f"run_{i}.log", "a") as log:
            log.write(f"== launch {stamp}: {len(part)} cases, methods {','.join(methods)}\n")
            log.flush()
            procs.append(subprocess.Popen([python, "-m", "astrid_repro.peers", "run", "--cases", str(cases),
                                           *extra, *part], cwd=here, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    left = pending(cases, methods, results)[1]
    print(len(todo), "cases,", runners, "runners, exit codes", codes, "; results", len(every) - len(left),
          "of", len(every), flush=True)
    return max([*codes, 1 if left else 0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the methods on the named cases, one after the other")
    r.add_argument("names", nargs="+", metavar="case")
    r.add_argument("--tuncat-cpus", type=int, default=TUNCAT_CPUS,
                   help="the cpu_count TUnCaT sizes its NMF pool by (runners share the CPU)")
    lp = sub.add_parser("launch", help="run every pending case in N processes")
    lp.add_argument("--runners", type=int, required=True)
    lp.add_argument("--python", default=sys.executable, help="the methods' environment's python")
    for p in (r, lp):
        p.add_argument("--cases", type=Path, required=True, help="the exported cases (index.json)")
        p.add_argument("--results", type=Path, default=None, help="where the results go (default <cases>/results)")
        p.add_argument("--methods", default=None, help=f"a comma-separated subset of {','.join(METHODS)} "
                       f"(default {','.join(DEFAULT_METHODS)})")
        p.add_argument("--tuncat-tol", default=None, help=f"TUnCaT's tolerance (default {TUNCAT['tol']:g})")
        p.add_argument("--cnmf-p", default=None, help=f"CNMF's AR order (default {CNMF['p']})")
        p.add_argument("--cnmf-gnb", default=None, help=f"CNMF's background components (default {CNMF['gnb']})")
    a = ap.parse_args(argv)
    try:
        methods = parse_methods(a.methods)
        tol = TUNCAT["tol"] if a.tuncat_tol is None else float(a.tuncat_tol)
    except ValueError as err:
        ap.error(str(err))
    if a.command == "launch":
        if a.runners < 1:
            ap.error("--runners must be at least 1")
        return launch(a.cases, a.runners, a.python, a.methods, a.results, a.tuncat_tol, a.cnmf_p, a.cnmf_gnb)
    cpus = a.tuncat_cpus
    mp.cpu_count = lambda: cpus           # TUnCaT sizes its NMF pool by cpu_count()
    settings = {**CNMF, **({} if a.cnmf_p is None else {"p": int(a.cnmf_p)}),
                **({} if a.cnmf_gnb is None else {"gnb": int(a.cnmf_gnb)})}
    for c in a.names:
        run_case(a.cases, c, methods, a.results, tol, None if settings == CNMF else settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
