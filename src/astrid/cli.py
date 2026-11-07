"""The command line: ``astrid <command> ...`` (or ``python -m astrid``); each command writes
``OUT/result.npz`` and ``OUT/result.json`` (``Result.save``).

    astrid demix    --movie M.h5:data --rois R.npz --fps 30 --um-per-px 0.8 --tau-r 0.045 --tau-d 0.142 --out OUT
    astrid suite2p  suite2p/plane0 --um-per-px 0.8 --tau-r 0.045 --tau-d 0.142 --out OUT
    astrid caiman   cnm.hdf5 --tau-r 0.045 --tau-d 0.142 --out OUT
    astrid aqua2    --movie M.h5 --rois R.npz --aqua2 res1.mat res2.mat --fps 30.9 --um-per-px 2 ... --out OUT
    astrid pipeline location.yaml
    astrid detect   --movie r1.h5 --movie r2.h5 --aqua2 r1.mat r2.mat --fps 30.9 --out ROIS

Masks (``--rois``, ``--territory``, ``--candidates``): ``file.npz[:key]`` or ``file.npy`` holding (n, H, W)
booleans or an (H, W) label image (0 = none), or ``file.mat[:key]`` (TUnCaT's ``FinalMasks``, (H, W, n)).
``--evidence``: an (H, W) map in [0, 1] (``.npz[:key]``/``.npy``). Settings: ``--settings S.yaml`` (the
paper's names) and ``--kappa-d``, ``--kappa-np``, ``--kappa-a``, ``--kappa-rho`` on top of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np

from astrid import Settings, __version__
from astrid.api import OVERLAP_RULES, Result, demix
from astrid.endpoints import detect_rois, from_caiman, from_suite2p, run_pipeline, territory_from_aqua2


def _split(spec: str) -> tuple[Path, str | None]:
    drive = len(spec) > 1 and spec[1] == ":"
    path, key = (spec.rsplit(":", 1) if spec.count(":") > (1 if drive else 0) else (spec, None))
    return Path(path), key


def load_array(spec: str) -> np.ndarray:
    """An array from ``file.npz[:key]`` (the first array without a key), ``file.npy`` or ``file.mat[:key]``."""
    p, key = _split(spec)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".mat":
        try:
            from scipy.io import loadmat
            m = loadmat(p)
            arr = np.asarray(m[key or "FinalMasks"])
            return arr.transpose(2, 0, 1) if arr.ndim == 3 else arr
        except NotImplementedError:          # a v7.3 file
            import h5py
            with h5py.File(p, "r") as f:
                arr = np.asarray(f[key or "FinalMasks"])   # (n, W, H) as h5py reads MATLAB's (H, W, n)
            return arr.transpose(0, 2, 1) if arr.ndim == 3 else arr.T
    with np.load(p) as z:
        return np.asarray(z[key or z.files[0]])


def load_masks(spec: str) -> np.ndarray:
    """(n, H, W) booleans from a mask file (module docstring); an (H, W) label image is split by label."""
    arr = load_array(spec)
    if arr.ndim == 2 and arr.dtype != bool:
        n = int(arr.max())
        return np.stack([arr == v for v in range(1, n + 1)]) if n else np.zeros((0, *arr.shape), bool)
    return arr.astype(bool)


def _settings(a: argparse.Namespace) -> Settings:
    s = Settings.from_yaml(a.settings) if a.settings else Settings()
    over = {k: getattr(a, k) for k in ("kappa_d", "kappa_np", "kappa_a", "kappa_rho") if getattr(a, k) is not None}
    if a.no_neuropil_cap:
        over["neuropil_cap"] = False
    return dataclasses.replace(s, **over) if over else s


def _common(p: argparse.ArgumentParser, movie: bool = True, rois: bool = True, rates: bool = True) -> None:
    if movie:
        p.add_argument("--movie", action="append", required=True,
                       help="movie file[:dataset] (.h5, .npy, .tif); repeat for recordings of one field")
    if rois:
        p.add_argument("--rois", required=True, help="ROI masks (see above)")
    if rates:
        p.add_argument("--fps", type=float, required=True, help="frame rate (Hz)")
        p.add_argument("--um-per-px", type=float, required=True, help="pixel size (um)")
    p.add_argument("--tau-r", type=float, help="the indicator's rise time constant (s)")
    p.add_argument("--tau-d", type=float, help="the indicator's decay time constant (s)")
    p.add_argument("--settings", help="a YAML file of settings in the paper's names")
    for k in ("kappa-d", "kappa-np", "kappa-a", "kappa-rho"):
        p.add_argument(f"--{k}", type=float)
    p.add_argument("--no-neuropil-cap", action="store_true", help="the model without the neuropil cap")
    p.add_argument("--border-px", type=int, default=0, help="registration margin not fitted (default 0)")
    p.add_argument("--overlaps", choices=OVERLAP_RULES, help="overlapping masks: refuse or nearest")
    p.add_argument("--workers", type=int, default=0, help="fit the ROIs in this many processes")
    p.add_argument("--export-components", action="store_true",
                   help="also write the components' contributions, traces and footprints")
    p.add_argument("--out", type=Path, required=True)


def _kwargs(a: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = dict(tau_r=a.tau_r, tau_d=a.tau_d, settings=_settings(a), border_px=a.border_px,
                              workers=a.workers, export_components=a.export_components)
    if a.overlaps:
        kw["overlaps"] = a.overlaps
    return kw


def _names(movies: list[str]) -> list[str]:
    names = [_split(m)[0].stem for m in movies]
    return names if len(set(names)) == len(names) else [f"rec{i}" for i in range(len(names))]


def _report(res: Result, out: Path) -> None:
    res.save(out)
    fitted = sum(r.fitted for r in res.rois)
    print(f"{out}: {fitted} of {len(res.rois)} ROIs fitted in {res.provenance['total_seconds']:.0f} s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="astrid", description="ASTRID: astrocyte-informed demixing of neuronal "
                                 "calcium traces.", epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demix", help="a movie and ROI masks")
    _common(p)
    p.add_argument("--territory", help="astrocyte territory masks, (H, W) or (m, H, W)")
    p.add_argument("--evidence", help="structural evidence map in [0, 1], (H, W)")
    p.add_argument("--candidates", help="map of candidate cells that are not ROIs, (H, W)")
    p = sub.add_parser("suite2p", help="a Suite2p plane folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--um-per-px", type=float, required=True)
    p.add_argument("--all-rois", action="store_true", help="every ROI, not only iscell's cells")
    _common(p, movie=False, rois=False, rates=False)
    p = sub.add_parser("caiman", help="a CaImAn result (cnm.save's HDF5 file)")
    p.add_argument("file", type=Path)
    p.add_argument("--movie-file", help="the motion-corrected movie (default: the run's memmap)")
    p.add_argument("--um-per-px", type=float, help="pixel size (default: the run's dxy)")
    _common(p, movie=False, rois=False, rates=False)
    p = sub.add_parser("aqua2", help="a movie and ROI masks, with the territory from AQuA2's results")
    _common(p)
    p.add_argument("--aqua2", nargs="+", required=True, help="AQuA2 result files (.mat)")
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--gap", type=int, help="AQuA2's regMaskGap, if the results do not record it")
    p = sub.add_parser("pipeline", help="one imaging location described by a YAML file (README)")
    p.add_argument("config", type=Path)
    p = sub.add_parser("detect", help="the paper's ROI detection on one imaging location's recordings")
    p.add_argument("--movie", action="append", required=True, help="movie file[:dataset]; one per recording")
    p.add_argument("--aqua2", nargs="+", help="one AQuA2 result per movie, in the same order (astrocyte masks)")
    p.add_argument("--gap", type=int, help="AQuA2's regMaskGap, if the results do not record it")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == "pipeline":
        return _pipeline(a.config)
    if a.command == "detect":
        det = detect_rois(a.movie, fps=a.fps, aqua2_results=a.aqua2, gap=a.gap)
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out / "rois.npz", rois=det.rois, candidates=det.candidates,
                            detected_in=det.detected_in, astro_union=det.astro_union)
        print(f"{a.out / 'rois.npz'}: {det.rois.shape[0]} ROIs from {len(a.movie)} recordings")
        return 0
    if a.command == "demix":
        res = demix(a.movie, load_masks(a.rois), fps=a.fps, um_per_px=a.um_per_px,
                    territory=load_masks(a.territory) if a.territory else None,
                    evidence=load_array(a.evidence) if a.evidence else None,
                    candidates=load_masks(a.candidates) if a.candidates else None,
                    recording_names=_names(a.movie), **_kwargs(a))
    elif a.command == "suite2p":
        res = from_suite2p(a.folder, um_per_px=a.um_per_px, cells_only=not a.all_rois, **_kwargs(a))
    elif a.command == "caiman":
        res = from_caiman(a.file, um_per_px=a.um_per_px, movie=a.movie_file, **_kwargs(a))
    else:
        res = territory_from_aqua2(a.movie, load_masks(a.rois), a.aqua2, fps=a.fps, um_per_px=a.um_per_px,
                                   min_events=a.min_events, gap=a.gap, recording_names=_names(a.movie),
                                   **_kwargs(a))
    _report(res, a.out)
    return 0


PIPELINE_KEYS = {"movies", "rois", "fps", "um_per_px", "tau_r", "tau_d", "aqua2_results", "marker", "candidates",
                 "min_events", "gap", "settings", "border_px", "overlaps", "workers", "export_components", "out",
                 "recording_names"}


def _pipeline(path: Path) -> int:
    """``run_pipeline`` from a YAML file: the keys of ``PIPELINE_KEYS``, paths relative to the file."""
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = sorted(set(cfg) - PIPELINE_KEYS)
    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown}")
    base = path.parent

    def rel(x: str) -> str:
        p, key = _split(str(x))
        full = p if p.is_absolute() else base / p
        return f"{full}:{key}" if key else str(full)

    movies = [rel(m) for m in cfg["movies"]]
    settings = Settings.from_dict(cfg["settings"]) if isinstance(cfg.get("settings"), dict) else Settings()
    res = run_pipeline(movies, load_masks(rel(cfg["rois"])) if cfg.get("rois") else None, fps=float(cfg["fps"]),
                       um_per_px=float(cfg["um_per_px"]), tau_r=cfg.get("tau_r"), tau_d=cfg.get("tau_d"),
                       aqua2_results=[rel(f) for f in cfg.get("aqua2_results") or []] or None,
                       marker=[rel(f) for f in cfg.get("marker") or []] or None,
                       candidates=load_masks(rel(cfg["candidates"])) if cfg.get("candidates") else None,
                       min_events=int(cfg.get("min_events", 2)), gap=cfg.get("gap"), settings=settings,
                       border_px=int(cfg.get("border_px", 0)), overlaps=cfg.get("overlaps", "refuse"),
                       workers=int(cfg.get("workers", 0)), export_components=bool(cfg.get("export_components")),
                       recording_names=cfg.get("recording_names") or _names(movies),
                       out=base / cfg.get("out", "astrid_out"))
    print(f"{base / cfg.get('out', 'astrid_out')}: {sum(r.fitted for r in res.rois)} of {len(res.rois)} ROIs "
          f"fitted in {res.provenance['total_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
