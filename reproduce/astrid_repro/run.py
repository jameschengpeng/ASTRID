"""ASTRID on a public dataset as the paper ran it: every video (on D3, every recording) at every setting of the
dataset's grid in ``datasets.yaml``, one fit per (video, setting), saved as
``<out>/<dataset>/ks<kappa_d>_kb<kappa_np>/<video>/result.npz``, the layout that ``tables_public`` reads
(``--ours <out>``). The setting folders are named as in the paper's tables (``ks2_kb0.1``: kappa_d 2, kappa_np
0.1, in ``{:g}`` format).

Every fit uses the dataset's frame rate (on D3, each recording's, rounded to four decimals), pixel size and
indicator time constants (on D3, by the cell's indicator), ``border_px`` 0 and ``overlaps="nearest"`` (the
released masks overlap), and the defaults for everything else. A fit whose ``result.npz`` exists is skipped, so
a stopped run resumes.

**Inputs.**
- D2, D4, D5: TUnCaT's archive layout, ``<data>/data/<ds>/<video>.h5`` (``mov``) and
  ``<data>/data/<ds>/GT Masks/FinalMasks_<video>.mat``; D2's movies are the Allen crops of ``abo_crop``.
- D3: the export of ``cai1_export`` (``index.json``, ``<recording>.h5`` with ``mov`` and the binned ``roi``).
D1, D6 and D7 need the authors' recordings, whose availability is to be announced.

    python -m astrid_repro.run D2 --data <archive> --out <results> [--workers 8] [--videos ...] [--settings ...]
    python -m astrid_repro.run D3 --data <export> --out <results> ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

import astrid
from astrid_repro import public_data as D

CONFIG = Path(__file__).resolve().parents[1] / "datasets.yaml"
DATASETS = {"D2": ("D2_ABO", "ABO"), "D3": ("D3_cai1", "cai1"), "D4": ("D4_NAOMi", "NAOMi"),
            "D5": ("D5_one_photon", "1p")}            # paper name -> (datasets.yaml entry, results folder)


def setting_name(kappa_d: float, kappa_np: float) -> str:
    return f"ks{kappa_d:g}_kb{kappa_np:g}"


def _pairs(x: Any) -> list[tuple[float, float]]:
    if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
        return [(float(x[0]), float(x[1]))]
    return [p for y in x for p in _pairs(y)]


def grid(dataset: str, config: str | Path = CONFIG) -> list[tuple[float, float]]:
    """The dataset's (kappa_d, kappa_np) pairs, in ``datasets.yaml``'s order: the 14 shared by the public
    datasets, then the dataset's own 5."""
    entry = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["datasets"][DATASETS[dataset][0]]
    pairs = _pairs(entry["grid"])
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{dataset}: a setting appears twice in the grid")
    return pairs


def units(dataset: str, data: str | Path, config: str | Path = CONFIG) -> list[dict[str, Any]]:
    """One fit per (video, setting), with its inputs: ``video``, ``movie``, ``rois`` (a path), ``fps``,
    ``um_per_px``, ``tau_r``, ``tau_d``, ``kappa_d``, ``kappa_np``."""
    entry = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["datasets"][DATASETS[dataset][0]]
    folder = DATASETS[dataset][1]
    pairs = grid(dataset, config)
    out = []
    if dataset == "D3":
        for row in D.read_cai1_index(data):
            ind = row["indicator"]
            for kd, knp in pairs:
                out.append({"video": row["name"], "movie": Path(data) / f"{row['name']}.h5", "rois": None,
                            "fps": float(f"{row['fps']:.4f}"), "um_per_px": float(entry["um_per_px"]),
                            "tau_r": float(entry["tau_r"][ind]), "tau_d": float(entry["tau_d"][ind]),
                            "kappa_d": kd, "kappa_np": knp})
        return out
    for video in D.VIDEOS[folder]:
        for kd, knp in pairs:
            out.append({"video": video, "movie": D.movie_path(data, folder, video),
                         "rois": D.masks_path(data, folder, video), "fps": float(entry["fps"]),
                         "um_per_px": float(entry["um_per_px"]), "tau_r": float(entry["tau_r"]),
                         "tau_d": float(entry["tau_d"]), "kappa_d": kd, "kappa_np": knp})
    return out


def destination(out: str | Path, dataset: str, unit: dict[str, Any]) -> Path:
    return Path(out) / DATASETS[dataset][1] / setting_name(unit["kappa_d"], unit["kappa_np"]) / unit["video"]


def fit(dataset: str, unit: dict[str, Any], workers: int = 0) -> astrid.Result:
    """One fit: the unit's movie and ROI masks, its settings, border 0 and the nearest-centroid rule."""
    if unit["rois"] is None:                                   # D3: the export's binned ROI
        with h5py.File(unit["movie"], "r") as f:
            rois = np.asarray(f["roi"], dtype=bool)[None]
    else:
        rois = D.read_masks(unit["rois"])
    settings = astrid.Settings(tau_r=unit["tau_r"], tau_d=unit["tau_d"], kappa_d=unit["kappa_d"],
                               kappa_np=unit["kappa_np"])
    return astrid.demix(f"{unit['movie']}:mov", rois, fps=unit["fps"], um_per_px=unit["um_per_px"],
                        settings=settings, border_px=0, overlaps="nearest", workers=workers,
                        recording_names=[unit["video"]], name=f"{DATASETS[dataset][1]}/{unit['video']}")


def run(dataset: str, data: str | Path, out: str | Path, *, videos: list[str] | None = None,
        settings: list[str] | None = None, workers: int = 0, config: str | Path = CONFIG,
        log: Any = print) -> list[Path]:
    """Every unit of the dataset not yet fitted (``videos`` and ``settings`` narrow them down); returns the
    result files written."""
    done = []
    todo = [u for u in units(dataset, data, config)
            if (not videos or u["video"] in videos)
            and (not settings or setting_name(u["kappa_d"], u["kappa_np"]) in settings)]
    for i, u in enumerate(todo, start=1):
        dest = destination(out, dataset, u)
        if (dest / "result.npz").exists():
            continue
        t0 = time.time()
        path = fit(dataset, u, workers).save(dest)
        done.append(path)
        log(f"{dataset} {u['video']} {setting_name(u['kappa_d'], u['kappa_np'])}: {time.time() - t0:.0f} s "
            f"({i} of {len(todo)})")
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--data", type=Path, required=True, help="TUnCaT's archive (D2, D4, D5) or D3's export")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videos", nargs="*", help="only these videos or recordings")
    ap.add_argument("--settings", nargs="*", help="only these settings, e.g. ks2_kb0.1")
    ap.add_argument("--workers", type=int, default=0, help="ROIs fitted at a time (0: in this process)")
    ap.add_argument("--list", action="store_true", help="print the units as JSON lines and stop")
    a = ap.parse_args(argv)
    if a.list:
        for u in units(a.dataset, a.data):
            print(json.dumps({**u, "movie": str(u["movie"]), "rois": None if u["rois"] is None else str(u["rois"]),
                              "setting": setting_name(u["kappa_d"], u["kappa_np"])}))
        return 0
    run(a.dataset, a.data, a.out, videos=a.videos, settings=a.settings, workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
