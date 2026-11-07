"""Astrocyte territory from AQuA2's results (Supplementary, "Astrocyte inputs").

AQuA2 (Mi et al. 2025; MATLAB, GPL-2, https://github.com/yu-lab-vt/AQuA2) is not part of this package: run it
on your movie (the settings the paper used are in this repository's ``aqua2_settings/``) and pass its result
files here. ASTRID uses only where the events were: the territory is the pixels touched by at least
``min_events`` events (default 2) over all the results given (the recordings of one imaging location). Event
times are not used.

AQuA2 drops ``regMaskGap`` pixels at each edge of the movie, and its event voxels index the cropped volume;
``event_counts`` puts the counts back on the full frame (the gap is read from the result's saved options,
or given).

Result files: the GUI's "Save" (``res``, a MATLAB v7.3 file) or "Send to workspace" saved with ``save``
(``res_dbg``, v5 or v7.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astrid.patch import TERRITORY_MIN_EVENTS


@dataclass(frozen=True)
class AquaEvents:
    events: list[np.ndarray]     # 0-based column-major linear indices into the cropped (x, y, T) volume
    x: int                       # rows of the cropped frame
    y: int                       # columns of the cropped frame
    n_frames: int
    gap: int | None              # regMaskGap from the saved options, if any


def _is_hdf5(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == b"\x89HDF\r\n\x1a\n" or (f.seek(512) == 512 and f.read(8) == b"\x89HDF\r\n\x1a\n")


def read_events(path: str | Path, variable: str | None = None) -> AquaEvents:
    """The events of one AQuA2 result file (``res`` or ``res_dbg``; ``variable`` picks one when the file has
    both). A result without events gives an empty list."""
    path = Path(path)
    if _is_hdf5(path):
        import h5py
        with h5py.File(path, "r") as f:
            name = variable or next((v for v in ("res", "res_dbg") if v in f), None)
            if name is None:
                raise ValueError(f"{path}: no 'res' or 'res_dbg' in the file")
            r = f[name]
            t, _, y, x = r["datOrg1"].shape                    # MATLAB [x, y, 1, T]
            evt = r["evt1"]
            refs = [] if evt.attrs.get("MATLAB_empty", 0) else evt[()].ravel()
            events = [np.asarray(f[ref][()]).ravel().astype(np.int64) - 1 for ref in refs]
            gap = None
            if "opts" in r and "regMaskGap" in r["opts"]:
                gap = int(np.asarray(r["opts"]["regMaskGap"][()]).ravel()[0])
        return AquaEvents(events, int(x), int(y), int(t), gap)
    import scipy.io as sio
    names = [v for v, *_ in sio.whosmat(path)]
    name = variable or next((v for v in ("res", "res_dbg") if v in names), None)
    if name is None:
        raise ValueError(f"{path}: no 'res' or 'res_dbg' in the file")
    res = sio.loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=[name])[name]
    x, y = res.datOrg1.shape[:2]
    events = [np.asarray(e, dtype=np.int64).ravel() - 1 for e in np.atleast_1d(res.evt1)]
    opts = getattr(res, "opts", None)
    gap = int(opts.regMaskGap) if opts is not None and hasattr(opts, "regMaskGap") else None
    return AquaEvents(events, int(x), int(y), int(res.datOrg1.shape[-1]), gap)


def event_counts(aqua: AquaEvents, gap: int | None = None) -> np.ndarray:
    """The number of events touching each pixel of the full (x + 2 gap, y + 2 gap) frame (each event counts
    once per pixel, whatever its duration). ``gap``: AQuA2's regMaskGap, default the one saved in the
    result."""
    e = aqua.gap if gap is None else int(gap)
    if e is None:
        raise ValueError("the result does not record regMaskGap: give gap= (AQuA2's default is 5)")
    hh, ww = aqua.x + 2 * e, aqua.y + 2 * e
    proj = np.zeros((hh, ww))
    for evt in aqua.events:
        rows = evt % aqua.x + e
        cols = (evt // aqua.x) % aqua.y + e
        u = np.unique(rows + cols * hh)
        proj[u % hh, u // hh] += 1
    return proj


def territory(results: str | Path | Sequence[str | Path], *, min_events: int = TERRITORY_MIN_EVENTS,
              gap: int | None = None, shape: tuple[int, int] | None = None) -> np.ndarray:
    """(H, W) boolean astrocyte territory: the pixels touched by at least ``min_events`` AQuA2 events, counted
    over every result in ``results`` (one file or the files of one imaging location's recordings).
    ``shape``: the movie's (H, W), checked against the results' frame."""
    files = [results] if isinstance(results, (str, Path)) else list(results)
    if not files:
        raise ValueError("no AQuA2 result given")
    total: np.ndarray | None = None
    for f in files:
        counts = event_counts(read_events(f), gap)
        if total is None:
            total = counts
        elif counts.shape != total.shape:
            raise ValueError(f"{f}: its frame is {counts.shape}, the others' {total.shape}")
        else:
            total += counts
    assert total is not None
    if shape is not None and total.shape != tuple(shape):
        raise ValueError(f"the AQuA2 results are on a {total.shape} frame, the movie {tuple(shape)}: run AQuA2 "
                         "on the movie ASTRID gets (or give gap=)")
    return total >= min_events
