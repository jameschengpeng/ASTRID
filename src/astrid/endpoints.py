"""The routes into ASTRID besides ``demix`` itself; each ends in ``demix`` and returns its ``Result``.

- ``from_suite2p``: the ROIs and registered movie of a Suite2p run;
- ``from_caiman``: the ROIs (from the spatial footprints) and movie of a CaImAn run;
- ``territory_from_aqua2``: a movie and ROI masks, with the astrocyte territory from AQuA2's results;
- ``run_pipeline``: one imaging location as in the paper's Fig. 1A: its recordings, the ROI masks, AQuA2's
  results (the territory) and optionally a structural marker (the evidence map). Several recordings are fitted
  jointly, an option: in the paper each recording was fitted separately.

Every keyword that ``demix`` takes (``tau_r``, ``tau_d``, ``settings``, ``border_px``, ``workers``,
``export_components``, ...) passes through.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from astrid import aqua2, detection, evidence
from astrid.api import Result, as_movie, demix
from astrid.patch import TERRITORY_MIN_EVENTS
from astrid.readers import read_caiman, read_suite2p


def from_suite2p(folder: str | Path, *, um_per_px: float, cells_only: bool = True,
                 include_overlap: bool = False, movie: Any = None, fps: float | None = None,
                 **kwargs: Any) -> Result:
    """ASTRID on a Suite2p plane folder (``astrid.readers.read_suite2p``). Result row i is Suite2p ROI
    ``result.provenance['roi_index'][i]``. ``fps`` defaults to Suite2p's ``ops['fs']``."""
    inp = read_suite2p(folder, cells_only=cells_only, include_overlap=include_overlap, movie=movie)
    kwargs.setdefault("overlaps", "nearest" if include_overlap else "refuse")
    res = demix(inp.movie, inp.rois, fps=fps or inp.fps, um_per_px=um_per_px, **kwargs)
    res.provenance.update(source=inp.source, roi_index=inp.index.tolist())
    return res


def from_caiman(path: str | Path, *, um_per_px: float | None = None, movie: Any = None,
                accepted_only: bool = True, threshold: float = 0.9, fps: float | None = None,
                **kwargs: Any) -> Result:
    """ASTRID on a CaImAn result (``astrid.readers.read_caiman``). ``um_per_px`` defaults to the run's
    ``dxy`` and ``fps`` to its ``fr``. Overlapping masks go to the nearest centroid unless ``overlaps`` says
    otherwise. Result row i is CaImAn component ``result.provenance['roi_index'][i]``."""
    inp = read_caiman(path, movie=movie, accepted_only=accepted_only, threshold=threshold)
    um = um_per_px if um_per_px is not None else inp.um_per_px
    if um is None:
        raise ValueError(f"{path}: the run records no pixel size (params.data.dxy); pass um_per_px")
    kwargs.setdefault("overlaps", "nearest")
    res = demix(inp.movie, inp.rois, fps=fps or inp.fps, um_per_px=um, **kwargs)
    res.provenance.update(source=inp.source, roi_index=inp.index.tolist())
    return res


def territory_from_aqua2(movies: Any, rois: np.ndarray, aqua2_results: str | Path | Sequence[str | Path], *,
                         fps: float, um_per_px: float, min_events: int = TERRITORY_MIN_EVENTS,
                         gap: int | None = None, **kwargs: Any) -> Result:
    """ASTRID with the astrocyte territory from AQuA2's result files (``astrid.aqua2.territory``: pixels
    touched by at least ``min_events`` events over all the files)."""
    shape = tuple(np.asarray(rois).shape[1:])
    terr = aqua2.territory(aqua2_results, min_events=min_events, gap=gap, shape=shape)  # type: ignore[arg-type]
    res = demix(movies, rois, fps=fps, um_per_px=um_per_px, territory=terr, **kwargs)
    res.provenance.update(aqua2_results=[str(p) for p in ([aqua2_results] if isinstance(
        aqua2_results, (str, Path)) else aqua2_results)], min_events=int(min_events),
        territory_px=int(terr.sum()))
    return res


def detect_rois(movies: Any, *, fps: float, aqua2_results: Sequence[str | Path] | None = None,
                gap: int | None = None, zmap: Any = None) -> detection.LocationDetection:
    """The paper's ROI detection on one imaging location (``astrid.detection``): every recording detected on
    its own and the detections merged. ``aqua2_results``: one AQuA2 result file per movie, in the same order,
    for each recording's astrocyte mask (none: no astrocyte exclusion). Returns the merged ROI masks
    (``.rois``, (n, H, W)) and the map of all candidates (``.candidates``) that ``demix`` takes."""
    mlist = movies if isinstance(movies, (list, tuple)) else [movies]
    mv = [as_movie(x, f"rec{i}") for i, x in enumerate(mlist)]
    shape = mv[0].shape()[1:]
    if aqua2_results is None:
        masks = [np.zeros(shape, dtype=bool) for _ in mv]
    else:
        files = [aqua2_results] if isinstance(aqua2_results, (str, Path)) else list(aqua2_results)
        if len(files) != len(mv):
            raise ValueError("detection needs one AQuA2 result per movie, in the same order")
        masks = [detection.astrocyte_mask(aqua2.event_counts(aqua2.read_events(f), gap)) for f in files]
        if any(m.shape != shape for m in masks):
            raise ValueError(f"the AQuA2 results' frame differs from the movies' {shape}")
    with ExitStack() as stack:
        arrays = [stack.enter_context(m._open()) for m in mv]
        return detection.detect_location(arrays, fps, masks, zmap)


def run_pipeline(movies: Any, rois: np.ndarray | None = None, *, fps: float, um_per_px: float,
                 aqua2_results: str | Path | Sequence[str | Path] | None = None,
                 marker: Any = None, candidates: np.ndarray | None = None,
                 min_events: int = TERRITORY_MIN_EVENTS, gap: int | None = None,
                 marker_margin_px: int = evidence.MARGIN_PX, out: str | Path | None = None,
                 **kwargs: Any) -> Result:
    """One imaging location, as in the paper's Fig. 1A.

    - ``movies``: the location's recordings (one or a list), registered to one pixel grid. Several recordings are
      fitted jointly, each ROI's spatial footprints shared across them. In the paper each recording was fitted
      separately (one call per recording); joint fitting is an option the paper did not use or evaluate;
    - ``rois``: the ROI masks (from any detection); None: detected here with the paper's detection
      (``detect_rois``, which then needs one AQuA2 result per movie for the astrocyte masks), whose candidates
      become ``candidates`` unless given;
    - ``aqua2_results``: AQuA2's result files for the location's recordings: the territory is the pixels touched
      by >= ``min_events`` events over all of them;
    - ``marker``: a structural astrocyte marker, as mean images or movies (one per recording, or one), from
      which the evidence map is made (``astrid.evidence.evidence_map``);
    - ``candidates``: a map of candidate cells that did not become ROIs (unclassified components).
    ``out``: also save the result, the territory, the evidence map (and detected ROIs) there."""
    detected = None
    if rois is None:
        aq = None if aqua2_results is None else (
            [aqua2_results] if isinstance(aqua2_results, (str, Path)) else list(aqua2_results))
        detected = detect_rois(movies, fps=fps, aqua2_results=aq, gap=gap)
        rois = detected.rois
        if candidates is None:
            candidates = detected.candidates
    shape = tuple(np.asarray(rois).shape[1:])
    terr = (aqua2.territory(aqua2_results, min_events=min_events, gap=gap, shape=shape)  # type: ignore[arg-type]
            if aqua2_results is not None else None)
    rho = None
    if marker is not None:
        items = marker if isinstance(marker, (list, tuple)) else [marker]
        images = []
        for i, m in enumerate(items):
            arr = m if not isinstance(m, (str, Path)) else None
            if arr is None or len(np.shape(arr)) == 3:
                mv = as_movie(m, f"marker{i}")
                with mv._open() as d:
                    images.append(evidence.mean_image(d))
            else:
                images.append(np.asarray(arr, dtype=np.float64))
        rho = evidence.evidence_map(images, margin_px=marker_margin_px)
    res = demix(movies, rois, fps=fps, um_per_px=um_per_px, territory=terr, evidence=rho,
                candidates=candidates, **kwargs)
    res.provenance.update(pipeline=True, min_events=int(min_events), rois_detected=detected is not None,
                          territory_px=None if terr is None else int(terr.sum()))
    if out is not None:
        d = Path(out)
        res.save(d)
        extra = {} if detected is None else {"rois": detected.rois, "candidates": detected.candidates}
        np.savez_compressed(d / "inputs.npz", territory=terr if terr is not None else np.zeros(shape, bool),
                            evidence=rho if rho is not None else np.zeros(shape), **extra)
    return res
