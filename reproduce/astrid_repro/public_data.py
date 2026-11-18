"""Readers for the public datasets D2-D5 and their ground truths.

D2 (ABO), D4 (NAOMi) and D5 (one-photon) come in the layout of TUnCaT's figshare archive (Bao et al. 2022):

- ``data/<ds>/<video>.h5``: the movie (``mov``, (T, H, W));
- ``data/<ds>/GT Masks/FinalMasks_<video>.mat``: the ROI masks every method received;
- ``data/<ds>/GT transients/output_<video>.mat`` (D2, D5): the manual transient labels, the ground truth;
- ``data/NAOMi/GT Masks/Traces_etc_<video>.mat`` (D4): the simulation's clean traces;
- ``results/<ds>/unmixed traces``: the peers' traces as their authors' scripts write them (a rerun of those
  scripts writes the same layout).

D4's ground-truth transients are derived from its clean traces, and the F1 of D2, D4 and D5 is computed, by
TUnCaT's evaluation code (GPL-2). It is not part of this package; ``cv.f1_grid`` takes it as a scorer.

D3 (CRCNS cai-1, Chen et al. 2013): the ground truth is the calcium predicted from the recorded spikes
(``predicted_calcium``, from the processed files), and the score is the correlation with it over the scored
frames (``corr``). Every method read the same export: per recording ``<name>.h5`` with ``mov`` (registered,
binned 4 x 4), ``roi``, ``valid`` (the scored frames), ``predicted``, ``fmean_roi`` and ``fmean_neuropil``, all
listed in ``index.json``. ASTRID's trace is ΔF/F0, its running baseline removed by the fit. In the
equal-baseline scoring, every other trace has its own running baseline removed by the same rule
(``minus_baseline``) before the correlation.

ASTRID's traces are read as ``astrid.Result.save`` writes them, one folder per setting:
``<root>/<ds>/<setting>/<video>/result.npz``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io

from astrid.patch import baseline as astrid_baseline
from astrid_repro.cv import CNMF_GRID, cnmf_setting

ABO_VIDEOS = ("501484643", "501574836", "501729039", "502608215", "503109347",
              "510214538", "524691284", "527048992", "531006860", "539670003")
NAOMI_VIDEOS = tuple(f"Video_{k}" for k in range(10))
ONE_PHOTON_VIDEOS = ("c25_59_228", "c27_12_326", "c28_83_210", "c25_163_267", "c27_114_176", "c28_161_149",
                     "c25_123_348", "c27_122_121", "c28_163_244")     # TUnCaT's order
VIDEOS = {"ABO": ABO_VIDEOS, "NAOMi": NAOMI_VIDEOS, "1p": ONE_PHOTON_VIDEOS}


def one_photon_recording(video: str) -> str:
    """The recording a D5 sub-video was cropped from: the first part of its name. The nine sub-videos are
    crops of three one-photon recordings, c25, c27 and c28 (Bao et al. 2022, Methods)."""
    if video not in ONE_PHOTON_VIDEOS:
        raise ValueError(f"{video} is not one of D5's sub-videos")
    return video.split("_")[0]
PAPER_NAMES = {"ABO": "D2", "cai1": "D3", "NAOMi": "D4", "1p": "D5"}
UM_PER_PX = {"ABO": 0.785, "NAOMi": 1.0, "1p": 3.5}                  # the paper's Supplementary Table S2

CAI1_FOLDERS = {"GCaMP6f": "GCaMP6f_11cells_Chen2013", "GCaMP6s": "GCaMP6s_9cells_Chen2013"}
# published single-spike responses (Chen et al. 2013): time to peak and half-decay, in seconds
CAI1_KERNELS_S = {"GCaMP6f": (0.045, 0.142), "GCaMP6s": (0.179, 0.550)}
CAI1_SERIES = ["fmean_roi", "fmean_neuropil", "raw_ephys", "filtered_ephys", "detected_spikes"]


# ---- D2, D4 and D5: TUnCaT's figshare layout ----

def masks_path(archive: str | Path, ds: str, video: str) -> Path:
    return Path(archive) / "data" / ds / "GT Masks" / f"FinalMasks_{video}.mat"


def labels_path(archive: str | Path, ds: str, video: str) -> Path:
    return Path(archive) / "data" / ds / "GT transients" / f"output_{video}.mat"


def movie_path(archive: str | Path, ds: str, video: str) -> Path:
    return Path(archive) / "data" / ds / f"{video}.h5"


def naomi_truth_path(archive: str | Path, video: str) -> Path:
    return Path(archive) / "data" / "NAOMi" / "GT Masks" / f"Traces_etc_{video}.mat"


def released_root(archive: str | Path, ds: str) -> Path:
    """The peers' traces as released in the archive."""
    return Path(archive) / "results" / ds / "unmixed traces"


def read_masks(path: str | Path, key: str = "FinalMasks") -> np.ndarray:
    """(n, H, W) boolean masks from a .mat file: MATLAB v5 ((H, W, n), transposed as TUnCaT's scripts do after
    ``loadmat``) or v7.3 (HDF5, (n, H, W) as stored)."""
    try:
        arr = np.asarray(scipy.io.loadmat(path)[key])
        if arr.ndim == 3:
            arr = arr.transpose(2, 1, 0)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            arr = np.asarray(f[key])
    return arr.astype(bool)


def read_labels(path: str | Path) -> list[np.ndarray | None]:
    """The manual labels: per mask, an (m, 3) int array of [start, end, valid] (1-based frames, inclusive;
    valid 1 for a true transient), or None for a mask that was not labelled (not scored)."""
    cells = scipy.io.loadmat(path, squeeze_me=False)["output"]
    out: list[np.ndarray | None] = []
    for entry in np.asarray(cells).ravel():
        a = np.asarray(entry)
        out.append(None if a.size == 0 else a.reshape(-1, 3).astype(int))
    return out


def read_naomi_truth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """D4's simulation truth: the clean calcium traces (n, T) and the spikes on the 100-Hz base (n, 10 T)."""
    d = scipy.io.loadmat(path)
    return np.asarray(d["clean_traces"], dtype=float), np.asarray(d["spikes"], dtype=float)


def _mat(p: Path) -> dict[str, Any]:
    return {k: v for k, v in scipy.io.loadmat(p).items() if not k.startswith("__")}


def merge_settings(parts: list[dict[str, np.ndarray]], where: str) -> dict[str, np.ndarray]:
    """Settings read from several roots (a run and its extension), merged in root order; a setting found
    under two roots is refused."""
    out: dict[str, np.ndarray] = {}
    for part in parts:
        dup = sorted(set(out) & set(part))
        if dup:
            raise ValueError(f"{where}: settings found under more than one root: {dup}")
        out.update(part)
    return out


def _alpha_dirs(folder: Path) -> list[tuple[float, Path]]:
    out = []
    for d in sorted(folder.iterdir()) if folder.exists() else []:
        if d.is_dir() and d.name.startswith("alpha="):
            out.append((float(d.name.split("=", 1)[1]), d))
    return sorted(out)


CnmfFill = Callable[[dict[str, Any], np.ndarray], np.ndarray]


def peer_traces(root: str | Path | Sequence[str | Path], method: str, video: str,
                masks: np.ndarray | None = None, cnmf_fill: CnmfFill | None = None) -> dict[str, np.ndarray]:
    """A peer method's traces in the layout of its authors' scripts: setting name -> (n, T).

    - "FISSA", "TUnCaT": one per alpha ("a=0.1"), ``unmixed_traces`` of ``traces_FISSA_Raw/alpha=<a>/`` and
      ``traces_nmfdemix`` (transposed) of ``traces_ours_Raw/alpha=<a>/``;
    - "CNMF": "p1", C + YrA of ``traces_CNMF_Raw_p1``;
    - "raw": the ROI means; "bgsubs": the ROI means minus TUnCaT's background traces; "allen": the Allen SDK's.

    Several roots: their settings merged in root order. CaImAn can drop seeded components: with ``masks``
    given, a CNMF result with fewer traces than masks needs ``cnmf_fill(mat, masks)`` -> (n, T). TUnCaT's
    evaluation gives the dropped ROIs zero traces; that rule is not part of this package.
    """
    if not isinstance(root, (str, Path)):
        return merge_settings([peer_traces(r, method, video, masks, cnmf_fill) for r in root],
                              f"{method} {video}")
    root = Path(root)
    out: dict[str, np.ndarray] = {}
    if method == "FISSA":
        for a, d in _alpha_dirs(root / "traces_FISSA_Raw"):
            p = d / f"{video}.mat"
            if p.exists():
                out[f"a={a:g}"] = np.asarray(_mat(p)["unmixed_traces"], dtype=float)
    elif method == "TUnCaT":
        for a, d in _alpha_dirs(root / "traces_ours_Raw"):
            p = d / f"{video}.mat"
            if p.exists():
                out[f"a={a:g}"] = np.asarray(_mat(p)["traces_nmfdemix"], dtype=float).T
    elif method == "CNMF":
        p = root / "traces_CNMF_Raw_p1" / f"{video}.mat"
        if p.exists():
            out["p1"] = _cnmf_traces(p, video, masks, cnmf_fill)
    elif method in ("raw", "bgsubs"):
        p = root / "traces_ours_Raw" / "raw" / f"{video}.mat"
        if p.exists():
            m = _mat(p)
            x = np.asarray(m["traces"], dtype=float).T
            out[method] = x if method == "raw" else x - np.asarray(m["bgtraces"], dtype=float).T
    elif method == "allen":
        p = root / "traces_AllenSDK_Raw" / f"{video}.mat"
        if p.exists():
            out["allen"] = np.asarray(_mat(p)["unmixed_traces"], dtype=float)
    else:
        raise ValueError(f"unknown method {method}")
    return out


def ours_traces(root: str | Path | Sequence[str | Path], dataset: str, video: str,
                key: str = "dff") -> dict[str, np.ndarray]:
    """ASTRID's traces, ``<root>/<dataset>/<setting>/<video>/result.npz`` (``key``, default the dF/F0, NaN rows
    for ROIs not fitted): setting -> (n, T). Several roots: their settings merged in root order."""
    if not isinstance(root, (str, Path)):
        return merge_settings([ours_traces(r, dataset, video, key) for r in root], f"ours {dataset}/{video}")
    root = Path(root)
    out: dict[str, np.ndarray] = {}
    base = root / dataset
    for d in sorted(base.iterdir()) if base.exists() else []:
        p = d / video / "result.npz"
        if d.is_dir() and p.exists():
            with np.load(p) as z:
                recs = [k for k in z.files if k.startswith(f"{key}__")]
                if len(recs) != 1:
                    raise ValueError(f"{p}: expected one recording, found {recs}")
                out[d.name] = np.asarray(z[recs[0]], dtype=float)
    return out


def npz_trace(path: str | Path, key: str = "trace") -> np.ndarray:
    """One array of an .npz file as float64 (Suite2p's F - 0.7 Fneu, saved as ``trace`` (ROIs, frames))."""
    with np.load(path) as z:
        return np.asarray(z[key], dtype=float)


def suite2p_path(root: str | Path, ds: str, video: str) -> Path:
    """Where Suite2p's trace of a video was saved: ``<root>/<ds>/<video>/suite2p/<video>.npz``."""
    return Path(root) / ds / video / "suite2p" / f"{video}.npz"


# ---- D3: CRCNS cai-1 ----

def cai1_processed_path(root: str | Path, indicator: str, cell: str, rec: str) -> Path:
    """The processed file of a recording (cell ``<date>_cell<k>``, recording ``<nnn>``)."""
    date, k = cell.split("_")
    return Path(root) / CAI1_FOLDERS[indicator] / "processed_data" / f"data_{date}_{k}_{rec}.mat"


def read_cai1_processed(path: str | Path, indicator: str | None = None) -> dict[str, Any]:
    """A processed file's traces and time bases: ``fmean_roi`` and ``fmean_neuropil`` (the dataset authors'
    ROI and neuropil means), ``t_frame``, ``t_ephys``, ``spike_times`` (s) and the indicator (``indicator``
    where the file does not say)."""
    obj = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["obj"]
    v = obj.timeSeriesArrayHash.value
    names = [str(k).replace(" ", "_") for k in obj.timeSeriesArrayHash.keyNames]
    if names[:5] != CAI1_SERIES:
        raise ValueError(f"{path}: time series {names}, expected {CAI1_SERIES}")
    spikes = np.asarray(v[4].valueMatrix).astype(bool)
    t_e = np.asarray(v[3].time, dtype=float)
    return {"fmean_roi": np.asarray(v[0].valueMatrix, dtype=float),
            "fmean_neuropil": np.asarray(v[1].valueMatrix, dtype=float),
            "t_frame": np.asarray(v[0].time, dtype=float), "t_ephys": t_e,
            "spike_times": t_e[spikes],
            "indicator": str(getattr(obj, "indicator", indicator))}


def single_spike_kernel(indicator: str, fps: float, length_s: float = 4.0) -> np.ndarray:
    """A difference of exponentials with the published time to peak and half-decay (``CAI1_KERNELS_S``), unit
    peak, on the frame grid."""
    t_peak, t_half = CAI1_KERNELS_S[indicator]
    tau_d = t_half / np.log(2)
    taus = np.linspace(1e-3, tau_d * 0.999, 4000)          # the rise constant, from the time to peak
    tp = tau_d * taus / (tau_d - taus) * np.log(tau_d / taus)
    tau_r = float(taus[np.argmin(np.abs(tp - t_peak))])
    t = np.arange(0, length_s, 1 / fps)
    k = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    return k / k.max()


def predicted_calcium(spike_times: np.ndarray, t_frame: np.ndarray, indicator: str) -> np.ndarray:
    """D3's ground truth: the spikes binned on the frame grid, convolved with the indicator's single-spike
    kernel."""
    fps = 1 / float(np.median(np.diff(t_frame)))
    edges = np.concatenate([t_frame - 0.5 / fps, [t_frame[-1] + 0.5 / fps]])
    counts, _ = np.histogram(spike_times, bins=edges)
    return np.convolve(counts.astype(float), single_spike_kernel(indicator, fps))[:len(t_frame)]


def read_cai1_index(export: str | Path) -> list[dict[str, Any]]:
    """The export's recordings (``name``, ``cell``, ``rec``, ``indicator``, ``fps``, ...), in order."""
    return json.loads((Path(export) / "index.json").read_text(encoding="utf-8"))


def read_cai1_scored(export: str | Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A recording's predicted calcium and its scored frames (present and within the electrophysiology)."""
    with h5py.File(Path(export) / f"{name}.h5", "r") as f:
        return np.asarray(f["predicted"], dtype=float), np.asarray(f["valid"], dtype=bool)


def read_cai1_fps(export: str | Path, name: str) -> float:
    """A recording's frame rate, as its export holds it (``fps``)."""
    with h5py.File(Path(export) / f"{name}.h5", "r") as f:
        return float(f.attrs["fps"])


def minus_baseline(x: np.ndarray, fps: float) -> np.ndarray:
    """A trace minus its running baseline under ASTRID's default rule (``astrid.patch.baseline``: the 10th
    percentile of 1-s means in a moving 60-s window, interpolated to the frames), for the equal-baseline
    scorings of D3 (``tables_public cai1 --baseline equal``) and D6 (``stats d6-scores --baseline equal``). A trace
    with a non-finite frame is refused."""
    y = np.asarray(x, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("a trace with non-finite frames: its 1-s means are undefined there")
    return y - astrid_baseline(y[None], fps)[0]


def _cnmf_traces(p: Path, video: str, masks: np.ndarray | None, cnmf_fill: CnmfFill | None) -> np.ndarray:
    """C + YrA of one CNMF result; a result with fewer traces than ``masks`` needs ``cnmf_fill`` (``peer_traces``)."""
    m = _mat(p)
    c = np.asarray(m["C_gt"], dtype=float)
    if masks is not None and c.shape[0] < len(masks):
        if cnmf_fill is None:
            raise ValueError(f"CNMF {video}: {c.shape[0]} traces for {len(masks)} masks; the dropped "
                             "components need cnmf_fill")
        return np.asarray(cnmf_fill(m, np.asarray(masks, dtype=bool)), dtype=float)
    return c + np.asarray(m["YrA_gt"], dtype=float)


def cnmf_grid_traces(grid: str | Path, dataset: str, video: str, masks: np.ndarray | None = None,
                     cnmf_fill: CnmfFill | None = None) -> dict[str, np.ndarray]:
    """CNMF's cross-validation grid on D2, D4 or D5 (``cv.CNMF_GRID``): each setting a run of ``tuncat_peers``
    into its own folder, ``<grid>/gnb<g>_p<p>/<dataset>/traces_CNMF_Raw_p<p>/<video>.mat`` -> setting
    "gnb<g>_p<p>" -> (n, T) traces as ``peer_traces`` reads CNMF's, in the grid's order (which breaks ties in the
    cross-validation). A setting without the video is left out."""
    out: dict[str, np.ndarray] = {}
    for g, q in CNMF_GRID:
        p = Path(grid) / cnmf_setting(g, q) / dataset / f"traces_CNMF_Raw_p{q}" / f"{video}.mat"
        if p.exists():
            out[cnmf_setting(g, q)] = _cnmf_traces(p, video, masks, cnmf_fill)
    return out


def cai1_cnmf_grid(grid: str | Path, name: str) -> dict[str, np.ndarray]:
    """CNMF's cross-validation grid on D3 (``cv.CNMF_GRID``): ``<grid>/gnb<g>_p<p>/cai1/<name>.npz``, written by
    ``cai1_peers --cnmf-gnb <g> --cnmf-p <p>`` -> setting -> trace (``cnmf``)."""
    out: dict[str, np.ndarray] = {}
    for g, q in CNMF_GRID:
        p = Path(grid) / cnmf_setting(g, q) / "cai1" / f"{name}.npz"
        if p.exists():
            with np.load(p) as z:
                if "cnmf" in z.files:
                    out[cnmf_setting(g, q)] = z["cnmf"]
    return out


def corr(x: np.ndarray, y: np.ndarray, ok: np.ndarray) -> float:
    """D3's score: the Pearson correlation of a trace with the predicted calcium over the scored frames (NaN
    with fewer than 10 such frames or a constant trace)."""
    n = min(x.size, y.size, ok.size)
    m = ok[:n] & np.isfinite(x[:n]) & np.isfinite(y[:n])
    if m.sum() < 10 or x[:n][m].std() == 0 or y[:n][m].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[:n][m], y[:n][m])[0, 1])


def cai1_traces(method: str, name: str, peers: str | Path | None = None,
                ours: Sequence[str | Path] | None = None) -> dict[str, np.ndarray]:
    """One recording's traces of a method: setting -> (T,). "ours": ASTRID's dF/F0 from
    ``<root>/cai1/<setting>/<name>/result.npz`` over the roots (a setting under two roots is refused);
    "TUnCaT", "FISSA" (one per alpha) and "CNMF": ``<peers>/<name>.npz`` (keys ``tuncat_a=<a>``,
    ``fissa_a=<a>``, ``cnmf``)."""
    if method == "ours":
        out: dict[str, np.ndarray] = {}
        for root in ours or []:
            base = Path(root) / "cai1"
            for d in sorted(base.iterdir()) if base.exists() else []:
                p = d / name / "result.npz"
                if p.exists():
                    if d.name in out:
                        raise ValueError(f"{name}: setting {d.name} found under more than one root")
                    with np.load(p) as z:
                        key = [k for k in z.files if k.startswith("dff__")][0]
                        out[d.name] = np.asarray(z[key], dtype=float)[0]
        return out
    p = Path(peers or ".") / f"{name}.npz"
    if not p.exists():
        return {}
    with np.load(p) as z:
        if method == "TUnCaT":
            return {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith("tuncat_a=")}
        if method == "FISSA":
            return {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith("fissa_a=")}
        if method == "CNMF":
            return {"cnmf": z["cnmf"]} if "cnmf" in z.files else {}
    return {}
