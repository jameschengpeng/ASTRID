"""The runtime table (``reproduce/astrid_repro/runtime.py``) from small per-unit time files: the summaries, the
tests against ASTRID with Holm's correction over each dataset's other methods, the fastest method, and the
checks of the input files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import metrics, runtime  # noqa: E402


def write_times(folder: Path, drop: tuple[str, str] | None = None) -> Path:
    """Two datasets: D1 over 6 locations with the seven methods, D2 over 5 videos with ASTRID and three peers."""
    rng = np.random.default_rng(7)
    rows, videos = [], []
    for ds, kind, n, methods in (("D1", "location", 6, runtime.METHODS),
                                 ("D2", "video", 5, ("demix", "tuncat", "fissa", "suite2p"))):
        for i in range(n):
            unit = f"{ds}_u{i}"
            for k, m in enumerate(methods):
                if (ds, m) != drop or i:
                    rows.append({"dataset": ds, "unit_kind": kind, "unit": unit, "method": m,
                                 "seconds": float(10 + 3 * k + rng.normal(0, 1))})
            videos.append({"dataset": ds, "video": f"{unit}_v", "rois": 20 + 5 * i})
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "runtime_units.csv", index=False)
    pd.DataFrame(videos).to_csv(folder / "runtime_videos.csv", index=False)
    return folder


def test_table(tmp_path: Path) -> None:
    rec = runtime.runtime_table(write_times(tmp_path / "times"))
    assert list(rec) == ["D1", "D2"]
    d1, d2 = rec["D1"], rec["D2"]
    assert (d1["unit"], d1["units"], d1["videos"]) == ("location", 6, 6)
    assert d1["rois"] == {"median": 32.5, "min": 20.0, "max": 45.0}
    assert set(d1["tests"]) == set(runtime.METHODS) - {"demix"} and set(d2["tests"]) == {"tuncat", "fissa", "suite2p"}
    units = pd.read_csv(tmp_path / "times" / "runtime_units.csv", float_precision="round_trip")
    w = units[units.dataset == "D2"].pivot(index="unit", columns="method", values="seconds")
    d = (w.demix - w.fissa).to_numpy()
    t = d2["tests"]["fissa"]
    assert t["p"] == metrics.wilcoxon_p(d) and t["faster"] == int((d < 0).sum()) and t["mean_diff"] == d.mean()
    raw = {m: d2["tests"][m]["p"] for m in d2["tests"]}
    assert {m: d2["tests"][m]["p_holm"] for m in d2["tests"]} == metrics.holm(raw)
    assert d2["summary"]["demix"]["sd"] == pytest.approx(w.demix.std(ddof=1)) and d2["fastest"] == "demix"
    assert runtime.main(["--times", str(tmp_path / "times"), "--out", str(tmp_path / "out")]) == 0
    assert json.loads((tmp_path / "out" / "runtime.json").read_text())["D1"]["units"] == 6
    assert "Rebuilt from the times of the paper's runs" in (tmp_path / "out" / "runtime.md").read_text(encoding="utf-8")


def test_inputs_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lacks the time"):
        runtime.read_times(write_times(tmp_path / "a", drop=("D1", "cnmf")))
    folder = write_times(tmp_path / "b")
    units = pd.read_csv(folder / "runtime_units.csv")
    units.loc[0, "method"] = "other"
    units.to_csv(folder / "runtime_units.csv", index=False)
    with pytest.raises(ValueError, match="unknown"):
        runtime.read_times(folder)
