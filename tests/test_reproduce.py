"""The reproduction folder: every dataset's settings make valid ASTRID settings, and the measures behave."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

import astrid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import metrics  # noqa: E402


def test_dataset_settings() -> None:
    cfg = yaml.safe_load((ROOT / "reproduce" / "datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    assert set(cfg) == {"D1", "D2_ABO", "D3_cai1", "D4_NAOMi", "D5_one_photon", "D6", "D7_location1",
                        "D7_location2"}
    for name, c in cfg.items():
        taus = [(c["tau_r"], c["tau_d"])] if not isinstance(c["tau_r"], dict) else \
            [(c["tau_r"][k], c["tau_d"][k]) for k in c["tau_r"]]
        for tau_r, tau_d in taus:
            s = astrid.Settings(tau_r=tau_r, tau_d=tau_d, **c.get("settings", {}))
            if "grid" in c:
                pairs = [tuple(p) for part in c["grid"] for p in (part if isinstance(part[0], list) else [part])]
                assert len(set(pairs)) == 19, name                     # 19 pairs on each public dataset
                for kd, knp in pairs:
                    astrid.Settings(tau_r=tau_r, tau_d=tau_d, kappa_d=kd, kappa_np=knp)
            else:
                assert s.kappa_d == 2.0 and s.kappa_np == 0.1          # the defaults on D1, D6 and D7


def test_measures() -> None:
    fps = 30.0
    t = np.arange(3000) / fps
    q = np.sin(2 * np.pi * 0.05 * t)
    assert metrics.error(q, q, fps) == 0.0
    assert abs(metrics.error(np.zeros_like(q) + 1e-9 * t, q, fps) - 1.0) < 1e-3     # a flat trace scores ~1
    assert abs(metrics.gain(2 * q, q) - 2) < 1e-12
    y = np.random.default_rng(0).normal(size=t.size)
    assert abs(metrics.leakage(y + q, y, q, fps) - 1.0) < 1e-12                      # the raw ROI average: 1
    assert metrics.leakage(y, y, q, fps) == 0.0
    assert not metrics.usable(np.ones(10)) and not metrics.usable(None) and metrics.usable(y)
    p = metrics.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert p == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_mit_code_never_imports_the_gpl_scorer() -> None:
    """reproduce/tuncat_eval is GPL-2 (a port of TUnCaT's evaluation): no MIT file may import it."""
    for path in [*(ROOT / "src").rglob("*.py"), *(ROOT / "reproduce" / "astrid_repro").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "import tuncat_eval" not in text and "from tuncat_eval" not in text, path
