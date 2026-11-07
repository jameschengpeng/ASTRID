"""ASTRID's settings, in the paper's names, with the defaults of Supplementary Table S (tab:defaults).

=====================  =======  =========================================================================
Setting                Default  Meaning (paper section)
=====================  =======  =========================================================================
``tau_r``, ``tau_d``   none     rise and decay time constants (s) of the indicator's impulse response
                                (Supplementary, "Temporal models"); set per indicator
``kappa_d``            2        drive setting: the height, in noise s.d., a transient must reach before
                                any drive is assigned to it
``kappa_a``            10       astrocyte setting: a cosine coefficient above 2 f_a shrinks by 1/(1+kappa_a)
``kappa_np``           0.1      neuropil setting: smoothness of the neuropil footprint
``kappa_rho``          1        evidence setting: astrocyte footprint on pixels with weak structural
                                evidence (has no effect without an evidence map)
``f_a``                0.2 Hz   cutoff of the astrocytes' low-frequency prior (Supplementary, "Temporal models")
``neuropil_cap``       True     the neuropil footprint may not be higher on the ROI than around it
                                (Methods); False only to reproduce the model without it
``max_iterations``     600      iterations of the alternating updates, at most
``footprint_steps``    200      steps of the footprint update per iteration, at most
``drive_steps``        300      steps of each neuronal drive's update per iteration, at most
``check_every``        10       iterations between checks of convergence
``tolerance``          1e-3     tolerance of the check of convergence
``step_tolerance``     1e-7     relative decrease of the objective that ends a footprint update early
``baseline_bin_s``     1 s      the baseline F0 of each pixel: by default the 10th percentile of its 1-s
``baseline_window_s``  60 s     means in a moving 60-s window (``baseline_percentile`` of the means over
``baseline_percentile`` 10      bins of ``baseline_bin_s``, moving over ``baseline_window_s``); with
                                ``baseline_window_s`` None, one percentile over the whole recording,
``trend_pct_per_s``    0        along a line whose slope is ``trend_pct_per_s`` percent per second
                                (negative for a recording that dims; the paper's one-photon D7: 3-s
                                bins, no window, its locations' bleaching rates, -0.139 and -0.073)
=====================  =======  =========================================================================

The fixed geometry (margins, the number and size of astrocyte regions, the noise band) is in ``astrid.patch``.

**The code's names.** The solver (``astrid.model.Params``) calls the weights kappa_s (= kappa_d),
kappa_b (= kappa_np), kappa_t (= kappa_rho) and kappa_a (= kappa_a), as the analysis code did;
``Settings.params`` makes them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrid.model import Params
from astrid.patch import BaselineRule

PAPER_TO_CODE = {"kappa_d": "kappa_s", "kappa_np": "kappa_b", "kappa_rho": "kappa_t", "kappa_a": "kappa_a",
                 "max_iterations": "n_outer", "footprint_steps": "n_h", "drive_steps": "n_deconv",
                 "check_every": "block", "tolerance": "tol_target", "step_tolerance": "tol"}


@dataclass(frozen=True)
class Settings:
    """ASTRID's settings (module docstring). ``tau_r`` and ``tau_d`` have no default: give them here or to
    ``demix``."""
    tau_r: float | None = None
    tau_d: float | None = None
    kappa_d: float = 2.0
    kappa_a: float = 10.0
    kappa_np: float = 0.1
    kappa_rho: float = 1.0
    f_a: float = 0.2
    neuropil_cap: bool = True
    max_iterations: int = 600
    footprint_steps: int = 200
    drive_steps: int = 300
    check_every: int = 10
    tolerance: float = 1e-3
    step_tolerance: float = 1e-7
    baseline_bin_s: float = 1.0
    baseline_window_s: float | None = 60.0
    baseline_percentile: float = 10.0
    trend_pct_per_s: float = 0.0

    def __post_init__(self) -> None:
        self.baseline_rule()                         # refuses an invalid baseline
        for name in ("kappa_d", "kappa_a", "kappa_np", "kappa_rho", "f_a", "tolerance", "step_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")
        if not self.f_a > 0:
            raise ValueError(f"f_a must be > 0, got {self.f_a!r}")
        for name in ("max_iterations", "footprint_steps", "drive_steps", "check_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if not isinstance(self.neuropil_cap, bool):
            raise ValueError(f"neuropil_cap must be True or False, got {self.neuropil_cap!r}")
        if (self.tau_r is None) != (self.tau_d is None):
            raise ValueError("give tau_r and tau_d together")
        if self.tau_r is not None and self.tau_d is not None and not 0 < self.tau_r < self.tau_d:
            raise ValueError(f"need 0 < tau_r < tau_d (s), got {self.tau_r}, {self.tau_d}")

    def with_time_constants(self, tau_r: float | None, tau_d: float | None) -> Settings:
        """These settings with ``tau_r`` and ``tau_d`` where given; refused if neither defines them."""
        if tau_r is not None or tau_d is not None:
            if tau_r is None or tau_d is None:
                raise ValueError("give tau_r and tau_d together")
            return dataclasses.replace(self, tau_r=float(tau_r), tau_d=float(tau_d))
        if self.tau_r is None:
            raise ValueError("ASTRID needs the indicator's time constants: pass tau_r and tau_d (s), e.g. "
                             "tau_r=0.045, tau_d=0.142 for GCaMP6f (see the README)")
        return self

    def params(self, fps: float) -> Params:
        """The solver's parameters at the data's frame rate ``fps``."""
        if self.tau_r is None or self.tau_d is None:
            raise ValueError("tau_r and tau_d are not set")
        return Params(kappa_s=float(self.kappa_d), kappa_a=float(self.kappa_a), kappa_b=float(self.kappa_np),
                      kappa_t=float(self.kappa_rho), neuropil_cap=self.neuropil_cap, f_a=float(self.f_a),
                      tau_r=float(self.tau_r), tau_d=float(self.tau_d), fs=float(fps),
                      n_outer=self.max_iterations, n_h=self.footprint_steps, n_deconv=self.drive_steps,
                      tol=float(self.step_tolerance), block=self.check_every,
                      tol_target=float(self.tolerance))

    def baseline_rule(self) -> BaselineRule:
        return BaselineRule(float(self.baseline_bin_s),
                            None if self.baseline_window_s is None else float(self.baseline_window_s),
                            float(self.baseline_percentile), float(self.trend_pct_per_s))

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Settings:
        """Settings from a mapping in the paper's names; unknown keys are refused."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            hint = [PAPER_TO_CODE_INV[k] for k in unknown if k in PAPER_TO_CODE_INV]
            raise ValueError(f"unknown settings {unknown}" + (f" (the paper's names are {hint})" if hint else ""))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        import yaml
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError(f"{path}: not a mapping of settings")
        return cls.from_dict(values)


PAPER_TO_CODE_INV = {code: paper for paper, code in PAPER_TO_CODE.items() if code != paper}
