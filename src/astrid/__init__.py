"""ASTRID: ASTRocyte-Informed Demixing of neuronal calcium traces.

The endpoints (each returns an ``astrid.Result``):

- ``demix(movies, rois, *, fps, um_per_px, tau_r, tau_d, territory=None, evidence=None, ...)``
- ``from_suite2p(folder, *, um_per_px, tau_r, tau_d, ...)``
- ``from_caiman(path, *, tau_r, tau_d, ...)``
- ``territory_from_aqua2(movies, rois, aqua2_results, *, fps, um_per_px, tau_r, tau_d, ...)``
- ``run_pipeline(movies, rois, *, fps, um_per_px, tau_r, tau_d, aqua2_results=None, marker=None, ...)``

``detect_rois(movies, *, fps, aqua2_results=None)`` is the paper's ROI detection (``astrid.detection``).
``Settings`` holds the model's settings in the paper's names, with its defaults. See the README.
"""

__version__ = "0.1.0.dev0"

from astrid.api import Result, component_footprint, demix  # noqa: E402
from astrid.endpoints import (  # noqa: E402
    detect_rois,
    from_caiman,
    from_suite2p,
    run_pipeline,
    territory_from_aqua2,
)
from astrid.settings import Settings  # noqa: E402

__all__ = ["Result", "Settings", "__version__", "component_footprint", "demix", "detect_rois", "from_caiman",
           "from_suite2p", "run_pipeline", "territory_from_aqua2"]
