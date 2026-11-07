"""TUnCaT's transient-detection evaluation, ported to Python: the F1 scorer of the paper's D2, D4 and D5.

**License: GNU General Public License, version 2 (``LICENSE`` in this folder), not MIT.** This folder is a
line-by-line port of the MATLAB evaluation code of TUnCaT (Bao Y, Redington E, Agarwal A, Gong Y.
Decontaminate traces from fluorescence calcium imaging videos using targeted non-negative matrix factorization.
Frontiers in Neuroscience 15:797421 (2022); https://github.com/YijunBao/TUnCaT), which is licensed under GPL-2.
A port of GPL-2 code keeps that license, so this folder is GPL-2 and is kept apart from the rest of the
repository. The ``astrid`` package and ``reproduce/astrid_repro`` are MIT and never import it: the scorer is
handed to them at run time (``astrid_repro.tables_public grids --scorer tuncat_eval.evaluation:scorer
--cnmf-fill tuncat_eval.evaluation:cnmf_fill``).

Ported (``transients.py``, ``evaluation.py``), each with the TUnCaT file it follows:
- ``evaluation/GetPerformance_SpikeDetection_split.m``, ``SNR_normalization.m`` and ``noise_PSD.m``: the
  baseline, the noise, the transient detection and the matching (``transients``);
- ``GT_transient_NAOMi_split.m``: the ground-truth transients of the NAOMi videos (D4) from their clean traces;
- the template step of the ``eval_*.m`` scripts: the indicator kernel the traces are filtered with;
- ``eval_NAOMi_CNMF.m``'s handling of components that CaImAn removed (``evaluation.cnmf_fill``).

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation; version 2 of the License. This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
"""
