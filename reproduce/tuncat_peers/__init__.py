"""The peer methods' reruns on D2, D4 and D5: FISSA, TUnCaT and CNMF run with the scripts of TUnCaT's
paper-reproduction archive, changed only as the paper's runs changed them.

**License: GNU General Public License, version 2 (``LICENSE`` in this folder), not MIT.** The scripts it runs are
those of TUnCaT's paper-reproduction archive (Bao Y, Redington E, Agarwal A, Gong Y. Decontaminate traces from
fluorescence calcium imaging videos using targeted non-negative matrix factorization. Frontiers in Neuroscience
15:797421 (2022); https://github.com/YijunBao/TUnCaT_paper_reproduction, also in its figshare archive), which is
GPL-2. This folder holds no copy of them but makes and records changes to them (``staging``), so it keeps their
license and is kept apart from the MIT code, which never imports it. Download the archive's scripts, install the
methods (FISSA 1.0.0 and the archive's TUnCaT v1.0 with numba in one environment; CaImAn 1.13.2 in another), and
run ``python -m tuncat_peers`` from ``reproduce/`` in the method's environment (``__main__``).

Why the reruns: the traces the archive releases come from the authors' own runs, with older versions of the
methods. The paper reran the scripts. On the video compared, its FISSA and TUnCaT traces match the released ones
(to 1e-14 and 8e-5), but its CNMF traces differ slightly (a median correlation of 0.9993 per neuron, 5 of 60
neurons below 0.99), since CaImAn has changed. The released traces remain a close second route.

What a rerun gives: run through this folder in the paper's environments (one D2 video, all settings), FISSA's
and CNMF's traces were the paper's bit for bit. TUnCaT's unmixing is not deterministic from run to run: its raw
and background traces were the paper's bit for bit, but its NMF ended a few iterations apart per neuron, so its
traces came within 2e-4 (relative) of the paper's, and its F1 within 0.006 at single thresholds (the best F1 of
10 of its 11 settings unchanged).

The changes (``staging``) touch no numerics: the platform's paths, Python 3.10's shared memory, FISSA's cache
folders, CaImAn's renamed calls, and CNMF's number of background components (1, as in the authors' released
traces; the script says 2). The lines they insert carry the marker ``# peers.sbatch``, the name of the job that
first applied them, so that the changed scripts are byte for byte those of the paper's runs.

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation; version 2 of the License. This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
"""
