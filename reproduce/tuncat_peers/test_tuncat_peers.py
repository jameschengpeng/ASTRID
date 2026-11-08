"""Tests of the staging of TUnCaT's archive scripts on synthetic script text (GPL-2, as the rest of this folder)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tuncat_peers import __main__ as M  # noqa: E402
from tuncat_peers import staging as S  # noqa: E402

TUNCAT = """\
    dir_video = '..\\\\data\\\\ABO\\\\'
    list_Exp_ID = ['501484643','501574836',
                   '501729039']
    if FinalMasks.sum()*T >= 7e7:
        traces = shm_branch()
    shm_video.close()
    shm_video.unlink()
    shm_masks.close()
    shm_masks.unlink()
"""
FISSA = ("        for alpha in list_alpha:\n"
         '            folder_alpha_out = "alpha={:6.3f}".format(alpha)\n'
         "            experiment = fissa.Experiment([video], [FinalMasks.tolist()], folder_FISSA, alpha=alpha, "
         "ncores_preparation=1)\n"
         "            experiment.separate(redo_prep=False, redo_sep=True)\n")
CNMF = """\
from past.utils import old_div
from skimage.external.tifffile import TiffFile
sys.path.insert(0, 'C:\\\\Other methods\\\\CaImAn')
Names_raw = glob.glob(dir_video_SNR+'\\\\*_memmap__*.mmap')
radius = np.int(np.median(np.sqrt(area)))
cnm = cnmf.CNMF(Ain=A_in.astype(np.bool), 'gnb': 2,
cnm = cnm.fit(images)
"""


def test_to_posix_and_filter() -> None:
    s = S.to_posix(TUNCAT)
    assert "dir_video = '../data/ABO/'" in s
    assert S.to_posix(CNMF) == CNMF          # doubled backslashes, as in the archive: left as they are
    single = "glob(d+'\\*_memmap__*.mmap'); x.rsplit('\\', 1)"       # single backslashes in the text
    assert S.to_posix(single) == "glob(d+'/*_memmap__*.mmap'); x.rsplit('/', 1)"
    assert S.filter_list(TUNCAT, "501729039").count("list_Exp_ID = ['501729039']") == 1
    naomi = "    list_Exp_ID = ['Video_'+str(x) for x in list(range(0,10))]\n"
    assert "list_Exp_ID = ['Video_3']" in S.filter_list(naomi, "Video_3")


def test_tuncat_branch_and_cleanup() -> None:
    s = S.patch_tuncat(TUNCAT)
    assert "if False:  # numba branch forced (peers.sbatch)" in s
    assert s.count("    try: shm_") == 4 and s.count("    except Exception: pass  # peers.sbatch") == 4
    with pytest.raises(ValueError, match="shared-memory branch"):
        S.patch_tuncat("nothing to patch")


def test_fissa_cache_and_alphas() -> None:
    s = S.patch_fissa(FISSA)
    assert "folder_alpha = folder_FISSA + '_alpha={:6.3f}'.format(alpha).replace(' ', '')" in s
    assert "fissa.Experiment([video], [FinalMasks.tolist()], folder_alpha," in s
    assert "redo_prep=True, redo_sep=True" in s
    t = S.fissa_alphas("        list_alpha = [0.01, 0.1]\n" + s, [0.0005])
    assert "list_alpha = [0.0005]  # peers.sbatch: FISSA_ALPHAS" in t
    assert '"alpha={:g}".format(alpha)' in t and "'_alpha={:g}'.format(alpha)" in t


def test_cnmf_lines_and_background() -> None:
    s = S.cnmf_gnb(S.patch_cnmf(CNMF), 1)
    for line in ("pass  # peers.sbatch: unused Python-2 shim dropped", "from tifffile import TiffFile",
                 "radius = int(np.median(", "Ain=A_in.astype(bool),", "'gnb': 1,  # peers.sbatch: CNMF_GNB",
                 "cnm.fit(images)  # peers.sbatch"):
        assert line in s
    with pytest.raises(ValueError, match="exactly one"):
        S.patch_cnmf("radius = np.int(np.median(")


def test_stage(tmp_path: Path) -> None:
    scripts, data = tmp_path / "repro", tmp_path / "archive"
    for sub, name, text in (("TUnCaT archive", "TUnCaT_multi_ABO.py", TUNCAT),
                            ("Other methods/FISSA", "FISSA_ABO.py", FISSA),
                            ("Other methods/CNMF_CaImAn", "CNMF_unmixing_seeded_ABO.py", CNMF),
                            ("Other methods/CNMF_CaImAn", "h5ToMmap.py", "datadir = '..\\\\..\\\\data\\\\*\\\\'\n"),
                            ("template", "t.mat", "")):
        (scripts / sub).mkdir(parents=True, exist_ok=True)
        (scripts / sub / name).write_text(text, encoding="utf-8")
    (data / "data" / "ABO" / "GT Masks").mkdir(parents=True)
    (data / "data" / "ABO" / "GT Masks" / "FinalMasks_1.mat").write_bytes(b"m")
    (data / "data" / "ABO" / "1.h5").write_bytes(b"movie")
    script = M.stage(scripts, data, "ABO", "cnmf", tmp_path / "stage", posix=True)
    text = script.read_text(encoding="utf-8")
    assert "'gnb': 1," in text and "sys.path.insert(0, 'C:\\\\Other methods\\\\CaImAn')" in text     # unchanged
    assert "../../data/*/" in (script.parent / "h5ToMmap.py").read_text(encoding="utf-8")
    assert (tmp_path / "stage" / "data" / "ABO" / "1.h5").read_bytes() == b"movie"
    assert (tmp_path / "stage" / "data" / "ABO" / "GT Masks" / "FinalMasks_1.mat").exists()
    assert (tmp_path / "stage" / "TUnCaT archive" / "TUnCaT_multi_ABO.py").read_text(encoding="utf-8") == TUNCAT
    assert M.arguments("tuncat", "Raw") == ["Raw", "1", "0", "1", "1"] and M.arguments("fissa", "Raw") == ["Raw"]
