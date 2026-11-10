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


def test_cnmf_ar_order() -> None:
    s = "    p = 1\n    params_movie = {'p': p,\n        p = params_movie['p']\n"
    assert S.cnmf_p(s, 2) == ("    p = 2  # peers.sbatch: CNMF_P\n    params_movie = {'p': p,\n"
                              "        p = params_movie['p']\n")
    with pytest.raises(ValueError, match="found 0 times"):
        S.cnmf_p("        p = params_movie['p']\n", 2)
    with pytest.raises(ValueError, match="found 2 times"):
        S.cnmf_p("    p = 1\n    p = 1\n", 2)


def _archive(tmp_path: Path) -> tuple[Path, Path]:
    scripts, data = tmp_path / "repro", tmp_path / "archive"
    for sub, name, text in (("TUnCaT archive", "TUnCaT_multi_ABO.py", TUNCAT),
                            ("Other methods/FISSA", "FISSA_ABO.py", FISSA),
                            ("Other methods/CNMF_CaImAn", "CNMF_unmixing_seeded_ABO.py", "    p = 1\n" + CNMF),
                            ("Other methods/CNMF_CaImAn", "h5ToMmap.py", "datadir = '..\\\\..\\\\data\\\\*\\\\'\n"),
                            ("template", "t.mat", "")):
        (scripts / sub).mkdir(parents=True, exist_ok=True)
        (scripts / sub / name).write_text(text, encoding="utf-8")
    (data / "data" / "ABO" / "GT Masks").mkdir(parents=True)
    (data / "data" / "ABO" / "GT Masks" / "FinalMasks_1.mat").write_bytes(b"m")
    for v in ("1", "2"):
        (data / "data" / "ABO" / f"{v}.h5").write_bytes(b"movie " + v.encode())
    return scripts, data


def test_stage_cnmf_grid_settings(tmp_path: Path) -> None:
    """The cross-validation grid: at (1, 1) the script is staged as the published runs staged it (no CNMF_P
    line); at (3, 2) only its gnb and p lines differ. A smoke run of CNMF stages only its own movie."""
    scripts, data = _archive(tmp_path)
    one = M.stage(scripts, data, "ABO", "cnmf", tmp_path / "s11", posix=True).read_text(encoding="utf-8")
    grid = M.stage(scripts, data, "ABO", "cnmf", tmp_path / "s32", posix=True, cnmf_gnb=3,
                   cnmf_p=2).read_text(encoding="utf-8")
    assert "CNMF_P" not in one and "    p = 1\n" in one
    changed = [(a, b) for a, b in zip(one.splitlines(), grid.splitlines(), strict=True) if a != b]
    assert changed == [("    p = 1", "    p = 2  # peers.sbatch: CNMF_P"),
                       ("cnm = cnmf.CNMF(Ain=A_in.astype(bool), 'gnb': 1,  # peers.sbatch: CNMF_GNB",
                        "cnm = cnmf.CNMF(Ain=A_in.astype(bool), 'gnb': 3,  # peers.sbatch: CNMF_GNB")]
    M.stage(scripts, data, "ABO", "cnmf", tmp_path / "smoke", posix=True, smoke="2")
    staged = sorted(p.name for p in (tmp_path / "smoke" / "data" / "ABO").iterdir())
    assert staged == ["2.h5", "GT Masks"]
    M.stage(scripts, data, "ABO", "fissa", tmp_path / "fissa", posix=True, smoke="2")
    assert sorted(p.name for p in (tmp_path / "fissa" / "data" / "ABO").iterdir()) == ["1.h5", "2.h5", "GT Masks"]


def test_run_cnmf_memmaps_in_a_temporary_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CNMF run points CaImAn's CAIMAN_TEMP into its staging folder and deletes it when the run ends (also with
    --keep-stage); the outputs are moved to <out>/<dataset>; the CLI passes --cnmf-gnb and --cnmf-p."""
    scripts, data = _archive(tmp_path)
    calls: list[tuple[str, str | None]] = []

    def fake_run(cmd: list[str], cwd: Path, check: bool, env: dict[str, str] | None = None) -> None:
        temp = None if env is None else env.get("CAIMAN_TEMP")
        calls.append((cmd[1], temp))
        assert temp is not None and Path(temp).is_dir()
        (Path(temp) / "x_memmap_.mmap").write_bytes(b"m")
        if cmd[1] != "h5ToMmap.py":
            out = Path(cwd).parents[1] / "data" / "ABO" / "traces_CNMF_Raw_p2"
            out.mkdir(parents=True)
            (out / "1.mat").write_bytes(b"t")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    moved = M.run(scripts, data, "ABO", "cnmf", tmp_path / "out", cnmf_gnb=3, cnmf_p=2, posix=True,
                  keep_stage=True)
    assert [c[0] for c in calls] == ["h5ToMmap.py", "CNMF_unmixing_seeded_ABO.py"]
    temp = Path(str(calls[0][1]))
    assert temp == tmp_path / "out" / "_stage_cnmf_ABO_Raw" / "caiman_temp" and not temp.exists()
    assert moved == [tmp_path / "out" / "ABO" / "traces_CNMF_Raw_p2"]
    seen: dict[str, object] = {}
    monkeypatch.setattr(M, "run", lambda *a, **k: seen.update(k) or [])
    M.main(["--scripts", str(scripts), "--data", str(data), "--dataset", "ABO", "--method", "cnmf", "--out",
            str(tmp_path / "o"), "--cnmf-gnb", "2", "--cnmf-p", "2"])
    assert (seen["cnmf_gnb"], seen["cnmf_p"]) == (2, 2)


def test_one_video_per_run_into_one_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--smoke runs (one video each, so each memmap is deleted right after its fit) into one --out: their traces
    join one folder, each run's timer named after its video; a second run of a video is refused."""
    scripts, data = _archive(tmp_path)

    def fake_run(cmd: list[str], cwd: Path, check: bool, env: dict[str, str] | None = None) -> None:
        if cmd[1] != "h5ToMmap.py":
            staged = sorted(p.stem for p in (Path(cwd).parents[1] / "data" / "ABO").glob("*.h5"))
            out = Path(cwd).parents[1] / "data" / "ABO" / "traces_CNMF_Raw_p1"
            out.mkdir(parents=True)
            for v in staged:                                  # only the run's own movie is staged
                (out / f"{v}.mat").write_bytes(v.encode())
            (out / "Table_time.mat").write_bytes(b"time of " + staged[0].encode())

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    for v in ("1", "2"):
        assert M.run(scripts, data, "ABO", "cnmf", tmp_path / "grid", smoke=v, posix=True) == \
            [tmp_path / "grid" / "ABO" / "traces_CNMF_Raw_p1"]
    folder = tmp_path / "grid" / "ABO" / "traces_CNMF_Raw_p1"
    assert sorted(p.name for p in folder.iterdir()) == ["1.mat", "2.mat", "Table_time_1.mat", "Table_time_2.mat"]
    assert (folder / "Table_time_2.mat").read_bytes() == b"time of 2"
    with pytest.raises(FileExistsError, match="overwrite"):
        M.run(scripts, data, "ABO", "cnmf", tmp_path / "grid", smoke="2", posix=True)
