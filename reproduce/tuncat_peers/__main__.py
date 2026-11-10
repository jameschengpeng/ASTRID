"""Run one peer method (FISSA, TUnCaT or CNMF) on one of D2, D4 and D5 with the scripts of TUnCaT's
paper-reproduction archive, as the paper did (GPL-2: see ``__init__.py``).

A private staging folder gets a copy of the archive's ``TUnCaT archive``, ``Other methods`` and ``template``
folders and a ``data/<dataset>`` folder with the dataset's files (masks and labels copied, movies linked or
copied), because the scripts write their outputs next to the data. The method's script is changed as
``staging`` describes, run with the arguments of the authors' ``.bat`` files in the current Python environment
(which must hold the method's packages), and its outputs, ``traces_<method>_<video>...``, are moved to
``<out>/<dataset>/``, the layout ``astrid_repro.tables_public grids --peers <out>/<dataset>`` reads. With
``--smoke <video>`` (one video per run), they join the other runs' files there, the run's ``Table_time.mat``
(its timer, of the one video) named ``Table_time_<video>.mat``; a file is never overwritten.

The paper's runs, each in its own output folder:
- TUnCaT, FISSA and CNMF on ``ABO``, ``NAOMi`` and ``1p`` (CNMF at ``--cnmf-gnb 1 --cnmf-p 1``, the defaults);
- FISSA on ``NAOMi`` with ``--fissa-alphas 0.0005`` (the extension of its grid), into a second folder that
  ``tables_public grids`` reads as a second ``--peers`` root;
- CNMF's cross-validation grid: ``--cnmf-gnb`` 1, 2, 3 × ``--cnmf-p`` 1, 2, each setting into its own folder
  ``<grid>/gnb<g>_p<p>``, which ``tables_public grids --cnmf-grid <grid>`` reads.

CaImAn writes its memory-mapped movies to ``$CAIMAN_TEMP``, else to ``~/caiman_data/temp``, whatever folder the
script names. A CNMF run points ``CAIMAN_TEMP`` to a temporary folder inside the staging folder and deletes it
when the run ends. The authors' script fits every video of the dataset in one run; with ``--smoke <video>`` a
run fits one video (and only that video's movie is staged), so its memmap is deleted right after its fit.

    python -m tuncat_peers --scripts <TUnCaT_paper_reproduction> --data <archive> --dataset ABO --method cnmf
        --out <peers> [--video Raw] [--smoke <video>] [--fissa-alphas 0.0005] [--cnmf-gnb 1] [--cnmf-p 1]
        [--keep-stage]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tuncat_peers import staging

CODE_FOLDERS = ("TUnCaT archive", "Other methods", "template")
SCRIPTS = {"tuncat": ("TUnCaT archive", "TUnCaT_multi_{}.py"), "fissa": ("Other methods/FISSA", "FISSA_{}.py"),
           "cnmf": ("Other methods/CNMF_CaImAn", "CNMF_unmixing_seeded_{}.py")}


def arguments(method: str, video: str) -> list[str]:
    """The arguments of the authors' .bat files (run_TUnCaT_all.bat: video type, nbin 1, th_residual 0,
    th_pertmin 1, flexible_alpha 1)."""
    return [video, "1", "0", "1", "1"] if method == "tuncat" else [video]


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _rewrite(path: Path, *changes: object) -> None:
    """Apply the changes to a script, read and written as text in UTF-8, as the paper's staging did."""
    s = open(path, encoding="utf-8").read()
    for change in changes:
        s = change(s)  # type: ignore[operator]
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def stage(scripts: str | Path, data: str | Path, dataset: str, method: str, folder: str | Path, *,
          posix: bool, smoke: str | None = None, fissa_alphas: list[float] | None = None,
          cnmf_gnb: int | None = 1, cnmf_p: int = 1) -> Path:
    """The staging folder (module docstring); returns the method's changed script."""
    scripts, data, folder = Path(scripts), Path(data), Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    for name in CODE_FOLDERS:
        shutil.copytree(scripts / name, folder / name)
    target = folder / "data" / dataset
    target.mkdir(parents=True)
    for f in sorted((data / "data" / dataset).iterdir()):
        if f.name.startswith("."):
            continue
        if f.suffix == ".h5":
            if method == "cnmf" and smoke and f.stem != smoke:
                continue                       # h5ToMmap.py memmaps every movie it finds: only the one fitted
            _link_or_copy(f, target / f.name)
        elif f.is_dir():
            shutil.copytree(f, target / f.name)
        else:
            shutil.copy2(f, target / f.name)
    sub, pattern = SCRIPTS[method]
    script = folder / sub / pattern.format(dataset)
    first = [staging.to_posix] if posix else []
    if smoke:
        first.append(lambda s: staging.filter_list(s, smoke))
    if method == "tuncat":
        _rewrite(script, *first, staging.patch_tuncat)
    elif method == "fissa":
        alphas = [] if fissa_alphas is None else [lambda s: staging.fissa_alphas(s, fissa_alphas)]
        _rewrite(script, *first, staging.patch_fissa, *alphas)
    else:
        if posix:
            _rewrite(script.parent / "h5ToMmap.py", staging.to_posix)
        gnb = [] if cnmf_gnb is None else [lambda s: staging.cnmf_gnb(s, cnmf_gnb)]
        p = [] if cnmf_p == 1 else [lambda s: staging.cnmf_p(s, cnmf_p)]
        _rewrite(script, *first, staging.patch_cnmf, *gnb, *p)
    return script


def run(scripts: str | Path, data: str | Path, dataset: str, method: str, out: str | Path, *, video: str = "Raw",
        smoke: str | None = None, fissa_alphas: list[float] | None = None, cnmf_gnb: int | None = 1,
        cnmf_p: int = 1, posix: bool | None = None, keep_stage: bool = False) -> list[Path]:
    """Stage, run and collect one (method, dataset); returns the output folders moved to ``<out>/<dataset>``."""
    out = Path(out)
    folder = out / f"_stage_{method}_{dataset}_{video}"
    script = stage(scripts, data, dataset, method, folder, posix=os.name != "nt" if posix is None else posix,
                   smoke=smoke, fissa_alphas=fissa_alphas, cnmf_gnb=cnmf_gnb, cnmf_p=cnmf_p)
    env = None
    if method == "cnmf":
        memmaps = folder / "caiman_temp"               # CaImAn's memmaps (module docstring), deleted below
        memmaps.mkdir()
        env = {**os.environ, "CAIMAN_TEMP": str(memmaps)}
        try:
            subprocess.run([sys.executable, "h5ToMmap.py"], cwd=script.parent, check=True, env=env)
            subprocess.run([sys.executable, script.name, *arguments(method, video)], cwd=script.parent,
                           check=True, env=env)
        finally:
            shutil.rmtree(memmaps, ignore_errors=True)
    else:
        subprocess.run([sys.executable, script.name, *arguments(method, video)], cwd=script.parent, check=True)
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for d in sorted((folder / "data" / dataset).glob("traces_*")):
        target = dest / d.name
        if smoke:                     # one video per run: its files join the other runs' in one folder
            target.mkdir(exist_ok=True)
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                rel = f.relative_to(d)
                if rel.name == "Table_time.mat":               # the run's timer, of this one video
                    rel = rel.with_name(f"Table_time_{smoke}.mat")
                if (target / rel).exists():
                    raise FileExistsError(f"{target / rel} exists: a run would overwrite another's output")
                (target / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target / rel))
        else:
            shutil.move(str(d), str(target))
        moved.append(target)
    if not keep_stage:
        shutil.rmtree(folder)
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scripts", type=Path, required=True,
                    help="TUnCaT's paper-reproduction archive (github.com/YijunBao/TUnCaT_paper_reproduction)")
    ap.add_argument("--data", type=Path, required=True, help="the archive root holding data/<dataset>")
    ap.add_argument("--dataset", choices=("ABO", "NAOMi", "1p"), required=True)
    ap.add_argument("--method", choices=tuple(SCRIPTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", default="Raw", help="the video type of the scripts: Raw (the paper's) or SNR")
    ap.add_argument("--smoke", help="run one video only")
    ap.add_argument("--fissa-alphas", help="comma-separated alphas instead of the script's list")
    ap.add_argument("--cnmf-gnb", type=int, default=1, help="CNMF's background components (the paper's 1; "
                    "0 keeps the script's 2)")
    ap.add_argument("--cnmf-p", type=int, default=1, help="CNMF's AR order (the paper's 1)")
    ap.add_argument("--posix", choices=("auto", "yes", "no"), default="auto",
                    help="rewrite the scripts' backslash paths (auto: not on Windows)")
    ap.add_argument("--keep-stage", action="store_true")
    a = ap.parse_args(argv)
    alphas = [float(x) for x in a.fissa_alphas.split(",")] if a.fissa_alphas else None
    posix = None if a.posix == "auto" else a.posix == "yes"
    for d in run(a.scripts, a.data, a.dataset, a.method, a.out, video=a.video, smoke=a.smoke, fissa_alphas=alphas,
                 cnmf_gnb=a.cnmf_gnb or None, cnmf_p=a.cnmf_p, posix=posix, keep_stage=a.keep_stage):
        print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
