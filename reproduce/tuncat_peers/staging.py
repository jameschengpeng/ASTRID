"""The changes made to the scripts of TUnCaT's paper-reproduction archive before they ran for the paper (GPL-2:
see this folder's ``__init__.py`` and ``LICENSE``). Each one is a function of a script's text; none touches the
numerics:

- ``to_posix``: the scripts join paths with backslashes; on Linux their relative path literals (those starting
  with ``..``) become slashes. Its three other replacements target single backslashes, while the archive's literals hold
  doubled ones, so they change nothing; they are kept so that the changed scripts are those of the paper's runs
  (on Linux the CNMF script then finds no memory-mapped movie and makes each one itself, by the same call on
  the same movie as ``h5ToMmap.py``);
- ``filter_list``: a smoke run keeps one video in the script's ``list_Exp_ID``;
- ``patch_tuncat``: Python 3.10's ``shared_memory`` refuses to close while numpy views exist, which breaks the
  archive's shared-memory trace extraction (written for Python 3.8): its numba branch, which the authors use for
  smaller videos and which computes the same traces, is taken, and the four end-of-run shared-memory cleanup
  calls are made best-effort;
- ``patch_fissa``: FISSA 1.0.0 (the authors ran 0.7.2) refuses a cache folder made at another alpha: each alpha
  gets its own cache folder, its preparation redone from the same data; the separation is unchanged;
- ``fissa_alphas``: only these alphas (the paper's extension of FISSA's grid on D4, alpha 0.0005), the output
  and cache folders named at full precision (``{:g}``; the script's ``{:6.3f}`` would name 0.0005 like 0.001);
- ``patch_cnmf``: the CNMF script is from CaImAn's 2019 demos. CaImAn 1.13.2 keeps every call it makes, but two
  of its unused imports no longer exist, numpy 2 has no ``np.int`` and ``np.bool``, and ``fit()`` now works in
  place (returns None): five lines rewritten;
- ``cnmf_gnb``: that many background components instead of the script's 2. The CNMF traces the authors
  released hold one, and the paper's runs used one.
"""

from __future__ import annotations

import re


def to_posix(s: str) -> str:
    # path literals: any single-quoted string holding '..\' or '\*' or 'SNR video' joins -> slashes
    s = re.sub(r"'(\.\.(?:\\[^']*)+)'", lambda m: "'" + m.group(1).replace("\\\\", "/") + "'", s)
    s = s.replace("'\\*_memmap__*.mmap'", "'/*_memmap__*.mmap'").replace(".rsplit('\\', 1)", ".rsplit('/', 1)")
    return s.replace("sys.path.insert(0, 'C:\\Other methods\\CaImAn')", "pass  # caiman from the environment")


def filter_list(s: str, keep: str) -> str:
    s2 = re.sub(r"list_Exp_ID\s*=\s*\[[^\]]*\]", f"list_Exp_ID = ['{keep}']", s, count=1, flags=re.S)
    s2 = re.sub(r"list_Exp_ID\s*=\s*\['Video_'\+str\(x\) for x in list\(range\(0,10\)\)\]", f"list_Exp_ID = ['{keep}']",
                s2)
    return re.sub(r"list_Exp_ID\s*=\s*\['Video_'\+str\(n\) for n in range\(10\)\]", f"list_Exp_ID = ['{keep}']", s2)


def patch_tuncat(s: str) -> str:
    s2 = s.replace("if FinalMasks.sum()*T >= 7e7:", "if False:  # numba branch forced (peers.sbatch)")
    if s2 == s:
        raise ValueError("the shared-memory branch line was not found")
    n = 0
    nl = chr(10)
    for call in ("shm_video.close()", "shm_video.unlink()", "shm_masks.close()", "shm_masks.unlink()"):
        pat = r"^(\s*)" + re.escape(call) + r"\s*$"

        def rep(m: re.Match[str], c: str = call) -> str:
            return m.group(1) + "try: " + c + nl + m.group(1) + "except Exception: pass  # peers.sbatch"

        s3 = re.sub(pat, rep, s2, count=1, flags=re.M)
        n += int(s3 != s2)
        s2 = s3
    if n != 4:
        raise ValueError(f"patched {n} of 4 shared-memory cleanup calls")
    return s2


def patch_fissa(s: str) -> str:
    old = ("experiment = fissa.Experiment([video], [FinalMasks.tolist()], folder_FISSA, alpha=alpha, "
           "ncores_preparation=1)")
    nl = chr(10)
    new = ("folder_alpha = folder_FISSA + '_alpha={:6.3f}'.format(alpha).replace(' ', '')  # peers.sbatch" + nl
           + "            experiment = fissa.Experiment([video], [FinalMasks.tolist()], folder_alpha, alpha=alpha, "
           + "ncores_preparation=1)")
    if s.count(old) != 1:
        raise ValueError("FISSA per-alpha Experiment line not found")
    s = s.replace(old, new)
    return s.replace("experiment.separate(redo_prep=False, redo_sep=True)",
                     "experiment.separate(redo_prep=True, redo_sep=True)")


def fissa_alphas(s: str, vals: list[float]) -> str:
    s, n = re.subn(r"^(\s*)list_alpha\s*=\s*\[[^\]]*\]",
                   lambda m: m.group(1) + "list_alpha = " + repr(vals) + "  # peers.sbatch: FISSA_ALPHAS", s, count=1,
                   flags=re.M)
    if n != 1:
        raise ValueError("the script's list_alpha line not found")
    for old, new in (('"alpha={:6.3f}".format(alpha)', '"alpha={:g}".format(alpha)'),
                     ("'_alpha={:6.3f}'.format(alpha)", "'_alpha={:g}'.format(alpha)")):
        if s.count(old) != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r}")
        s = s.replace(old, new)
    return s


def patch_cnmf(s: str) -> str:
    for old, new in (("from past.utils import old_div", "pass  # peers.sbatch: unused Python-2 shim dropped"),
                     ("from skimage.external.tifffile import TiffFile",
                      "from tifffile import TiffFile  # peers.sbatch"),
                     ("radius = np.int(np.median(", "radius = int(np.median("),
                     ("Ain=A_in.astype(np.bool),", "Ain=A_in.astype(bool),"),
                     ("cnm = cnm.fit(images)", "cnm.fit(images)  # peers.sbatch: fit() returns None since CaImAn 1.8")):
        if s.count(old) != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r}")
        s = s.replace(old, new)
    return s


def cnmf_gnb(s: str, n: int) -> str:
    if s.count("'gnb': 2,") != 1:
        raise ValueError("the script's gnb line not found")
    return s.replace("'gnb': 2,", f"'gnb': {n},  # peers.sbatch: CNMF_GNB")
