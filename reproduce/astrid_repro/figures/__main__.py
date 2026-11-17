"""Draw the data panels of one figure's dataset and save each as ``<out>/<panel>_<dataset>_<part>.<format>``
with its record (``.json``):

    python -m astrid_repro.figures cai1 --table <tables>/cv_cai1_equal_baseline.json --export <export>
        --ours <root> [...] --peers <folder> --cnmf-grid <grid> --suite2p <root> --out <folder>
    python -m astrid_repro.figures public --dataset ABO --table <tables>/cv_ABO.json --archive <archive>
        --ours <root> [...] --peers <root> [...] --cnmf-grid <grid> --suite2p <root>
        --evaluator tuncat_eval.evaluation:view --cnmf-fill tuncat_eval.evaluation:cnmf_fill --out <folder>

Run from ``reproduce/`` (so that ``astrid_repro`` and ``tuncat_eval`` import). ``cai1`` draws Figure 4 (A, B);
``public`` draws one row of the supplementary figure of D2, D4 and D5: D2 (``ABO``: A, B), D4 (``NAOMi``: C,
D) or D5 (``1p``: E, F). Paths are those given to ``tables_public`` (the tables are its outputs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cai1", help="Figure 4 (D3)")
    c.add_argument("--table", type=Path, required=True, help="cv_cai1_equal_baseline.json (tables_public cai1 "
                   "--baseline equal, the paper's Figure 4) or cv_cai1.json (as reported)")
    c.add_argument("--export", type=Path, required=True, help="the D3 export (index.json, <recording>.h5)")
    c.add_argument("--ours", type=Path, nargs="+", required=True, help="ASTRID's result roots")
    c.add_argument("--peers", type=Path, required=True, help="the peers' traces (<recording>.npz)")
    c.add_argument("--cnmf-grid", type=Path, help="CNMF's cross-validation grid, as given to tables_public cai1")
    c.add_argument("--suite2p", type=Path, required=True, help="Suite2p's traces")
    p = sub.add_parser("public", help="one row of the supplementary figure of D2, D4 and D5")
    p.add_argument("--dataset", choices=("ABO", "NAOMi", "1p"), required=True)
    p.add_argument("--table", type=Path, required=True, help="cv_<dataset>.json (tables_public public)")
    p.add_argument("--archive", type=Path, required=True, help="TUnCaT's figshare archive")
    p.add_argument("--ours", type=Path, nargs="+", required=True, help="ASTRID's result roots")
    p.add_argument("--peers", type=Path, nargs="+", help="the peers' reruns (default: the archive's released)")
    p.add_argument("--suite2p", type=Path, required=True, help="Suite2p's traces")
    p.add_argument("--evaluator", required=True, help="module:function, (dataset, video, archive) -> the "
                   "scoring ROI by ROI (tuncat_eval.evaluation:view)")
    p.add_argument("--cnmf-fill", help="module:function, (mat, masks) -> (n, T) (tuncat_eval.evaluation:cnmf_fill)")
    p.add_argument("--cnmf-grid", type=Path, help="CNMF's cross-validation grid, as given to tables_public grids")
    for q in (c, p):
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--formats", nargs="+", default=["png", "svg"])
    a = ap.parse_args(argv)
    matplotlib.use("Agg")
    from astrid_repro.figures import cai1, common, public

    if a.cmd == "cai1":
        made = cai1.panels(a.table, export=a.export, ours=a.ours, peers=a.peers, suite2p=a.suite2p,
                           cnmf_grid=a.cnmf_grid)
    else:
        made = public.panels(a.dataset, a.table, archive=a.archive, ours=a.ours, peers=a.peers, suite2p=a.suite2p,
                             evaluator=common.callable_from(a.evaluator),
                             cnmf_fill=common.callable_from(a.cnmf_fill), cnmf_grid=a.cnmf_grid)
    for stem, fig, record in made:
        for path in common.save(fig, a.out, stem, record, a.formats):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
