"""The machinery of D1's injection and leakage tests (``reproduce/astrid_repro/cases.py``) on a small synthetic
location: the rules that choose the cases, the patch of a case (the signal added to the data only), the fits,
the export that the peer methods read, and the scores against ``metrics.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

pd = pytest.importorskip("pandas")

import astrid  # noqa: E402
from astrid.api import Movie, load_pixels  # noqa: E402
from astrid.noise import noise_covariance  # noqa: E402
from astrid.patch import Spec, build, dilate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduce"))
from astrid_repro import cases as C  # noqa: E402
from astrid_repro import metrics, peers  # noqa: E402

SHAPE = (90, 100)
FPS = 30.9
REC = "rec1"


def disc(cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


ROIS = [disc(22, 22, 3), disc(22, 76, 3), disc(60, 34, 2.5), disc(62, 56, 2.2), disc(45, 88, 1), disc(3, 45, 2)]


def toy(t: int = 200, seed: int = 0) -> tuple[C.Location, np.ndarray, np.ndarray]:
    """Six ROIs: two bright sources far from astrocyte territory (ROI 2 the more active), two ROIs next to a
    blob of territory (ROI 3 with more of it around), one of 5 pixels and one on the registration border; two
    candidate pieces; two astrocyte ROIs (label 1 the larger)."""
    rng = np.random.default_rng(seed)
    neurons = np.zeros(SHAPE, dtype=np.int32)
    for i, m in enumerate(ROIS):
        neurons[m] = i + 1
    cand = (disc(40, 50, 2) | disc(75, 80, 2)).astype(np.int32)
    events = np.zeros(SHAPE, dtype=np.int32)
    events[disc(62, 42, 13)] = 5
    events[disc(62, 42, 15) & ~disc(62, 42, 13)] = 1
    events[disc(62, 30, 6)] = 6
    tau = np.full(SHAPE, 0.1)
    tau[disc(62, 42, 14)] = 0.9
    max_proj = np.full(SHAPE, 100.0, dtype=np.float32)
    for m in ROIS:
        max_proj[m] += 300
    max_proj += rng.normal(0, 5, SHAPE).astype(np.float32)
    astro = np.zeros(SHAPE, dtype=np.int32)
    astro[disc(70, 70, 5)] = 1
    astro[disc(30, 50, 2)] = 2
    loc = C.location("toy_loc01", neurons, cand, events, tau, max_proj, astro, FPS, 2.0, 5)
    dff = rng.gamma(1.0, 0.1, (len(ROIS), t))
    dff[1] *= 2
    astro_dff = 0.5 * (1 + np.sin(2 * np.pi * 0.05 * np.arange(t) / FPS))[None].repeat(2, axis=0)
    return loc, dff, astro_dff


def toy_movie(loc: C.Location, dff: np.ndarray, astro_dff: np.ndarray, seed: int = 1) -> np.ndarray:
    """(T, H, W) float32: a baseline of 100, the ROIs' activity, an astrocyte signal on the territory, noise."""
    rng = np.random.default_rng(seed)
    t = dff.shape[1]
    mov = 100 + rng.normal(0, 3, (t, *SHAPE))
    for v in range(1, len(ROIS) + 1):
        mov += (loc.neurons == v)[None] * (100 * dff[v - 1])[:, None, None]
    mov += loc.field.territory[None] * (30 * astro_dff[0])[:, None, None]
    return mov.astype(np.float32)


def pixels(loc: C.Location, cases: list[C.Case], mov: np.ndarray) -> astrid.patch.PixelData:
    flat = np.unique(np.concatenate([C.case_flat(loc, c) for c in cases]))
    return load_pixels([Movie(REC, data=mov)], FPS, flat, 0, astrid.Settings().baseline_rule(), high_pass=False)


# --------------------------------------------------------------------------------------------- the location
def test_location() -> None:
    loc, _, _ = toy()
    fld = loc.field
    assert np.array_equal(fld.territory, loc.events >= 2) and fld.px_scale == 1.0 and fld.tagrfp_channel
    assert fld.candidates.max() == 2 and set(np.unique(fld.candidates[disc(40, 50, 2)])) == {1}   # 4-connected
    assert fld.inner[5, 5] and not fld.inner[4, 50] and loc.max_proj.dtype == np.float32
    assert C.largest_astrocyte(loc.astro_label) == 1 and C.largest_astrocyte(np.zeros(SHAPE, int)) == 0
    tie = np.zeros(SHAPE, dtype=int)
    tie[:2, :2], tie[5:7, 5:7] = 2, 1
    assert C.largest_astrocyte(tie) == 1                                     # ties to the lower label
    args = (fld.neurons, fld.candidates, loc.events, fld.tau, loc.max_proj, loc.astro_label, FPS, 2.0, 5)
    with pytest.raises(ValueError, match="tau"):
        C.location("bad", *args[:3], fld.tau + 1, *args[4:])
    with pytest.raises(ValueError, match="counts"):
        C.location("bad", *args[:2], loc.events + 0.5, *args[3:])


# --------------------------------------------------------------------------------------------- the rules
def test_targets_and_sources() -> None:
    loc, dff, _ = toy()
    d = {REC: dff}
    assert C.leakage_targets(loc, d) == [3, 4]          # the most territory first; ROI 5 too small, 6 on the border
    assert C.leakage_targets(loc, d, 1) == [3]
    assert C.injection_sources(loc, d) == [2, 1]        # events near 3 and 4, 5 too small, 6 on the border
    assert all(C.usable_source(loc, v) for v in (1, 2))
    roi, foot = C.source_footprint(loc, 1)
    assert foot.dtype == np.float32 and foot.sum() == pytest.approx(1.0, abs=1e-6)
    assert not foot[~dilate(roi, 1)].any() and (foot[roi] > 0).all()
    crowded = loc.max_proj.copy()
    crowded[dilate(ROIS[0], 6)] = crowded[ROIS[0]].min()      # no brighter than its surroundings
    flat = C.Location(loc.field, loc.events, crowded, loc.astro_label)
    assert not C.usable_source(flat, 1) and C.injection_sources(flat, d) == [2]


def test_sites() -> None:
    loc, _, _ = toy()
    fld = loc.field
    occupied = (fld.neurons > 0) | (fld.candidates > 0)
    d_occ = ndi.distance_transform_edt(~occupied)
    for kind in (True, False):
        sites = C.free_sites(loc, kind)
        assert sites == C.free_sites(loc, kind) and sites != C.free_sites(loc, kind, seed=1)
        assert 0 < len(sites) <= C.SITES
        for i, (r, c) in enumerate(sites):
            assert 15 <= r < SHAPE[0] - 15 and 15 <= c < SHAPE[1] - 15 and d_occ[r, c] > 4
            assert all(np.hypot(r - q[0], c - q[1]) >= 15 for q in sites[:i])
            if kind:
                assert loc.events[r, c] >= 4 and fld.tau[r, c] >= 0.5
            else:
                assert not dilate(loc.events > 0, 6)[r, c] and fld.tau[r, c] < 0.2
    assert len(C.free_sites(loc, False, 2)) == 2


def test_injections_and_placement() -> None:
    loc, dff, astro_dff = toy()
    inj = C.injection(loc, {REC: dff}, 2, (40, 30), 3.0)
    assert np.allclose(np.argwhere(inj.roi).mean(axis=0), (40, 30), atol=0.5) and inj.roi.sum() == ROIS[1].sum()
    assert inj.footprint.sum() == pytest.approx(1.0, abs=1e-6) and not inj.footprint[~dilate(inj.roi, 1)].any()
    assert np.array_equal(inj.dff[REC], 3.0 * dff[1]) and inj.source == "M-002"
    a = C.astro_injection(loc, 1, {REC: astro_dff[0]}, 3, 1.0)
    roi = loc.neurons == 3
    assert (a.shape & roi).sum() >= 0.5 * roi.sum() and a.shape.sum() == (loc.astro_label == 1).sum()
    assert a.footprint.sum() == pytest.approx(1.0) and not a.footprint[~a.shape].any() and a.source == "A-01"
    # the first distance at which a direction covers half of the ROI (here 2.5 px: at 5 px none does), and
    # there the direction whose placed astrocyte lies most in territory
    t0, a0 = np.argwhere(roi).mean(axis=0), np.argwhere(loc.astro_label == 1).mean(axis=0)
    for off in C.PLACEMENT_OFFSETS_PX:
        placed = []
        for ang in np.arange(8 if off > 0 else 1) * np.pi / 4:
            c = t0 + off * np.array([np.sin(ang), np.cos(ang)])
            s = C.moved(loc.astro_label == 1, (int(round(c[0] - a0[0])), int(round(c[1] - a0[1]))))
            if (s & roi).sum() >= 0.5 * roi.sum():
                placed.append(float(loc.field.territory[s].mean()))
        if placed:
            break
    assert off == 2.5 and float(loc.field.territory[a.shape].mean()) == max(placed)
    offset = np.argwhere(a.shape).mean(axis=0) - t0
    assert np.hypot(*offset) == pytest.approx(off, abs=0.75) and np.hypot(*a.shift) > 0
    cases, ch = C.injection_cases(loc, {REC: dff})
    n_sites = len(ch["sites"])
    assert [c.name for c in cases[:2]] == ["inj00-astro-x1", "inj00-astro-x3"] and len(cases) == 2 * n_sites
    assert [c.inject.source for c in cases[::2]] == [f"M-{[2, 1][j % 2]:03d}" for j in range(n_sites)]
    leak, ch = C.leakage_cases(loc, {REC: dff}, {REC: astro_dff[0]}, 1)
    assert [c.name for c in leak[:3]] == ["M-003-astro-x0", "M-003-astro-x1", "M-003-astro-x3"]
    assert ch["targets"] == [3, 4] and ch["astro_source"] == "A-01"
    for c in cases + leak:                                  # the bases of the export, as the peers read them
        assert c.base == peers.base_of(c.name)
    assert not leak[0].astro.dff[REC].any() and np.array_equal(leak[2].astro.dff[REC], 3.0 * astro_dff[0])


# --------------------------------------------------------------------------------------------- the patch
def test_case_patch_adds_the_signal_to_the_data_only() -> None:
    loc, dff, astro_dff = toy()
    mov = toy_movie(loc, dff, astro_dff)
    leak, _ = C.leakage_cases(loc, {REC: dff}, {REC: astro_dff[0]}, 1)
    inj, _ = C.injection_cases(loc, {REC: dff})
    case, case0, injected = leak[1], leak[0], inj[1]
    data = pixels(loc, [case, injected], mov)
    plain = build(loc.field, data, Spec(case.name, 3))
    cp = C.build_case(loc, data, case)
    b, p = cp.built, plain.patch
    assert b.names == [*plain.names[:-1], "A-inj", "neuropil"] and b.patch.components[-2].kind == "astro"
    shape = case.astro.shape.ravel(order="F")[b.pixels]
    support = dilate(case.astro.shape, 1).ravel(order="F")[b.pixels]
    assert np.array_equal(b.patch.components[-2].support, support)
    assert np.array_equal(b.patch.m[0], p.m[0]) and np.array_equal(b.patch.w, p.w)    # F0, weights: before
    assert np.array_equal(b.pixels, plain.pixels) and np.array_equal(b.ring, plain.ring)
    tau = b.patch.tau
    assert (tau[shape] >= 0.8).all() and np.array_equal(tau[~shape], p.tau[~shape])
    assert np.array_equal(tau[shape], np.maximum(p.tau[shape], 0.8))
    added = b.patch.y[0] - p.y[0]
    f0 = p.m[0][shape].mean()
    assert np.allclose(added[shape].mean(axis=0) / f0, astro_dff[0], rtol=1e-9, atol=1e-12)   # u, on the mask
    h = case.astro.footprint.ravel(order="F")[b.pixels]
    assert np.allclose(cp.astro_roi[0], added[b.roi].mean(axis=0), rtol=1e-9) and cp.truth is None
    assert np.allclose(cp.astro_roi[0], h[b.roi].mean() * astro_dff[0] * f0 / h[shape].mean(), rtol=1e-12)
    assert np.array_equal(b.sigma, noise_covariance([b.patch.y[0] - b.patch.m[0]], FPS)[0])   # after
    zero = C.build_case(loc, data, case0).built                           # x0: A-inj and the evidence, no signal
    assert zero.names[1:] == b.names[1:] and np.array_equal(zero.patch.y[0], p.y[0])
    assert np.array_equal(zero.patch.tau, tau)
    ablated = C.build_case(loc, data, case, astro_components=False)
    assert "astro" not in [c.kind for c in ablated.built.patch.components]
    assert np.array_equal(ablated.built.patch.y[0], b.patch.y[0]) and np.array_equal(ablated.built.patch.tau, tau)
    ci = C.build_case(loc, data, injected)                                 # an injected ROI: no label left out
    bi = ci.built
    assert bi.names[0] == injected.name and bi.patch.components[0].target
    rows = data.rows(bi.pixels)
    added = bi.patch.y[0] - data.y[REC][rows].astype(np.float64)
    roi = bi.roi
    assert np.array_equal(bi.patch.m[0], data.m[REC][rows].astype(np.float64))
    # a source's footprint is float32 and its ROI mean is taken in float32, so these hold to float32 precision
    assert np.allclose(added[roi].mean(axis=0) / bi.patch.m[0][roi].mean(), 3.0 * dff[1], rtol=1e-6, atol=1e-12)
    assert np.allclose(ci.truth[0], added[roi].mean(axis=0), rtol=1e-6) and ci.astro_roi is None


# --------------------------------------------------------------------------------------------- the fits
def test_fits_in_workers_are_the_fits_in_process(tmp_path: Path) -> None:
    loc, dff, astro_dff = toy()
    mov = toy_movie(loc, dff, astro_dff)
    inj, _ = C.injection_cases(loc, {REC: dff})
    leak, _ = C.leakage_cases(loc, {REC: dff}, {REC: astro_dff[0]}, 1)
    cases = [inj[0], leak[1]]
    data = pixels(loc, cases, mov)
    params = astrid.Settings(tau_r=0.05, tau_d=0.4, max_iterations=40).params(FPS)
    rows = C.fit_cases(loc, data, cases, params, tmp_path / "a", log=lambda s: None)
    C.fit_cases(loc, data, cases, params, tmp_path / "b", workers=2, log=lambda s: None)
    assert [r["case"] for r in rows] == [c.name for c in cases] and rows[0]["iterations"] <= 40
    for c in cases:
        with np.load(tmp_path / "a" / f"{c.name}.npz") as za, np.load(tmp_path / "b" / f"{c.name}.npz") as zb:
            for k in za.files:
                if k != "seconds":
                    assert np.array_equal(za[k], zb[k]), (c.name, k)
            assert za["delivered"].shape == (1, dff.shape[1]) and za["names"][-1] == "neuropil"
    with np.load(tmp_path / "a" / f"{inj[0].name}.npz") as z:
        assert z["truth"].shape == (1, dff.shape[1]) and z["astro_roi"].size == 0
    with np.load(tmp_path / "a" / f"{leak[1].name}.npz") as z:
        assert z["astro_roi"].shape == (1, dff.shape[1]) and z["truth"].size == 0 and "A-inj" in z["names"]


# --------------------------------------------------------------------------------------------- the export
def test_export_is_the_peers_case_format(tmp_path: Path) -> None:
    loc, dff, astro_dff = toy()
    mov = toy_movie(loc, dff, astro_dff)
    inj, _ = C.injection_cases(loc, {REC: dff})
    leak, _ = C.leakage_cases(loc, {REC: dff}, {REC: astro_dff[0]}, 1)
    cases = inj[:2] + leak[:3]
    data = pixels(loc, cases, mov)
    params = astrid.Settings(tau_r=0.05, tau_d=0.4, max_iterations=20).params(FPS)
    C.fit_cases(loc, data, inj[:2], params, tmp_path / "injection", log=lambda s: None)
    C.fit_cases(loc, data, leak[:3], params, tmp_path / "leakage", log=lambda s: None)
    out = tmp_path / "compare"
    crops = C.read_crops(mov, C.export_boxes(loc, cases), block=64)
    index = C.export_cases(loc, crops, cases, {"injection": tmp_path / "injection", "leakage": tmp_path / "leakage"},
                           out)
    C.write_index(out, index)
    assert [r["spec"] for r in index] == [c.name for c in cases] and peers.pending(out)[1] == [c.name for c in cases]
    assert index[0] == {"spec": "inj00-astro-x1", "base": "inj00-astro", "test": "injection", "scale": 1.0,
                        "site": "astro", "neighbours": index[0]["neighbours"], "atoms": index[0]["atoms"]}
    assert index[2]["test"] == "leakage" and index[2]["site"] == "" and index[2]["scale"] == 0.0
    assert json.loads((out / "index.json").read_text()) == index
    for c in (cases[1], cases[3]):
        movie, masks = peers.case_movie(out, c.name)
        r0, r1, c0, c1 = masks["box"]
        assert (r1 - r0, c1 - c0) == movie.shape[1:] and movie.dtype == np.float32
        roi = c.roi(loc)
        assert np.array_equal(masks["target"], roi[r0:r1, c0:c1])
        assert np.array_equal(crops[c.base], mov[:, r0:r1, c0:c1])
        z = np.load(out / f"{c.name}.npz")
        foot = (c.inject or c.astro).footprint[r0:r1, c0:c1].astype(np.float32)
        expect = (mov[:, r0:r1, c0:c1] + foot[None] * z["amplitude"][:, None, None].astype(np.float32))
        assert np.array_equal(movie, expect.astype(np.float32))
        h = (c.inject or c.astro).footprint
        ref = z["truth"] if c.inject is not None else z["astro_roi"]
        assert np.allclose(z["amplitude"] * h[roi].mean(), ref, rtol=1e-6)
        box = dilate(roi, 15) & loc.inner                                 # 30 um around the ROI, inside the border
        assert (r0, c0) == tuple(np.argwhere(box).min(axis=0)) and (r1, c1) == tuple(np.argwhere(box).max(axis=0) + 1)
    with np.load(out / "M-003_masks.npz") as m:                           # the added astrocyte's support is an atom
        r0, r1, c0, c1 = m["box"]
        region = astrid.patch.patch_region(loc.field, loc.neurons == 3)
        assert any(np.array_equal(a, (dilate(leak[0].astro.shape, 1) & region)[r0:r1, c0:c1]) for a in m["atoms"])
        assert len(m["neighbours"]) == index[2]["neighbours"] and len(m["atoms"]) == index[2]["atoms"]


# --------------------------------------------------------------------------------------------- the scores
def test_scores_are_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    t = 400
    truth = np.convolve(rng.random(t) < 0.03, np.exp(-np.arange(60) / 10.0))[:t] * 50 + 5
    astro = 20 * np.sin(2 * np.pi * np.arange(t) / 300) + 3
    cases, fits, res = tmp_path / "cases", tmp_path / "fits", tmp_path / "results"
    for d in (cases, fits, res):
        d.mkdir()
    index = [{"spec": "inj00-astro-x1", "base": "inj00-astro", "test": "injection", "scale": 1.0, "site": "astro",
              "neighbours": 0, "atoms": 0}]
    index += [{"spec": f"M-001-astro-x{s:g}", "base": "M-001", "test": "leakage", "scale": s, "site": "",
               "neighbours": 0, "atoms": 0} for s in (0.0, 1.0)]
    np.savez(cases / "inj00-astro-x1.npz", truth=truth)
    np.savez(cases / "M-001-astro-x1.npz", astro_roi=astro)
    traces = {}
    for name, extra in (("inj00-astro-x1", truth), ("M-001-astro-x0", 0.0), ("M-001-astro-x1", 0.4 * astro)):
        y = extra + rng.normal(0, 5, t)
        traces[name] = {"demix": y, "ring": y + 2 * rng.normal(0, 5, t),
                        "fissa": y + rng.normal(0, 5, t), "roi": (y + 30).astype(np.float32)}
        np.savez(fits / f"{name}.npz", delivered=traces[name]["demix"][None], ring_trace=traces[name]["ring"][None])
        np.savez(res / f"{name}.npz", fissa=traces[name]["fissa"], tuncat_raw_roi=traces[name]["roi"])
    inj = C.injection_scores(cases, fits, index, FPS, [res])
    assert list(inj.columns) == ["case", "site", "scale", "method", "corr", "gain", "err_raw", "err_2hz", "err_05hz"]
    assert list(inj.method) == ["demix", "ring", "fissa", "roi"]
    for r in inj.itertuples():
        x = traces["inj00-astro-x1"][r.method]
        assert r.err_05hz == metrics.error(x, truth, FPS) and r.err_2hz == metrics.error(x, truth, FPS, 2.0)
        if r.method != "roi":                                              # float64 traces: metrics' values
            assert r.err_raw == metrics.error(x, truth, FPS, None) and r.gain == metrics.gain(x, truth)
        else:                                                              # float32: its mean removed in float32
            ac, q = x - x.mean(), truth - truth.mean()
            assert r.err_raw == float(np.linalg.norm(ac - q) / np.linalg.norm(q))
            assert r.err_raw == pytest.approx(metrics.error(x, truth, FPS, None), rel=1e-5)
    leak = C.leakage_scores(cases, fits, index, FPS, [res])
    assert list(leak.columns) == ["target", "scale", "method", "leak", "leak_05hz", "kept_corr"]
    for r in leak.itertuples():
        y1, y0 = traces["M-001-astro-x1"][r.method], traces["M-001-astro-x0"][r.method]
        assert r.leak_05hz == metrics.leakage(y1, y0, astro, FPS, 0.5)
        assert r.kept_corr == float(np.corrcoef(y1, y0)[0, 1]) and r.target == "M-001" and r.scale == 1.0
        if r.method != "roi":
            assert r.leak == metrics.leakage(y1, y0, astro, FPS)
        else:
            assert r.leak == pytest.approx(metrics.leakage(y1, y0, astro, FPS), rel=1e-5)
    text = C.summary(inj, leak)
    assert "error below 0.5 Hz" in text and "leakage" in text and C.summary(inj[:0], leak[:0]) == "no scored case"
