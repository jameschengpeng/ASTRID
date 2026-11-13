"""The compiled drive update and the two speed-ups around the fit keep its arithmetic bitwise:
- the compiled FISTA loop (``kernels._fista``) against the reference loop (``kernels._deconvolve_reference``);
- a patch's noise covariance from the field's high-passed pixels (``api.load_pixels(high_pass=True)``) against
  the patch's own;
- ``demix``'s results whatever the order of the fits (``api.largest_first``)."""

from __future__ import annotations

import numpy as np
import pytest

import astrid
from astrid import api, kernels
from astrid.kernels import Kernel
from astrid.patch import Spec, build, patch_flat

FPS = 30.0


@pytest.fixture
def reference_drive(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Sets whether ``deconvolve`` runs compiled (True) or the reference loop (False)."""
    def use(compiled: bool) -> None:
        monkeypatch.setattr(kernels, "COMPILED", compiled)
    return use


@pytest.mark.parametrize("n", [1, 7, 8, 9, 64, 128, 129, 1000, 8191, 8192, 8193, 16385, 20011])
def test_numpy_sum_order(n: int) -> None:
    rng = np.random.default_rng(n)
    for scale in (1e-6, 1.0, 1e6):
        a = np.ascontiguousarray(rng.standard_normal(n) ** 2 * scale)
        assert kernels._numpy_sum(a, np.getbufsize()) == np.sum(a)


def problems():  # type: ignore[no-untyped-def]
    """Deconvolution problems over trace lengths, kernels and starting points (none, near the drive, negative)."""
    rng = np.random.default_rng(3)
    for n in (1, 2, 7, 129, 1000, 8193, 20011):
        for taus, fs in (((0.05, 0.4), 30.9), ((0.045, 0.142), 30.0), ((0.025, 2.41), 22.7)):
            k = Kernel.from_taus(*taus, fs)
            drive = np.where(rng.random(n) < 0.02, rng.uniform(1, 20, n), 0.0)
            y = kernels.calcium(drive, k) + rng.standard_normal(n) * 0.5
            for start in ("cold", "warm", "negative"):
                s0 = z0 = None
                if start == "warm":
                    s0, z0 = np.maximum(drive + rng.standard_normal(n) * 0.1, 0), np.array([0.3, 0.1])
                elif start == "negative":
                    s0, z0 = rng.standard_normal(n), np.array([-1.0, 2.0])
                yield y, float(rng.uniform(0.5, 3.0)), float(rng.uniform(0.01, 2.0)), k, s0, z0


@pytest.mark.parametrize("n_iter", [0, 1, 300])
def test_compiled_drive_update_is_the_reference_bitwise(n_iter: int, reference_drive) -> None:  # type: ignore[no-untyped-def]
    for y, weight, lam, k, s0, z0 in problems():
        reference_drive(False)
        ref = kernels.deconvolve(y, weight, lam, k, s0=s0, z0=z0, n_iter=n_iter)
        reference_drive(True)
        new = kernels.deconvolve(y, weight, lam, k, s0=s0, z0=z0, n_iter=n_iter)
        assert np.array_equal(new.s, ref.s) and np.array_equal(new.z, ref.z) and np.array_equal(new.c, ref.c)
        assert new.objective == ref.objective


def disc(shape: tuple[int, int], cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def field_scene(seed: int, t: int = 600) -> dict:
    """Five neurons of different sizes, a slow astrocyte signal over a region among them, noise on a constant
    baseline."""
    rng = np.random.default_rng(seed)
    shape = (48, 56)
    rois = np.stack([disc(shape, cy, cx, r) for cy, cx, r in
                     ((12, 12, 2.0), (12, 34, 4.0), (26, 22, 3.0), (34, 42, 5.0), (38, 12, 2.5))])
    lag = np.arange(90) / FPS
    kernel = np.exp(-lag / 0.4) - np.exp(-lag / 0.05)
    mov = 100 + rng.normal(0, 2, (t, *shape))
    for roi in rois:
        spikes = (rng.random(t) < 0.02).astype(float)
        mov += roi[None] * (np.convolve(spikes, kernel)[:t] * 40)[:, None, None]
    terr = disc(shape, 20, 30, 7)
    mov += terr[None] * (30 * np.sin(2 * np.pi * 0.05 * np.arange(t) / FPS) ** 2)[:, None, None]
    return {"movie": mov.astype(np.float32), "rois": rois, "territory": terr}


def assert_same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """Every array of two ``Result.arrays()`` equal, the fit times apart."""
    assert a.keys() == b.keys()
    for key in a:
        if key != "seconds":
            assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"), key


def test_demix_is_bitwise_the_same_with_the_compiled_drive(reference_drive) -> None:  # type: ignore[no-untyped-def]
    sc = field_scene(seed=4, t=500)
    kw = {"fps": FPS, "um_per_px": 2.0, "tau_r": 0.05, "tau_d": 0.4, "territory": sc["territory"],
          "export_components": True}
    reference_drive(False)
    ref = astrid.demix(sc["movie"], sc["rois"], **kw).arrays()
    reference_drive(True)
    new = astrid.demix(sc["movie"], sc["rois"], **kw).arrays()
    assert ref["fitted"].all()
    assert_same(ref, new)


def test_field_high_pass_gives_the_same_noise_covariance() -> None:
    a, b = field_scene(seed=2), field_scene(seed=5, t=450)
    movies = [api.as_movie(a["movie"], "rec0"), api.as_movie(b["movie"], "rec1")]
    fld = api.make_field(a["rois"], FPS, 2.0, territory=a["territory"])
    specs = [Spec(f"R-{v:04d}", label=v) for v in range(1, 6)]
    flat = np.unique(np.concatenate([patch_flat(fld, s) for s in specs]))
    plain = api.load_pixels(movies, FPS, flat)
    kept = api.load_pixels(movies, FPS, flat, high_pass=True)
    blocks = api.load_pixels(movies, FPS, flat, workers=2, high_pass=True)    # in worker processes
    assert plain.hp is None and kept.hp is not None and kept.hp_fs == FPS
    for data in (kept, blocks):                  # the rest of the pixel data is untouched
        for r in plain.y:
            assert np.array_equal(plain.y[r], data.y[r]) and np.array_equal(plain.m[r], data.m[r])
            assert np.array_equal(kept.hp[r], data.hp[r])
        assert np.array_equal(plain.noise_var, data.noise_var)
    for s in specs:
        own = build(fld, plain.subset(patch_flat(fld, s)), s)
        assert own.sigma.shape[0] < flat.size     # the patch's rows are a part of the field's
        for data in (kept, blocks):
            assert np.array_equal(own.sigma, build(fld, data.subset(patch_flat(fld, s)), s).sigma)


def test_largest_first_is_by_pixels_times_components() -> None:
    sc = field_scene(seed=2)
    fld = api.make_field(sc["rois"], FPS, 2.0, territory=sc["territory"])
    movies = [api.as_movie(sc["movie"], "rec0")]
    specs = [Spec(f"R-{v:04d}", label=v) for v in range(1, 6)]
    sizes = []
    for s in specs:
        b = build(fld, api.load_pixels(movies, FPS, patch_flat(fld, s)), s)
        sizes.append(b.patch.n_pixels * b.patch.n_components)
    order = api.largest_first(fld, specs)
    assert [sizes[i] for i in order] == sorted(sizes, reverse=True)
    assert sorted(order) == list(range(len(specs))) and order != list(range(len(specs)))


def test_demix_results_do_not_depend_on_the_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serial fits (label order) against parallel fits (largest first; in shared memory where processes
    fork)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    sc = field_scene(seed=6, t=400)
    kw = {"fps": FPS, "um_per_px": 2.0, "tau_r": 0.05, "tau_d": 0.4, "territory": sc["territory"],
          "export_components": True}
    serial = astrid.demix(sc["movie"], sc["rois"], **kw).arrays()
    parallel = astrid.demix(sc["movie"], sc["rois"], workers=2, **kw).arrays()
    assert_same(serial, parallel)
