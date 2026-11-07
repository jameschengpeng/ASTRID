"""SynQuant's z-score map (Wang et al. 2020), the proposal step of ``astrid.detection``.

The order-statistics test is adapted from the SynQuant (Wang et al. 2020) and AQuA2 (Mi et al. 2025) MATLAB code
of Guoqiang Yu's lab at Virginia Tech (now at Tsinghua University), with modifications by James Cheng Peng, used
under the MIT license with the permission of Guoqiang Yu's lab.

``zmap(img)``: for every intensity threshold from the image's maximum down to ``MIN_INTENSITY`` (100) in steps of
``STEP`` (30), each 4-connected component of ``MIN_SIZE`` to ``MAX_SIZE`` pixels (5 to 100) of the thresholded
image is compared with a ring of neighbouring pixels (``get_neighbor2d``). The comparison is an order-statistics
test of whether the component is brighter than its surround: the difference of the two means, in units of the
surround's noise corrected for the selection by rank (``std_correction``), minus the difference expected from
the ranks alone, over the test's standard deviation (``ord_stat_approx_ksec``). Each pixel keeps its highest
z-score over the thresholds, and ``Z_FLOOR`` (-10) where no component was scored.

Arithmetic follows the MATLAB code: the image is single precision, and operations between single and double
values are done in single precision after converting the double, as MATLAB does.

Wang Y, Wang C, Ranefall P, et al. SynQuant: an automatic tool to quantify synapses from microscopy images.
Bioinformatics 36(5):1599-1606 (2020). Mi X, Chen AB-Y, Duarte D, et al. Fast, accurate, and versatile data
analysis platform for the quantification of molecular spatiotemporal signals. Cell 188(10):2794-2809 (2025).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.special import erfc, erfcinv
from scipy.stats import rankdata

from astrid.detection import Z_FLOOR, connected_components, from_linear, mask_from_pixels

F32 = np.float32
SQRT2PI = np.sqrt(2 * np.pi)
MAX_SIZE, MIN_SIZE = 100, 5         # pixels: the components scored
MIN_INTENSITY, STEP = 100.0, 30.0   # the lowest threshold and the step down from the image's maximum


def normcdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(-np.asarray(x, dtype=np.float64) / np.sqrt(2))


def normpdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x**2) / SQRT2PI


def norminv(p: np.ndarray) -> np.ndarray:
    return -np.sqrt(2) * erfcinv(2 * np.asarray(p, dtype=np.float64))


def matlab_colon(a: float, step: float, b: float) -> np.ndarray:
    """MATLAB ``a:step:b`` for doubles: the first half a + k step, the second half counted back from the end."""
    n = int(np.floor((b - a) / step + 1e-10))
    k = np.arange(n + 1, dtype=np.float64)
    last = a + n * step
    half = (n + 1) // 2
    out = a + k * step
    out[half:] = last - (n - k[half:]) * step
    return out


def _f1(x: np.ndarray) -> np.ndarray:
    return x * normcdf(x) + normpdf(x)


def _f2(x: np.ndarray) -> np.ndarray:
    c = normcdf(x)
    return 0.5 * (c * x**2 - c + normpdf(x) * x)


# ---------------------------------------------------------------- order statistics
@lru_cache(maxsize=4096)
def _ord_variance(n: int) -> tuple[np.ndarray, np.ndarray]:
    p = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n
    mus = norminv(p)
    vars_ = p * (1 - p) / normpdf(mus) ** 2 / n
    x = matlab_colon(-8.0, 0.001, 8.0)
    cdf = normcdf(x)
    with np.errstate(divide="ignore"):
        big_f = np.log(cdf)
        big_f2 = np.log(1 - cdf)
    f = np.log(normpdf(x))
    c = 0.0
    for r in range(1, min(int(np.ceil(n / 2)), 30) + 1):
        c = c - np.log(r) + np.log(n - r + 1)
        with np.errstate(invalid="ignore"):
            cur = np.exp(c + (r - 1) * big_f + (n - r) * big_f2 + f)
        cur = cur / cur.sum()
        mus[r - 1] = x @ cur
        mus[n - r] = -mus[r - 1]
        vars_[r - 1] = (x**2 - mus[r - 1] ** 2) @ cur
        vars_[n - r] = vars_[r - 1]
    mus.setflags(write=False)
    vars_.setflags(write=False)
    return mus, vars_


def get_ord_variance(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The expected value and variance of each order statistic of n standard normals (the extreme ranks, up
    to 30 at each end, by numerical integration; the others by the normal approximation)."""
    return _ord_variance(int(n))


def _floor_tiedrank(v: np.ndarray) -> np.ndarray:
    return np.floor(rankdata(v, method="average")).astype(np.int64)


def std_correction(fg: np.ndarray, bg: np.ndarray) -> np.float32:
    """The surround's standard deviation corrected for the bias of selecting pixels by rank (single
    precision)."""
    m, n_bg = fg.size, bg.size
    ranking = _floor_tiedrank(np.concatenate([fg, bg]))
    mu_s, _ = get_ord_variance(m + n_bg)
    a_bg = (np.sum(mu_s[ranking[m:] - 1]) / n_bg) ** 2
    var_bg = np.var(bg.astype(F32), ddof=1, dtype=F32)
    sigma_sqr = F32(var_bg) / F32(1 - a_bg)
    return F32(np.sqrt(sigma_sqr, dtype=F32))


def get_vec_01(fg: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """1 at each rank held only by foreground pixels, 0 elsewhere."""
    m = fg.size
    ranking = _floor_tiedrank(np.concatenate([fg, bg]))
    total = ranking.size
    vec = np.zeros(total)
    pos = np.arange(1, total + 1)
    for r in range(1, total + 1):
        where = pos[ranking == r]
        if where.size and np.all(where <= m):
            vec[r - 1] = 1
    return vec


def get_t1_new(ai: np.ndarray, finv_t: np.ndarray, finv_b: np.ndarray) -> float:
    """sum_i ai (f1(t_i) - f1(b_i)) sum_{j>i} aj (t_j - b_j), in O(K)."""
    prev = float(ai[1:] @ (finv_t[1:] - finv_b[1:]))
    t1 = float(ai[0] * prev * (_f1(finv_t[0]) - _f1(finv_b[0])))
    for i in range(1, ai.size - 1):
        prev = prev - ai[i] * (finv_t[i] - finv_b[i])
        t1 = t1 + ai[i] * prev * (_f1(finv_t[i]) - _f1(finv_b[i]))
    return t1


def ord_stat_approx_ksec(fg: np.ndarray, bg: np.ndarray) -> tuple[float, float]:
    """(mu, sigma) of the rank statistic of foreground against background, ties averaged."""
    if fg.size == 0 or bg.size == 0:
        return float("nan"), float("nan")
    m, n_bg = fg.size, bg.size
    n = m + n_bg
    values = np.concatenate([bg.ravel(), fg.ravel()])
    labels = np.concatenate([-np.ones(n_bg), np.ones(m)])
    od = np.argsort(values, kind="stable")
    labels = labels[od]
    sorted_vals = np.sort(values)
    sorted_rnk = np.searchsorted(sorted_vals, sorted_vals, side="left") + 1   # competition ranks, sorted
    ai = np.empty(n)
    group_start = 0
    while group_start < n:
        group_end = group_start
        while group_end + 1 < n and sorted_rnk[group_end + 1] == sorted_rnk[group_start]:
            group_end += 1
        lab = labels[group_start:group_end + 1]
        n_fg_g, n_bg_g = int(np.sum(lab == 1)), int(np.sum(lab == -1))
        ai[group_start:group_end + 1] = (n / m * n_fg_g - n / n_bg * n_bg_g) / (n_fg_g + n_bg_g)
        group_start = group_end + 1
    ii = np.arange(1, n + 1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        finv_b = norminv((ii - 1) / n)
        finv_t = norminv(ii / n)
    finv_b[0] = -1e5
    finv_t[-1] = 1e5
    mu = float(np.sum(ai * (normpdf(finv_b) - normpdf(finv_t))))
    t1 = get_t1_new(ai, finv_t, finv_b)
    t2 = float(np.sum(ai * ai * finv_t * (_f1(finv_t) - _f1(finv_b))))
    t3 = float(np.sum(ai * ai * (_f2(finv_t) - _f2(finv_b))))
    big_a = 2 * (t1 + t2 - t3)
    big_b = float(ai @ (_f1(finv_t) - _f1(finv_b))) ** 2
    return mu, float(np.sqrt(big_a - big_b) / np.sqrt(n))


# ---------------------------------------------------------------- neighbours and z-scores
def get_neighbor2d(shape: tuple[int, int], com: np.ndarray) -> np.ndarray:
    """The surround of a component: a square ring (radius 1, or 2 for >= 50 px) grown until it holds at least
    80% as many pixels as the component, stopping early when the previous ring's share of the component
    exceeds the component's share of the new ring. Sorted 0-based column-major indices."""
    mask = mask_from_pixels(shape, com)
    size = com.size
    cycle = 2 if size >= 50 else 1
    prev = np.empty(0, dtype=np.int64)
    nb = np.empty(0, dtype=np.int64)
    while nb.size < 0.8 * size:
        dil = ndi.binary_dilation(mask, structure=np.ones((2 * cycle + 1, 2 * cycle + 1), dtype=bool))
        nb = np.flatnonzero((dil & ~mask).ravel(order="F"))
        if prev.size / size > size / nb.size:
            nb = prev
            break
        prev = nb
        cycle += 1
    return nb


def component_z(img: np.ndarray, com: np.ndarray) -> float:
    """The z-score of one component (NaN where it cannot be computed)."""
    nb = get_neighbor2d(img.shape, com)
    flat = img.ravel(order="F")
    fg, bg = flat[com].astype(F32), flat[nb].astype(F32)
    s = std_correction(fg, bg)
    big_l = F32(fg.mean(dtype=F32) - bg.mean(dtype=F32))
    _, sigma = ord_stat_approx_ksec(fg / s, bg / s)
    mus, _ = get_ord_variance(fg.size + bg.size)
    v01 = get_vec_01(fg, bg)
    with np.errstate(invalid="ignore"):
        mu = np.mean(mus[v01 == 1]) - np.mean(mus[v01 == 0]) if (v01 == 1).any() and (v01 == 0).any() \
            else float("nan")
    return float(F32(F32(big_l / s) - F32(mu)) / F32(sigma))


def synquant_thre(img: np.ndarray, max_size: int, min_size: int, thre: float) -> np.ndarray:
    """The z-score map of one intensity threshold (``Z_FLOOR`` where no component was scored)."""
    z_map = np.full(img.shape, Z_FLOOR)
    for com in connected_components(img >= F32(thre)):
        if min_size <= com.size <= max_size:
            z = component_z(img, com)
            if not np.isnan(z):
                z_map[from_linear(img.shape, com)] = max(z, Z_FLOOR)
    return z_map


def synquant_extraction(img: np.ndarray, max_size: int, min_size: int, max_intensity: float,
                        min_intensity: float, step: float = STEP) -> np.ndarray:
    """The element-wise maximum of ``synquant_thre`` over the thresholds max_intensity:-step:min_intensity."""
    z_map = np.full(img.shape, Z_FLOOR)
    thre = float(max_intensity)
    while thre >= min_intensity - 1e-9:
        z_map = np.maximum(z_map, synquant_thre(img, max_size, min_size, float(F32(thre))))
        thre -= step
    return z_map


def zmap(img: np.ndarray) -> np.ndarray:
    """SynQuant's z-score map of ``img`` (float32) as the paper's detection ran it: components of 5 to 100
    pixels, thresholds from the image's maximum down to 100 in steps of 30."""
    img = np.asarray(img, dtype=F32)
    return synquant_extraction(img, MAX_SIZE, MIN_SIZE, float(img.max()), MIN_INTENSITY)
