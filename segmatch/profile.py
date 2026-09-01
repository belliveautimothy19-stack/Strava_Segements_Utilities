"""
Normalized route representation.

The matcher never touches raw GPX points or raw Strava streams. Everything
is first converted to a Profile: elevation and grade sampled on a uniform
arc-length grid. That single step removes the largest source of spurious
mismatch in the old implementation, which compared a target measured at
1 to 4 m GPX spacing against windows measured on a 60 to 100 m decimated
grid and reported a 50 percent gain deviation for identical terrain.

Two ideas carry the whole module:

1. Uniform arc-length sampling. Once both profiles live on a grid of the
   same spacing, every downstream operation is sampling-rate independent.
   Two recordings of the same hill at different GPS rates produce the same
   Profile to within interpolation error.

2. Grade is estimated at an explicit physical scale. Grade is a derivative,
   and differentiating noisy elevation amplifies noise by 1/dx. The scale
   at which the derivative is taken is therefore a real modelling choice,
   not an implementation detail, so it is a named parameter (res_m) rather
   than a constant buried in a call. The estimator is the ordinary least
   squares slope over a window of width res_m, which uses every sample in
   the window instead of only its two endpoints.
"""

import numpy as np

__all__ = ["Profile", "sanitize", "build_profile", "vertical_change",
           "detect_quantization"]


class Profile:
    """Elevation and grade on a uniform arc-length grid.

    Attributes:
        dx       sample spacing in metres
        length   total profile length in metres
        elev     elevation in metres at k*dx, shape (n,)
        grade    grade in percent at k*dx, shape (n,)
        res_m    physical scale the grade was estimated over
    """

    __slots__ = ("dx", "length", "elev", "grade", "res_m", "_grid",
                 "_vgrid", "_cum_gain", "_cum_loss")

    def __init__(self, dx, elev, grade, res_m, vert_resample_m=25.0):
        self.dx = float(dx)
        self.elev = elev
        self.grade = grade
        self.res_m = float(res_m)
        self.length = float((len(elev) - 1) * dx)
        # Cached once. grade_at and elev_at are called for every candidate
        # window, so rebuilding the distance axis inside them showed up as
        # a measurable share of total runtime.
        self._grid = np.arange(len(elev)) * self.dx

        # Cumulative ascent and descent on a fixed grid, so a window's
        # vertical is two interpolations rather than a fresh resample and
        # sum. Ascent is additive over adjacent intervals, so the
        # cumulative difference is exact whenever the window boundaries
        # land on grid points and off by at most one cell otherwise.
        n_v = max(2, int(round(self.length / float(vert_resample_m))) + 1)
        self._vgrid = np.linspace(0.0, self.length, n_v)
        ve = np.interp(self._vgrid, self._grid, elev)
        d = np.diff(ve)
        self._cum_gain = np.concatenate(([0.0], np.cumsum(
            np.where(d > 0, d, 0.0))))
        self._cum_loss = np.concatenate(([0.0], np.cumsum(
            np.where(d < 0, -d, 0.0))))

    def __len__(self):
        return len(self.elev)

    def grade_at(self, dists):
        """Grade sampled at arbitrary distances along the profile."""
        return np.interp(dists, self._grid, self.grade)

    def elev_at(self, dists):
        return np.interp(dists, self._grid, self.elev)

    def window_vertical(self, start, end):
        """(gain, loss) in metres over [start, end], in O(1)."""
        g0, g1 = np.interp([start, end], self._vgrid, self._cum_gain)
        l0, l1 = np.interp([start, end], self._vgrid, self._cum_loss)
        return float(g1 - g0), float(l1 - l0)


def sanitize(cum_dist, elev):
    """Return (cum_dist, elev) as finite float arrays with strictly
    increasing distance.

    Handles the three malformed inputs seen in practice:
      - missing or NaN elevations, interpolated across from neighbours
      - duplicate distance samples from a stationary recorder, which make
        np.interp ambiguous and can produce an infinite grade
      - non monotonic distance, which is dropped rather than trusted

    Raises ValueError if nothing usable survives.
    """
    d = np.asarray(cum_dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    if d.size != e.size:
        raise ValueError("distance and elevation lengths differ")
    if d.size < 2:
        raise ValueError("need at least two samples")

    good_e = np.isfinite(e)
    if not good_e.any():
        raise ValueError("no finite elevation samples")
    if not good_e.all():
        e = np.interp(d, d[good_e], e[good_e])

    finite_d = np.isfinite(d)
    d, e = d[finite_d], e[finite_d]
    # Keep only strictly increasing distance. np.maximum.accumulate then a
    # forward difference test is O(n) and drops both duplicates and any
    # backward jump.
    keep = np.ones(d.size, dtype=bool)
    keep[1:] = np.diff(d) > 0
    d, e = d[keep], e[keep]
    if d.size < 2:
        raise ValueError("profile collapses to a single point")
    return d, e


def _ols_slope_kernel(n):
    """Weights w such that dot(w, y) is the least squares slope of y
    against an evenly spaced x with unit spacing.

    slope = sum((x_k - xbar) * y_k) / sum((x_k - xbar)^2)
    """
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    return xc / np.dot(xc, xc)


def build_profile(cum_dist, elev, res_m, oversample=4,
                  vert_resample_m=25.0):
    """Resample onto a uniform grid and estimate grade at scale res_m.

    dx is res_m / oversample. Oversampling keeps the grid fine enough that
    the smoothed derivative is not aliased, while res_m alone controls the
    physical resolution of the representation. Both profiles in any
    comparison must be built with the same res_m and oversample so their
    grids align.

    Returns a Profile, or None if the input is too short to support even
    one grade estimate at this resolution.
    """
    d, e = sanitize(cum_dist, elev)
    total = float(d[-1] - d[0])
    dx = float(res_m) / float(oversample)
    if total <= 0 or total < dx:
        return None

    # Cover the input span exactly. Taking int(total // dx) + 1 samples
    # discards the final partial cell, which silently drops up to dx of
    # real terrain from the end of every profile: a 5992 m segment became
    # a 5970 m one. The tail is then unsearchable, and a target's own
    # length is quantized. Instead, round to the nearest whole number of
    # cells and stretch the spacing by under half a cell to land on the
    # true endpoint.
    n = max(2, int(round(total / dx)) + 1)
    grid = np.linspace(d[0], d[-1], n)
    dx = float(grid[1] - grid[0])
    ge = np.interp(grid, d, e)

    # Grade as the OLS slope over a window of res_m, in percent. The window
    # spans `oversample + 1` samples so it covers exactly res_m of ground.
    win = int(oversample) + 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 2:
        return None
    k = _ols_slope_kernel(win)
    half = win // 2

    # Pad by linear extrapolation so every output sample, including the
    # first and last, is estimated from a full-width window. Computing
    # edge samples from a truncated window instead gives them a different
    # estimator with different bias, which made the shape distance
    # oversensitive to where a profile happens to end: truncating a 6000 m
    # segment by 40 m moved the shape term by 0.58.
    if n >= win:
        head_slope = (ge[win - 1] - ge[0]) / (win - 1)
        tail_slope = (ge[-1] - ge[-win]) / (win - 1)
    else:
        head_slope = tail_slope = (ge[-1] - ge[0]) / max(1, n - 1)
    head = ge[0] + np.arange(-half, 0) * head_slope
    tail = ge[-1] + np.arange(1, half + 1) * tail_slope
    padded = np.concatenate([head, ge, tail])
    # correlate 'valid' with the padded array yields exactly n samples,
    # each the OLS slope over a centred window of width `win`.
    slope = np.correlate(padded, k, mode="valid")
    grade = slope / dx * 100.0
    grade[~np.isfinite(grade)] = 0.0
    return Profile(dx, ge, grade, res_m, vert_resample_m)


def vertical_change(cum_dist, elev, resample_m=25.0):
    """Total ascent and descent in metres, measured at a fixed spatial
    interval.

    Gain is not a property of terrain until the sampling interval is
    fixed: the same hill measures 2172 m of gain sampled every 1 m and
    300 m sampled every 100 m, because barometric jitter accumulates.
    Fixing the interval makes the number a property of the hill, and makes
    a target and a candidate window comparable.

    Returns positive magnitudes (gain, loss).
    """
    d, e = np.asarray(cum_dist, dtype=float), np.asarray(elev, dtype=float)
    if d.size < 2:
        return 0.0, 0.0
    good = np.isfinite(e)
    if not good.any():
        return 0.0, 0.0
    if not good.all():
        e = np.interp(d, d[good], e[good])
    span = float(d[-1] - d[0])
    if span <= 0:
        return 0.0, 0.0
    n = max(2, int(round(span / float(resample_m))) + 1)
    grid = np.linspace(d[0], d[-1], n)
    diff = np.diff(np.interp(grid, d, e))
    diff = diff[np.isfinite(diff)]
    # abs() rather than negation: summing an empty selection yields -0.0,
    # which prints as "-0 ft".
    return float(diff[diff > 0].sum()), float(abs(diff[diff < 0].sum()))


def detect_quantization(elev, max_step_m=10.0):
    """Estimate the elevation quantization step, or 0.0 if unquantized.

    Elevation served from a DEM is often rounded, and rounding is not
    noise: it is a deterministic staircase whose steps can be large
    compared with the elevation change across one grade sample, so it
    manufactures alternating flat and steep samples on a uniform climb.
    The finer the grade resolution, the more of that artifact survives
    into the representation. Measured on synthetic terrain, the same route
    quantized to 5 m scored 1.48 against its unquantized self at
    res_m 120, 2.52 at res_m 70 and 3.82 at res_m 50, where unrelated
    terrain scored about 3.1. At 50 m resolution, coarse quantization is
    therefore worse than no match at all.

    Returned so callers can warn and suggest a coarser resolution rather
    than silently producing bad matches. Works by taking the greatest
    common divisor of the distinct elevation values, in units of 1 cm.
    """
    e = np.asarray(elev, dtype=float)
    e = e[np.isfinite(e)]
    if e.size < 8:
        return 0.0
    cent = np.round(e * 100.0).astype(np.int64)
    uniq = np.unique(cent)
    if uniq.size < 3:
        return 0.0
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    step = int(np.gcd.reduce(diffs))
    step_m = step / 100.0
    # A step of 1 cm or less is just float noise, not quantization.
    return step_m if 0.01 < step_m <= max_step_m else 0.0
