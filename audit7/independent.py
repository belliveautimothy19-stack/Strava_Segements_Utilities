"""
An independent measuring instrument.

WHY THIS EXISTS

Six audits in a row found defects in the measurement code rather than in
the matcher: a fixed-grid interpolation artifact, an inert quantization
probe, contaminated category A positives, an archetype classifier that
keyed on sign changes alone, a pooled category B whose composition moved
with the variable under study, a bounding-box projection bug, a false
negative rate pinned at 0.5 by construction, AUC quoted without an
interval on samples of twelve, and a grade fraction read as a percent.

Every one of those was found by the same person who wrote the code that
contained it, using the same abstractions. That is the structural
problem this module addresses. It recomputes, from the raw cached
streams, everything the conclusions depend on:

    window extraction        arc length from raw samples
    grade representation     finite differences on a uniform grid
    geographic overlap       spherical law of cosines, one fixed frame
    category labels          geometry and independently derived shape
    AUC and intervals        rank statistics written out longhand

It imports NOTHING from segmatch or bench except the matcher entry point
whose output is under test. It deliberately uses different algorithms
where a different algorithm is possible: central differences rather than
a windowed OLS slope, haversine rather than an equirectangular
projection, an explicit double loop for AUC rather than a broadcast
comparison. Where the two paths agree, the agreement means something,
because a shared bug would have to be reproduced twice in two different
forms.

It is deliberately boring. Clarity beats speed here; a fast instrument
nobody can check is worth less than a slow one anybody can.

WHAT IT ASSERTS

Assertions are the point, not decoration. Each one corresponds to a class
of defect that has actually occurred in this project or is a near
neighbour of one that has:

    coordinate frames       one projection origin for all geometry
    array lengths           distance, elevation and GPS must agree
    grade units             percent, checked against a known ramp
    overlap bounds          a fraction in [0, 1], symmetric
    duplicate windows       identical byte content cannot be a positive
    pair disjointness       a positive may not overlap along the route
    population leakage      no window may appear in two categories
    sample adequacy         an interval wider than the effect is refused
    degenerate thresholds   a threshold at a distribution edge is refused
"""

import math

import numpy as np

# The matcher is the object under test. Nothing else is imported from it.
from segmatch.match import MatchConfig, prepare_target, match_segment

EARTH_R = 6371008.8          # metres, IUGG mean radius
SAME_GROUND_M = 40.0         # two fixes closer than this are one place
GEO_SAME_FRAC = 0.70         # overlap to call two windows one hill
GEO_APART_FRAC = 0.10        # overlap below which they are separate
MIN_APART_M = 150.0          # and they must also be this far apart


# ---------------------------------------------------------------------
# geometry, recomputed from scratch
# ---------------------------------------------------------------------

def haversine_m(a_lat, a_lon, b_lat, b_lon):
    """Great-circle distance. Written out rather than projected, so it
    shares no code and no approximation with the production path or with
    the bench module whose projection bug this replaces."""
    p1 = np.radians(a_lat)
    p2 = np.radians(b_lat)
    dp = np.radians(b_lat - a_lat)
    dl = np.radians(b_lon - a_lon)
    h = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def min_distance_to_set(A, B, block=256):
    """For each fix in A, the distance to the nearest fix in B, metres.

    Blocked only to bound memory. No projection, no bounding box, no
    early exit: the prefilter that carried the projection bug is
    deliberately absent, and this is the reference the fast path must
    agree with.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    out = np.empty(len(A))
    for i in range(0, len(A), block):
        chunk = A[i:i + block]
        d = haversine_m(chunk[:, 0][:, None], chunk[:, 1][:, None],
                        B[:, 0][None, :], B[:, 1][None, :])
        out[i:i + block] = d.min(axis=1)
    return out


def overlap_fraction(A, B):
    """(overlap, min_separation_m), symmetric by construction.

    Symmetry is taken as the max of the two directions, matching the
    earlier definition so the numbers are comparable, but both directions
    are returned so asymmetry cannot hide.
    """
    da = min_distance_to_set(A, B)
    db = min_distance_to_set(B, A)
    fa = float(np.mean(da <= SAME_GROUND_M))
    fb = float(np.mean(db <= SAME_GROUND_M))
    assert 0.0 <= fa <= 1.0 and 0.0 <= fb <= 1.0
    return max(fa, fb), float(min(da.min(), db.min())), (fa, fb)


# ---------------------------------------------------------------------
# terrain representation, recomputed from scratch
# ---------------------------------------------------------------------

def resample_uniform(dist, elev, step_m):
    """Uniform arc-length resampling by linear interpolation.

    Deliberately the simplest correct thing. The production path builds a
    Profile with an oversampled grid and a windowed OLS slope; this does
    not, so a defect in that construction cannot reach both.
    """
    dist = np.asarray(dist, float)
    elev = np.asarray(elev, float)
    assert dist.shape == elev.shape, (dist.shape, elev.shape)
    assert np.all(np.diff(dist) > 0), "distance must be strictly increasing"
    n = int(math.floor((dist[-1] - dist[0]) / step_m)) + 1
    assert n >= 3, "too few samples for a profile"
    grid = dist[0] + np.arange(n) * step_m
    return grid, np.interp(grid, dist, elev)


def grade_percent(grid, elev):
    """Grade in PERCENT by central differences.

    Percent, not fraction. The unit is asserted against a known ramp in
    the test suite because a fraction read as a percent has already cost
    this project one experiment: a 5 percent ramp built as grade=5.0
    produced a 500 percent slope and a saturated score that was briefly
    mistaken for a resolution measurement.
    """
    step = grid[1] - grid[0]
    g = np.gradient(elev, step) * 100.0
    assert np.all(np.isfinite(g))
    return g


def vertical_change_m(dist, elev, step_m=25.0):
    """Gain and loss at a fixed spatial interval, so the answer does not
    depend on how densely the route happened to be recorded."""
    _, e = resample_uniform(dist, elev, step_m)
    d = np.diff(e)
    return float(d[d > 0].sum()), float(-d[d < 0].sum())


# ---------------------------------------------------------------------
# rank statistics, written longhand
# ---------------------------------------------------------------------

def auc_lower_is_better(pos, neg):
    """P(a positive scores below a negative), ties at half.

    An explicit double loop. The broadcast version is faster and is what
    the other instrument uses; this one exists so that the two can be
    checked against each other.
    """
    pos = list(map(float, pos))
    neg = list(map(float, neg))
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p < n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def bootstrap_ci(pos, neg, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for the AUC."""
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = np.asarray(pos, float)
    n = np.asarray(neg, float)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        a = rng.choice(p, len(p), replace=True)
        b = rng.choice(n, len(n), replace=True)
        # broadcast here is fine; the longhand loop above pins its meaning
        vals[i] = ((a[:, None] < b[None, :]).sum()
                   + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b))
    return (float(np.quantile(vals, alpha / 2.0)),
            float(np.quantile(vals, 1.0 - alpha / 2.0)))


MIN_POSITIVES_FOR_ADEQUACY = 30


def interval_is_adequate(lo, hi, effect, n_pos=None,
                         min_pos=MIN_POSITIVES_FOR_ADEQUACY):
    """Is the comparison strong enough to resolve the effect claimed?

    Interval width alone is not sufficient, and the shortfall is not
    theoretical: with twelve positives cleanly separated from four
    hundred negatives, the bootstrap AUC interval is [0.9925, 1.0]. It is
    narrow because the statistic is pinned against its ceiling, not
    because the estimate is well determined. Every resample lands at the
    same place. A width test alone passes that case and would have
    licensed exactly the "AUC 0.986 from twelve pairs" claim this gate
    exists to refuse.

    So adequacy requires both a narrow interval AND enough positives for
    the interval to mean anything. The count floor is a blunt instrument
    and is deliberately so: it does not depend on the data, and therefore
    cannot be moved by the data.
    """
    if (hi - lo) >= effect:
        return False
    if n_pos is not None and n_pos < min_pos:
        return False
    return True


def cluster_bootstrap_ci(pos, neg, pos_trail, n_boot=2000, seed=0, alpha=0.05):
    """AUC interval resampling TRAILS, not pairs.

    Pairs cut from one trail are nested observations, not independent
    ones: 59 category A pairs from a single 24 km trail are repeated
    looks at the same terrain. Resampling pairs treats them as 59
    independent draws and returns an interval that is too narrow. Every
    interval in audits 4 to 7 has that flaw.

    `pos_trail` gives the trail identifier for each positive. Trails are
    drawn with replacement and all of a drawn trail's pairs come with it,
    so between-trail variance enters the interval where it belongs.

    Negatives are resampled at the pair level: they are drawn from across
    the whole corpus and are not clustered in the same way. That is a
    simplification and is stated rather than hidden; it makes the
    interval slightly narrower than a fully clustered one would be.
    """
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    # Coerce to opaque string labels. Trail identifiers are naturally
    # tuples of route ids, and numpy turns a list of tuples into a 2-D
    # array rather than an array of labels.
    trails = np.asarray([str(t) for t in pos_trail], dtype=object)
    uniq = np.unique(trails)
    if len(uniq) < 2:
        # One trail cannot support a between-trail interval. Say so rather
        # than returning a number that looks like one.
        return float("nan"), float("nan")
    by = {t: pos[trails == t] for t in uniq}
    vals = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.choice(uniq, len(uniq), replace=True)
        p = np.concatenate([by[t] for t in drawn])
        b = rng.choice(neg, len(neg), replace=True)
        vals[i] = ((p[:, None] < b[None, :]).sum()
                   + 0.5 * (p[:, None] == b[None, :]).sum()) / (len(p) * len(b))
    return (float(np.quantile(vals, alpha / 2.0)),
            float(np.quantile(vals, 1.0 - alpha / 2.0)))
