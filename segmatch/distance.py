"""
Metrics on normalized grade sequences.

Three distances, each with a stated job:

  dtw_band        ordered shape, tolerant of bounded misalignment
  wasserstein1    unordered grade composition, exact
  hist_distance   unordered grade composition, binned, for cheap screening

and one lower bound:

  lb_keogh        provably never exceeds dtw_band, so it can prune
                  candidate windows without ever discarding one that
                  would have scored better

The lower bound is what makes the search both exact and fast. A window is
skipped only when its lower bound already exceeds the best score found so
far, which by admissibility means its true score would have too.
"""

import numpy as np

__all__ = ["dtw_band", "lb_keogh", "keogh_envelope", "wasserstein1",
           "hist_distance", "grade_histogram"]


def dtw_band(a, b, band):
    """Exact dynamic time warping under a Sakoe-Chiba band, L1 local cost.

    Returns the accumulated |difference| along the optimal admissible
    warping path, divided by max(len(a), len(b)).

    The band is the crux. Unconstrained DTW may stretch a single sample to
    cover arbitrarily many, which makes it blind to how long each feature
    physically is: a target of "1.25 mi at 8 percent then 2.5 mi at 2
    percent" scores a perfect 0.000 against a window holding only 0.25 mi
    at 8 percent. Bounding the drift to `band` samples means a feature may
    appear early or late by at most band*dx metres and still match, while
    a feature of the wrong length cannot.

    `band` is in samples and must be >= |len(a) - len(b)| or no path
    reaches the far corner; it is widened silently if it is not.

    Exactness: the recurrence explores every admissible path by induction
    on (i, j), so the value returned is the minimum over the whole
    admissible set, not an approximation of it.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("inf")
    w = int(max(band, abs(n - m), 1))

    # Plain Python lists rather than numpy scalars inside the band scan.
    # The band holds only 2w+1 cells, so per-element numpy indexing
    # overhead dominates the arithmetic; lists are several times faster at
    # this size. Values and result are identical.
    al = a.tolist()
    bl = b.tolist()
    inf = float("inf")
    # A fresh row per iteration. Reusing two buffers was tried and was
    # not measurably faster, while requiring a fiddly invariant about
    # which stale cells must be re-cleared (including index 0, which
    # holds the DTW origin and is valid only for the first row). The
    # allocation is not the bottleneck; the band scan is.
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        lo = i - w
        if lo < 1:
            lo = 1
        hi = i + w
        if hi > m:
            hi = m
        cur = [inf] * (m + 1)
        ai = al[i - 1]
        left = inf
        for j in range(lo, hi + 1):
            up = prev[j]
            diag = prev[j - 1]
            best = diag if diag < up else up
            if left < best:
                best = left
            d = ai - bl[j - 1]
            left = (d if d >= 0.0 else -d) + best
            cur[j] = left
        prev = cur
    return float(prev[m] / max(n, m))


def keogh_envelope(q, band):
    """Upper and lower envelopes of q under a Sakoe-Chiba band.

    U_i is the max of q over [i-band, i+band], L_i the min. Computed with
    a sliding maximum in O(n * band); n is small enough here that the
    monotonic-deque version is not worth the complexity.
    """
    q = np.asarray(q, dtype=float)
    n = q.size
    w = int(max(band, 1))
    idx = np.arange(n)
    lo = np.maximum(idx - w, 0)
    hi = np.minimum(idx + w + 1, n)
    up = np.empty(n)
    dn = np.empty(n)
    for i in range(n):
        seg = q[lo[i]:hi[i]]
        up[i] = seg.max()
        dn[i] = seg.min()
    return up, dn


def lb_keogh(c, up, dn, norm):
    """Lower bound on dtw_band(c, q) given q's envelope.

    For any admissible path, c_i is matched to some q_j with |i-j| <= band,
    and every such q_j lies in [dn_i, up_i]. So the local cost at i is at
    least max(0, c_i - up_i, dn_i - c_i). Costs are non-negative and each
    c_i appears at least once on the path, so the sum of those per-sample
    minima lower-bounds the path total. Dividing by the same `norm` that
    dtw_band uses keeps the bound valid after normalization.

    This is what licenses pruning: if lb_keogh already exceeds the best
    score seen, the exact distance cannot beat it either.
    """
    c = np.asarray(c, dtype=float)
    excess = np.maximum(c - up, dn - c)
    np.maximum(excess, 0.0, out=excess)
    return float(excess.sum() / norm)


def wasserstein1(a, b):
    """Exact 1D Wasserstein-1 (earth mover's) distance.

    W1 = integral |F_a(x) - F_b(x)| dx over the merged support. Order
    independent by construction, so it measures grade composition while
    dtw_band measures grade sequence.

    Exact rather than quantile-approximated: mapping samples onto plotting
    positions k/(n-1) squeezes both empirical CDFs inward and returns
    4.505 where the answer is 5.0, with the error depending on sample
    count and therefore differing between the window lengths being
    compared.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("inf")
    allv = np.concatenate([a, b])
    allv.sort()
    if allv.size < 2:
        return 0.0
    ca = np.searchsorted(a, allv[:-1], side="right") / a.size
    cb = np.searchsorted(b, allv[:-1], side="right") / b.size
    return float(np.sum(np.abs(ca - cb) * np.diff(allv)))


def grade_histogram(g, bin_w, lo=-30.0, hi=30.0):
    """Normalized histogram of grade values on a fixed grid.

    The grid is fixed rather than data-derived so two histograms are
    always directly comparable. Grades outside [lo, hi] are clipped into
    the end bins; real running terrain does not exceed those bounds, and
    clipping keeps a single spike of stream noise from adding a bin.
    """
    g = np.asarray(g, dtype=float)
    edges = np.arange(lo, hi + bin_w, bin_w)
    h, _ = np.histogram(np.clip(g, lo, hi - 1e-9), bins=edges)
    total = h.sum()
    return (h / total) if total else h.astype(float)


def hist_distance(ha, hb, bin_w):
    """Earth mover's distance between two histograms on a common grid.

    For 1D histograms this is the L1 norm of the cumulative difference,
    scaled by bin width, which is the discrete analogue of wasserstein1.
    Using EMD rather than a per-bin difference matters: a per-bin metric
    calls 7 vs 8 percent and 2 vs 10 percent equally wrong.
    """
    return float(np.abs(np.cumsum(ha) - np.cumsum(hb)).sum() * bin_w)
