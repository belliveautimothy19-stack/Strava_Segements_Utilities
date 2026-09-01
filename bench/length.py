"""
Does the matcher hold up at the length the tool is actually used for?

Every earlier experiment cut 1000 m windows. The stated use case is a
6 km segment, six times longer, and nothing had tested that. Two
questions are separated here because they have different answers and
different consequences.

  A  physical shape recovery   the same ground, re-measured or
                               re-sampled, must come back as the same
                               shape. What is being measured is a
                               property of the matcher alone, since each
                               window is compared only against itself and
                               no other route can affect the answer.

                               It is NOT corpus-free. `recovery()` reads
                               real cached GPS streams through
                               `bench.semantic._routes_with_gps()`, so it
                               needs the private real-data corpus and
                               reports nothing without it. An earlier
                               version of this text said "needs no
                               corpus", which conflated "no OTHER route
                               influences the score" with "no data
                               required", and left a corpus-availability
                               failure looking like a matcher defect.

  B  archetype similarity      a different piece of ground of the same
                               kind must score closer than unrelated
                               ground. This is a retrieval question and
                               its answer depends on what else is in the
                               corpus, which is why the corpus is
                               reported alongside every number.

PROVENANCE

Category A at long window lengths is built from out-and-back retraces,
because that is the only place real data offers a genuine second pass
over the same ground. A single long route can therefore supply every A
pair on its own, and an accuracy figure computed from it describes that
one trail rather than the matcher. Every report here prints the
route-pairings behind its A pairs so that the reader can see how wide
the evidence actually is.
"""

import itertools

import numpy as np

from bench.semantic import (build_windows, categorize, score_pairs,
                            _routes_with_gps)
from segmatch.match import MatchConfig, prepare_target, match_segment
from segmatch.profile import build_profile


def provenance(pairs, cat="A"):
    """Which route pairings produced the pairs of one category."""
    counts = {}
    for p in pairs:
        if p["cat"] != cat and not p["cat"].startswith(cat):
            continue
        key = tuple(sorted((p["A"]["route"], p["B"]["route"])))
        counts[key] = counts.get(key, 0) + 1
    return counts


def auc(pos, neg):
    """Probability a positive outranks a negative. Lower score is better,
    so the comparison is inverted."""
    if not pos or not neg:
        return float("nan")
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    wins = (pos[:, None] < neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n_boot=2000, seed=0):
    """Bootstrap 95 percent interval for the AUC.

    Reported because the interval, not the point estimate, is what
    decides whether two configurations differ. Category A at long window
    lengths draws on a handful of retraces, and an AUC computed from a
    dozen positives carries an interval wide enough to swallow every
    difference this sweep was built to detect. Quoting the point estimate
    alone would invite exactly the false conclusion.
    """
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    vals = []
    for _ in range(n_boot):
        a = rng.choice(pos, len(pos), replace=True)
        b = rng.choice(neg, len(neg), replace=True)
        vals.append(auc(a.tolist(), b.tolist()))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _bucket(scored):
    """Group scores by category, keeping the two B variants apart.

    They used to be pooled into a single B. That number fell from 0.735
    at 1 km to 0.574 at 3 km and read as the matcher losing its grip on
    terrain similarity at long range. It was an artifact of the pooling:
    B_strict is flat at 0.75 to 0.84 across the same range and B_loose
    sits at 0.41 to 0.63, so the pooled value tracks nothing but the
    shifting ratio between them. A pooled "B" is still reported for
    continuity with the earlier runs and is marked as uninterpretable.
    """
    b = {}
    for r in scored:
        b.setdefault(r["cat"], []).append(r["score"])
        if r["cat"].startswith("B"):
            b.setdefault("B_pooled_do_not_interpret", []).append(r["score"])
    return b


def operating_point(pos, neg, recall=0.90):
    """Threshold that accepts `recall` of the positives, and the false
    positive rate it costs.

    An earlier version of this report set the threshold at the median of
    the positives and then reported the false negative rate against it.
    That number is 0.5 by construction and says nothing about the
    matcher; it appeared as a flat 0.494 at every window length, which is
    what exposed it. The threshold is now stated as a recall target so
    that the reported false positive rate is the quantity that varies.
    """
    if not pos or not neg:
        return float("nan"), float("nan"), float("nan")
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    thr = float(np.quantile(pos, recall))
    return thr, float(np.mean(neg <= thr)), float(np.mean(pos > thr))


def eer(pos, neg):
    """Equal error rate: the single threshold-free summary of the
    operating curve that does not need a recall target chosen by hand."""
    if not pos or not neg:
        return float("nan"), float("nan")
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    cand = np.unique(np.concatenate([pos, neg]))
    best, bt = 1.0, float("nan")
    for t in cand:
        fn = float(np.mean(pos > t))
        fp = float(np.mean(neg <= t))
        if abs(fn - fp) < best:
            best, bt = abs(fn - fp), t
    fn = float(np.mean(pos > bt))
    fp = float(np.mean(neg <= bt))
    return (fn + fp) / 2.0, float(bt)


def _subsample(pairs, max_per_cat, seed):
    """Cap each category at `max_per_cat` pairs, chosen at random.

    Short windows produce hundreds of thousands of pairs, almost all of
    them category D, and scoring every one costs more than it buys: the
    quantities reported are rank statistics, whose standard error is set
    by the sample size and not by the fraction of the population sampled.
    The cap is applied per category so the small categories are never
    thinned, and the seed is fixed so a sweep is reproducible.
    """
    if not max_per_cat:
        return pairs
    rng = np.random.default_rng(seed)
    by = {}
    for i, p in enumerate(pairs):
        by.setdefault(p["cat"], []).append(i)
    keep = []
    for c, idx in by.items():
        if len(idx) > max_per_cat:
            idx = rng.choice(idx, max_per_cat, replace=False)
        keep.extend(int(i) for i in idx)
    keep.sort()
    return [pairs[i] for i in keep]


def length_report(win_m, res_m=70.0, stride_m=None, cap=None,
                  max_per_cat=600, seed=0):
    stride_m = stride_m if stride_m is not None else win_m / 4.0
    w = build_windows(win_m=win_m, stride_m=stride_m)
    pairs = _subsample(categorize(w), max_per_cat, seed)
    scored = score_pairs(pairs, res_m, cap=cap)
    b = _bucket(scored)
    a, d = b.get("A", []), b.get("D", [])
    b_strict, b_loose = b.get("B_strict", []), b.get("B_loose", [])
    bb = b.get("B_pooled_do_not_interpret", [])
    # A pairs drawn from one recording are a retrace within a single
    # file: they share a barometer calibration and one GPS fix. A pairs
    # drawn from two recordings of the same ground share neither. They
    # are different strengths of evidence and are kept apart.
    a_same = [r["score"] for r in scored
              if r["cat"] == "A" and not r["cross_route"]]
    a_cross = [r["score"] for r in scored
               if r["cat"] == "A" and r["cross_route"]]
    thr, fpr, fnr = operating_point(a, d)
    e, e_thr = eer(a, d)
    return {
        "win_m": win_m, "res_m": res_m,
        "n_windows": len(w), "n_routes": len({x["route"] for x in w}),
        "counts": {k: len(v) for k, v in b.items()},
        "n_A_same_recording": len(a_same), "n_A_cross_recording": len(a_cross),
        "auc_AD": auc(a, d), "auc_AD_ci": auc_ci(a, d),
        "auc_BstrictD": auc(b_strict, d),
        "auc_BstrictD_ci": auc_ci(b_strict, d),
        "auc_BlooseD": auc(b_loose, d),
        "auc_BlooseD_ci": auc_ci(b_loose, d),
        "n_B_strict": len(b_strict), "n_B_loose": len(b_loose),
        "auc_BD_pooled_do_not_interpret": auc(bb, d),
        "auc_AD_cross_only": auc(a_cross, d),
        "median": {k: float(np.median(v)) for k, v in b.items() if v},
        "prov_A": provenance(scored, "A"),
        "prov_D": len(provenance(scored, "D")),
        "thr_at_90_recall": thr, "fpr_at_90_recall": fpr,
        "fnr_at_90_recall": fnr, "eer": e, "eer_thr": e_thr,
        "scored": scored,
    }


# ---------------------------------------------------------------------
# Experiment A: shape recovery
#
# Self-comparison, so no other route influences any score, but it reads
# the private real-data corpus and returns n=0 without it.
# ---------------------------------------------------------------------

def _cut(d, e, start, win_m):
    m = (d >= start) & (d <= start + win_m)
    return d[m] - d[m][0], e[m]


def recovery(win_m, res_m=70.0, stride_m=None, seed=0):
    """Self-match error under decimation, offset and both together.

    The target is a window of real terrain. The haystack is the route it
    came from. A perfect matcher returns the same stretch with score 0.

    REQUIRES THE PRIVATE REAL-DATA CORPUS. Windows are cut from cached
    Strava streams carrying GPS, read from
    `~/.strava_segment_matcher_cache/streams`. With no corpus present this
    returns `n = 0` and NaN medians rather than raising, so a caller that
    does not check `n` will read an absent corpus as a result.
    """
    stride_m = stride_m if stride_m is not None else win_m
    rng = np.random.default_rng(seed)
    cfg = MatchConfig(res_m=res_m)
    dec, loc, comb = [], [], []
    for name, d, e, ll in _routes_with_gps():
        if d[-1] < win_m * 2.2:
            continue
        for s in np.arange(d[0], d[-1] - win_m * 1.6, stride_m):
            dd, ee = _cut(d, e, s, win_m)
            if len(dd) < 25:
                continue
            t = prepare_target(dd, ee, cfg)
            if t is None:
                continue
            # decimation: same ground sampled every third point
            ms = match_segment(dd[::3], ee[::3], t, cfg)
            if ms:
                dec.append(ms[0].score)
            # localization: the window must be found inside a longer
            # stretch that contains it plus unrelated terrain
            hs = min(s + win_m * 1.6, d[-1])
            hd, he = _cut(d, e, s - win_m * 0.3 if s > win_m * 0.3 else d[0],
                          hs - (s - win_m * 0.3 if s > win_m * 0.3 else d[0]))
            ms = match_segment(hd, he, t, cfg)
            if ms:
                loc.append(ms[0].score)
            # both, plus barometric-scale noise
            nd, ne = hd[::3], he[::3] + rng.normal(0, 0.5, len(he[::3]))
            ms = match_segment(nd, ne, t, cfg)
            if ms:
                comb.append(ms[0].score)
    med = lambda v: float(np.median(v)) if v else float("nan")
    return {"win_m": win_m, "res_m": res_m, "n": len(dec),
            "decimation": med(dec), "localization": med(loc),
            "combined": med(comb)}
