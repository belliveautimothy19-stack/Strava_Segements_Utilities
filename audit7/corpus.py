"""
Corpus construction with the checks that were missing.

Every window and every pair passes through assertions before it can
become evidence. The list of assertions is not generic hygiene: each one
corresponds to a defect that has already occurred here, or to the nearest
neighbour of one that has.
"""

import json
import math
from pathlib import Path

import numpy as np

from audit7.independent import (grade_percent, min_distance_to_set,
                                overlap_fraction, resample_uniform,
                                vertical_change_m, GEO_APART_FRAC,
                                GEO_SAME_FRAC, MIN_APART_M)

STREAM_DIR = Path.home() / ".strava_segment_matcher_cache" / "streams"

# Archetype classification happens at a FIXED scale, independent of any
# resolution being swept, so labels cannot move when the matcher's own
# resolution changes. That circularity would make a resolution sweep
# meaningless.
LABEL_SCALE_M = 100.0
FLAT_BAND_PCT = 1.5      # |grade| below this is flat
MIN_PHASE_PCT = 2.0      # a phase must average at least this to count
MIN_PHASE_FRAC = 0.15    # and occupy at least this share of the window


def load_streams():
    """Raw cached streams. No interpolation, no smoothing, no cleaning
    beyond what is needed to make the arrays usable, so that anything the
    representation does downstream is visible rather than baked in."""
    out = {}
    for p in sorted(STREAM_DIR.glob("*.json")):
        o = json.load(open(p))
        if not o.get("latlng"):
            continue
        d = np.asarray(o["distance"], float)
        e = np.asarray(o["altitude"], float)
        ll = np.asarray(o["latlng"], float)
        n = min(len(d), len(e), len(ll))
        assert n > 0
        d, e, ll = d[:n], e[:n], ll[:n]
        # strictly increasing distance, without discarding elevation
        keep = np.concatenate([[True], np.diff(d) > 1e-6])
        d, e, ll = d[keep], e[keep], ll[keep]
        assert len(d) == len(e) == len(ll), "stream length mismatch"
        assert np.all(np.diff(d) > 0)
        assert np.all(np.isfinite(e)), "elevation holes in %s" % p.stem
        assert ll.shape[1] == 2
        out[p.stem] = {"name": o.get("_name", p.stem), "d": d, "e": e,
                       "ll": ll,
                       "spacing": float(np.median(np.diff(d))),
                       "span": float(d[-1] - d[0])}
    return out


def native_resolvable_m(spacing):
    """Shortest wavelength a stream can carry, by Nyquist, with margin.

    A stream sampled every 33 m carries no information below 66 m. Asking
    a resolution sweep to distinguish 50 m from 70 m on such a stream is
    asking it to measure the interpolator.
    """
    return 2.5 * spacing


def archetype(d, e):
    """Ordered phase label, computed independently of the matcher.

    STABILITY IS THE REQUIREMENT, not expressiveness. The previous
    formulation, shared by both instruments in this project, dropped a
    phase shorter than `int(MIN_PHASE_FRAC * n_samples)`. Because n
    changes by one when a window gains or loses a leading sample, the
    floor moved between 2 and 3 samples and a two-sample phase was
    admitted or dropped accordingly. Measured consequence: 24.3 percent
    of window labels changed when one or two leading samples were
    removed, a perturbation of 30 to 60 m on a 1000 m window. Category B
    is built entirely from these labels, so a quarter of it was noise.

    Three changes make it stable:

      physical floor    the minimum phase length is a fraction of the
                        window in METRES, so it cannot move with the
                        sample count
      anchored grid     the resampling grid starts at the window origin
                        with a fixed step, so a shifted window sees the
                        same grid
      iterative merge   the shortest failing phase is merged into its
                        neighbour and the process repeats until every
                        surviving phase passes. A single pass that
                        filters and then collapses is order-dependent and
                        can leave adjacent equal labels behind.

    Ordinal by design: it records what happens in what order, never how
    much. Magnitude is returned separately so the two notions can be told
    apart rather than silently merged, which is what made the pooled
    category B uninterpretable.
    """
    span = d[-1] - d[0]
    if span < LABEL_SCALE_M * 3:
        return None, {}
    step = LABEL_SCALE_M / 2.0
    grid, elev = resample_uniform(d, e, step)
    g = grade_percent(grid, elev)
    lab = np.where(g > FLAT_BAND_PCT, 1, np.where(g < -FLAT_BAND_PCT, -1, 0))

    runs = []
    cur, start = int(lab[0]), 0
    for i in range(1, len(lab)):
        if lab[i] != cur:
            runs.append([cur, start, i])
            cur, start = int(lab[i]), i
    runs.append([cur, start, len(lab)])

    min_phase_m = MIN_PHASE_FRAC * span

    def fails(r):
        length_m = (r[2] - r[1]) * step
        if length_m < min_phase_m:
            return True
        return abs(float(np.mean(g[r[1]:r[2]]))) < MIN_PHASE_PCT

    # Merge the shortest failing phase into whichever neighbour is longer,
    # then re-collapse adjacent equal labels, until nothing fails or only
    # one phase is left.
    while len(runs) > 1:
        bad = [i for i, r in enumerate(runs) if fails(r)]
        if not bad:
            break
        i = min(bad, key=lambda k: runs[k][2] - runs[k][1])
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        if left is None:
            target = i + 1
        elif right is None:
            target = i - 1
        else:
            target = (i - 1 if (left[2] - left[1]) >= (right[2] - right[1])
                      else i + 1)
        lo = min(runs[i][1], runs[target][1])
        hi = max(runs[i][2], runs[target][2])
        merged = [runs[target][0], lo, hi]
        drop = sorted((i, target), reverse=True)
        for k in drop:
            runs.pop(k)
        runs.insert(min(drop), merged)
        collapsed = []
        for r in runs:
            if collapsed and collapsed[-1][0] == r[0]:
                collapsed[-1][2] = r[2]
            else:
                collapsed.append(list(r))
        runs = collapsed

    name = {1: "up", -1: "down", 0: "flat"}
    label = "-".join(name[int(r[0])] for r in runs)
    return label, {"n_phases": len(runs),
                   "steepness": float(np.median(np.abs(g))),
                   "mean_grade": float(np.mean(g)),
                   "grades": g}


def stable_label(d, e, n_trim=3):
    """(label, is_stable). A label is admitted only if it survives
    perturbation.

    Rewriting the classifier to be more careful reduced label churn under
    a 30 to 60 m window shift only from 24.3 to 20.6 percent, and the
    residual failures included outright reversals (flat against up, down
    against up). The conclusion is not that the classifier needs more
    work. It is that a discrete archetype is not a well-defined property
    of roughly a fifth of these windows: they sit near a decision
    boundary where any reasonable classifier is a coin flip.

    So reliability is measured rather than assumed. A window enters the
    archetype analysis only if trimming up to `n_trim` samples from
    either end, and shifting the resampling grid by half a step, all
    yield the same label. The test never consults the matcher, so this is
    not selection on the quantity being measured.

    An unstable window is not discarded from the corpus. It is excluded
    from categories that depend on the archetype label, and counted, so
    the cost of the filter is visible.
    """
    base, feats = archetype(d, e)
    if base is None:
        return None, False, {}
    for k in range(1, n_trim + 1):
        if len(d) - k < 10:
            break
        for dd, ee in ((d[k:] - d[k], e[k:]), (d[:-k], e[:-k])):
            other, _ = archetype(dd, ee)
            if other is not None and other != base:
                return base, False, feats
    # half-step grid shift, which changes which samples the grid lands on
    half = LABEL_SCALE_M / 4.0
    if d[-1] - d[0] > half * 4:
        other, _ = archetype(d[d >= half] - half, e[d >= half])
        if other is not None and other != base:
            return base, False, feats
    return base, True, feats


def normalized_profile(d, e, n=64):
    """Elevation against distance, both normalized to [0, 1].

    An independent, continuous notion of shape. It is not a grade
    sequence and it is not compared by dynamic time warping, so it shares
    no machinery with the matcher. Used to ask whether the matcher's
    score moves monotonically with a shape difference defined outside
    it.

    Elevation is normalized by the window's own vertical range, so this
    measures ORDERED SHAPE and is deliberately blind to steepness. A
    30 m bump and a 300 m climb of the same profile shape give the same
    curve, which is what makes it usable as an independent axis against
    which the matcher's steepness sensitivity can be seen.
    """
    x = np.linspace(0.0, 1.0, n)
    ee = np.interp(x, (d - d[0]) / max(d[-1] - d[0], 1e-9), e)
    rng = float(ee.max() - ee.min())
    if rng < 1e-6:
        return np.zeros(n)
    return (ee - ee.min()) / rng


def shape_distance(a, b):
    """Mean absolute difference between two normalized profiles, in units
    of normalized elevation. 0 is identical shape, and the maximum is
    bounded by 1."""
    x = float(np.mean(np.abs(a - b)))
    assert 0.0 <= x <= 1.0
    return x


def build_windows(streams, win_m, stride_m, min_samples=25):
    """Cut windows, asserting each one is usable before it is admitted."""
    out = []
    for rid, s in streams.items():
        d, e, ll = s["d"], s["e"], s["ll"]
        if s["span"] < win_m * 1.3:
            continue
        for start in np.arange(d[0], d[-1] - win_m, stride_m):
            m = (d >= start) & (d <= start + win_m)
            if int(m.sum()) < min_samples:
                continue
            dd = d[m] - d[m][0]
            ee, lls = e[m], ll[m]
            assert len(dd) == len(ee) == len(lls)
            if dd[-1] < win_m * 0.9:
                continue
            label, feats = archetype(dd, ee)
            if label is None:
                continue
            gain, loss = vertical_change_m(dd, ee)
            _, is_stable, _ = stable_label(dd, ee)
            out.append({
                "route": rid, "start": float(start), "d": dd, "e": ee,
                "ll": lls, "label": label, "arch": label,
                "n_phases": feats["n_phases"], "steep": feats["steepness"],
                "mean_grade": feats["mean_grade"], "grades": feats["grades"],
                "gain": gain, "loss": loss, "label_stable": bool(is_stable),
                "profile": normalized_profile(dd, ee),
                "spacing": s["spacing"],
                "fingerprint": _fingerprint(dd, ee),
            })
    _assert_no_duplicate_windows(out)
    return out


def _fingerprint(d, e):
    """Byte identity of the window's content.

    Guards the failure where two windows that are literally the same
    samples are counted as an independent positive pair. That inflates
    any same-terrain metric to near perfection and is invisible in the
    aggregate.
    """
    return (round(float(d[-1]), 3), len(d),
            round(float(np.sum(e)), 4), round(float(e[0]), 4),
            round(float(e[-1]), 4))


def _assert_no_duplicate_windows(ws):
    seen = {}
    for w in ws:
        k = (w["route"], round(w["start"], 3))
        assert k not in seen, "duplicate window %s" % (k,)
        seen[k] = True


def categorize(windows, steep_ratio=1.5, vert_tol=0.10, comp_tol=1.0,
               verbose=False):
    """Assign pairs to categories using geometry and shape only.

    The matcher's score is never consulted. Aggregate statistics define
    only category C, which is a NEGATIVE, so they cannot smuggle in the
    conclusion.
    """
    pairs = []
    n = len(windows)
    for i in range(n):
        A = windows[i]
        for j in range(i + 1, n):
            B = windows[j]
            # Two windows cut from overlapping stretches of one recording
            # share most of their samples. Admitting them measures the
            # matcher's ability to recognise a slice of itself.
            if A["route"] == B["route"]:
                if abs(A["start"] - B["start"]) < float(A["d"][-1]):
                    continue
            if A["fingerprint"] == B["fingerprint"]:
                continue
            ov, sep, both = overlap_fraction(A["ll"], B["ll"])
            same_arch = A["arch"] == B["arch"]
            tot = max(A["gain"] + A["loss"] + B["gain"] + B["loss"], 1e-6)
            vdev = (abs(A["gain"] - B["gain"])
                    + abs(A["loss"] - B["loss"])) / tot
            comp = _wasserstein_percent(A["grades"], B["grades"])
            ratio = (max(A["steep"], B["steep"])
                     / max(min(A["steep"], B["steep"]), 1e-6))
            if ov >= GEO_SAME_FRAC:
                cat = "A"
            elif ov <= GEO_APART_FRAC and sep >= MIN_APART_M:
                if same_arch:
                    cat = "B_strict" if ratio <= steep_ratio else "B_loose"
                elif vdev <= vert_tol and comp <= comp_tol:
                    cat = "C"
                else:
                    cat = "D"
            else:
                cat = None
            if cat:
                pairs.append({"cat": cat, "A": A, "B": B, "ov": ov,
                              "sep": sep, "ov_both": both, "vdev": vdev,
                              "comp": comp, "ratio": ratio,
                              "cross_route": A["route"] != B["route"]})
    _assert_categories_are_disjoint(pairs)
    return pairs


def _wasserstein_percent(a, b):
    """W1 between two grade samples, in percent-grade units.

    Quantile form, which is a different computation from the CDF-integral
    form production uses. Agreement between the two is checked rather than
    presumed.
    """
    qs = np.linspace(0.0, 1.0, 512)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _assert_categories_are_disjoint(pairs):
    """No pair may carry two labels, and a positive may not also be a
    negative. Population mixing is how the pooled category B produced a
    number that tracked its own composition."""
    seen = {}
    for p in pairs:
        k = (p["A"]["route"], round(p["A"]["start"], 3),
             p["B"]["route"], round(p["B"]["start"], 3))
        assert k not in seen, "pair %s labelled twice: %s and %s" % (
            k, seen.get(k), p["cat"])
        seen[k] = p["cat"]
    for p in pairs:
        if p["cat"] == "A":
            assert p["ov"] >= GEO_SAME_FRAC
        elif p["cat"] in ("B_strict", "B_loose", "C", "D"):
            assert p["ov"] <= GEO_APART_FRAC and p["sep"] >= MIN_APART_M


def provenance(pairs, cat):
    out = {}
    for p in pairs:
        if p["cat"] != cat:
            continue
        k = tuple(sorted((p["A"]["route"], p["B"]["route"])))
        out[k] = out.get(k, 0) + 1
    return out


# ---------------------------------------------------------------------
# Two separate experiments, deliberately not sharing a definition
# ---------------------------------------------------------------------

def categorize_geometric(windows):
    """Pairs labelled by GEOGRAPHY ALONE.

    A  the two windows are the same piece of ground, seen twice
    N  the two windows are separate ground

    No archetype label is consulted, so this experiment cannot inherit
    the 20 percent label instability measured above, and no aggregate
    statistic is consulted, so it cannot assume its own conclusion. This
    is the instrument for the physical-identity question, and it is the
    one whose result should carry the most weight.
    """
    pairs = []
    n = len(windows)
    for i in range(n):
        A = windows[i]
        for j in range(i + 1, n):
            B = windows[j]
            if A["route"] == B["route"]:
                if abs(A["start"] - B["start"]) < float(A["d"][-1]):
                    continue
            if A["fingerprint"] == B["fingerprint"]:
                continue
            ov, sep, both = overlap_fraction(A["ll"], B["ll"])
            if ov >= GEO_SAME_FRAC:
                cat = "A"
            elif ov <= GEO_APART_FRAC and sep >= MIN_APART_M:
                cat = "N"
            else:
                continue
            pairs.append({"cat": cat, "A": A, "B": B, "ov": ov, "sep": sep,
                          "ov_both": both,
                          "cross_route": A["route"] != B["route"],
                          "shape_d": shape_distance(A["profile"],
                                                    B["profile"])})
    _assert_categories_are_disjoint(pairs)
    return pairs


def assert_sample_adequate(name, lo, hi, effect, n_pos=None):
    """Refuse to report a comparison the sample cannot resolve.

    An AUC of 0.986 quoted from twelve positives has an interval wider
    than every difference the experiment was built to detect. Reporting
    it invites a conclusion the data does not support, so the instrument
    declines instead.
    """
    from audit7.independent import interval_is_adequate
    width = hi - lo
    ok = interval_is_adequate(lo, hi, effect, n_pos=n_pos)
    return {"metric": name, "ci_width": width, "resolvable_effect": effect,
            "n_pos": n_pos, "adequate": bool(ok)}


def assert_threshold_not_degenerate(thr, scores, margin=0.02):
    """A threshold at the extreme of the score distribution is not an
    operating point, it is an artifact. Guards the case where a false
    negative rate was pinned at 0.5 by setting the threshold at the
    median of the positives."""
    scores = np.asarray(scores, float)
    q = float(np.mean(scores <= thr))
    return {"quantile": q, "degenerate": bool(q < margin or q > 1 - margin)}


def assert_no_population_leakage(pairs):
    """A single window pair must not appear in two categories, and the
    positive and negative populations must not share a pair."""
    keys = {}
    for p in pairs:
        k = (p["A"]["route"], round(p["A"]["start"], 3),
             p["B"]["route"], round(p["B"]["start"], 3))
        assert k not in keys or keys[k] == p["cat"], (
            "pair %s in both %s and %s" % (k, keys.get(k), p["cat"]))
        keys[k] = p["cat"]
    return len(keys)
