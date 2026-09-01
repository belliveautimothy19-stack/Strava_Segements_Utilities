"""
Does the matcher capture "same kind of hill", or only "same hill"?

Every experiment before this one measured self-consistency (the same
terrain re-measured) or separation (unrelated terrain rejected). Neither
answers the question a user actually asks the tool: find me a DIFFERENT
hill that is like this one.

METHOD

Windows are cut from real routes and sorted into four categories. The
category boundaries come from two sources that are independent of the
matcher and independent of each other:

  geography   GPS traces decide whether two windows are the same piece of
              ground. This is what separates "same hill" from "different
              hill", and it cannot be faked by statistics.

  structure   An ordered-shape classifier decides the terrain archetype:
              the sequence of climb, descent and flat phases, with short
              phases merged away. It reads ORDER, not magnitude, and it
              is computed at a fixed 100 m reference scale so it does not
              move when the matcher's own resolution is swept.

  A  same physical terrain          geo overlap >= 0.7, and the two
                                    windows do NOT overlap along the
                                    route, so this is a genuine second
                                    pass over the same ground (an
                                    out-and-back retrace), not a slice of
                                    itself
  B  same archetype, different hill geo distinct, same phase signature
  C  statistics matched only        geo distinct, different signature,
                                    but matched gain/loss/composition
  D  clearly different              geo distinct, different signature,
                                    unmatched statistics

B is the category that matters. If B does not sit closer to A than to D,
the matcher does not encode terrain similarity, only terrain identity.

WHY AGGREGATE STATISTICS DO NOT DEFINE THE POSITIVE

Matching gain, length and grade distribution was already shown to be
insufficient, so using it to label positives would assume the conclusion.
It appears here only as the definition of category C, which is a
NEGATIVE, and as reported diagnostics.

One caveat is stated rather than hidden: a "strict" variant of B also
requires comparable characteristic steepness, because a 3 percent and a
9 percent sustained climb share a phase signature but no runner calls
them the same hill. Steepness is an aggregate quantity, so B_strict is
partly statistically defined. Both variants are reported.
"""

import itertools
import math

import numpy as np

from segmatch.match import MatchConfig, prepare_target, match_segment
from segmatch.profile import build_profile, vertical_change
from segmatch.distance import wasserstein1
from bench.real_data import load_cache

# Grade is classified at this fixed scale regardless of the res_m being
# swept, so the archetype labels stay constant across the sweep and the
# comparison is not circular.
REF_RES_M = 100.0
UP_PCT = 1.5           # grade above this counts as climbing
MIN_PHASE_FRAC = 0.15  # phases shorter than this are merged away
MIN_PHASE_GRADE = 2.0  # a phase must average at least this to be a phase
SAME_GROUND_M = 40.0   # GPS points closer than this are the same ground
GEO_SAME = 0.70        # overlap fraction to call two windows one hill
GEO_DISTINCT = 0.10    # overlap fraction below which they are separate
MIN_SEPARATION_M = 150.0


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _pair_distances(A, B):
    """Min distance from each point of A to the set B, in metres.

    A local equirectangular approximation is enough at these separations
    and avoids an O(n*m) haversine.
    """
    lat0 = float(np.mean(A[:, 0]))
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    ax = A[:, 1] * kx
    ay = A[:, 0] * ky
    bx = B[:, 1] * kx
    by = B[:, 0] * ky
    d2 = (ax[:, None] - bx[None, :]) ** 2 + (ay[:, None] - by[None, :]) ** 2
    return np.sqrt(d2.min(axis=1))


def geo_relation(A, B):
    """(overlap_frac, min_separation_m) between two GPS traces."""
    da = _pair_distances(A, B)
    db = _pair_distances(B, A)
    overlap = max(float(np.mean(da <= SAME_GROUND_M)),
                  float(np.mean(db <= SAME_GROUND_M)))
    return overlap, float(min(da.min(), db.min()))


def phase_signature(dist, elev, ref_res_m=REF_RES_M):
    """Ordered sequence of climb/descent/flat phases.

    Deliberately ordinal: it records WHAT HAPPENS IN WHAT ORDER and
    nothing about how much. Two climbs of 3 and 9 percent produce the
    same signature, which is the point; magnitude is reported separately
    so the two notions can be told apart in the analysis.
    """
    p = build_profile(dist, elev, ref_res_m, 8)
    if p is None or p.length < ref_res_m * 2:
        return None, {}
    n = max(8, int(p.length / (ref_res_m / 2)))
    g = p.grade_at(np.linspace(0, p.length, n))
    lab = np.where(g > UP_PCT, 1, np.where(g < -UP_PCT, -1, 0))
    # merge runs, dropping any phase shorter than MIN_PHASE_FRAC
    runs = []
    cur, start = lab[0], 0
    for i in range(1, len(lab)):
        if lab[i] != cur:
            runs.append([cur, start, i])
            cur, start = lab[i], i
    runs.append([cur, start, len(lab)])
    minlen = max(1, int(MIN_PHASE_FRAC * len(lab)))
    # A phase must be both long enough AND a real climb or descent. The
    # length test alone is not sufficient: on gentle terrain a wobble of a
    # degree either side of a 4 percent mean produces long "up" and "down"
    # runs and the window gets labelled rolling, when a runner would call
    # it a steady gentle climb. Measured consequence, before this test was
    # added: 14 of 16 supposedly different-archetype pairs were rolling
    # against sustained at 3 to 5 percent, which is a threshold artifact
    # rather than a difference in kind.
    keep = [r for r in runs
            if r[2] - r[1] >= minlen
            and abs(float(np.mean(g[r[1]:r[2]]))) >= MIN_PHASE_GRADE]
    if not keep:
        keep = [max(runs, key=lambda r: r[2] - r[1])]
    # collapse adjacent equal labels left behind by the drop
    sig = []
    for r in keep:
        if not sig or sig[-1] != r[0]:
            sig.append(int(r[0]))
    name = {1: "up", -1: "down", 0: "flat"}
    label = "-".join(name[s] for s in sig)
    feats = {"steepness": float(np.median(np.abs(g))),
             "mean_grade": float(np.mean(g)),
             "n_phases": len(sig), "grades": g}
    return label, feats


def archetype_of(label, n_phases):
    if label in ("up", "down", "flat"):
        return {"up": "sustained_climb", "down": "sustained_descent",
                "flat": "flat"}[label]
    if n_phases >= 3:
        return "rolling"
    return {"up-down": "climb_then_descent",
            "down-up": "descent_then_climb",
            "up-flat": "climb_then_flat",
            "flat-up": "flat_then_climb",
            "down-flat": "descent_then_flat",
            "flat-down": "flat_then_descent"}.get(label, "mixed")


def build_windows(win_m=1000.0, stride_m=250.0, min_points=25):
    """Cut windows from every cached route that carries GPS."""
    out = []
    for name, d, e, ll in _routes_with_gps():
        total = d[-1] - d[0]
        if total < win_m * 1.3:
            continue
        for s in np.arange(d[0], d[-1] - win_m, stride_m):
            m = (d >= s) & (d <= s + win_m)
            if m.sum() < min_points:
                continue
            dd = d[m] - d[m][0]
            ee = e[m]
            label, feats = phase_signature(dd, ee)
            if label is None:
                continue
            g, l = vertical_change(dd, ee)
            out.append({"route": name, "start": float(s), "d": dd, "e": ee,
                        "ll": ll[m], "label": label,
                        "arch": archetype_of(label, feats["n_phases"]),
                        "steep": feats["steepness"],
                        "mean_grade": feats["mean_grade"],
                        "grades": feats["grades"], "gain": g, "loss": l})
    return out


def _routes_with_gps():
    import json
    from pathlib import Path
    base = Path.home() / ".strava_segment_matcher_cache" / "streams"
    out = []
    for p in sorted(base.glob("*.json")):
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if not o.get("latlng"):
            continue
        d = np.asarray(o["distance"], float)
        e = np.asarray(o["altitude"], float)
        ll = np.asarray(o["latlng"], float)
        n = min(len(d), len(e), len(ll))
        out.append((p.stem, d[:n], e[:n], ll[:n]))
    return out


def categorize(windows, steep_ratio=1.5, vert_tol=0.10, comp_tol=1.0):
    """Assign every window pair to A, B, C, D or unclassified."""
    pairs = []
    for A, B in itertools.combinations(windows, 2):
        # Reject trivial shifted copies outright. Two windows cut from
        # overlapping stretches of the SAME route share most of their
        # samples, so calling them a positive would measure nothing but
        # the matcher's ability to recognise a slice of itself. An earlier
        # version of this function admitted them and category A filled up
        # with pairs 250 m apart on one route sharing 750 m of ground.
        if A["route"] == B["route"]:
            span = abs(A["start"] - B["start"])
            win = float(A["d"][-1])
            if span < win:
                continue
        ov, sep = geo_relation(A["ll"], B["ll"])
        same_arch = A["arch"] == B["arch"]
        tot = max(A["gain"] + A["loss"] + B["gain"] + B["loss"], 1e-6)
        vdev = (abs(A["gain"] - B["gain"]) + abs(A["loss"] - B["loss"])) / tot
        comp = wasserstein1(A["grades"], B["grades"])
        ratio = (max(A["steep"], B["steep"])
                 / max(min(A["steep"], B["steep"]), 1e-6))
        if ov >= GEO_SAME:
            cat = "A"
        elif ov <= GEO_DISTINCT and sep >= MIN_SEPARATION_M:
            if same_arch:
                cat = "B_strict" if ratio <= steep_ratio else "B_loose"
            elif vdev <= vert_tol and comp <= comp_tol:
                cat = "C"
            else:
                cat = "D"
        else:
            cat = None
        if cat:
            pairs.append({"cat": cat, "A": A, "B": B, "ov": ov, "sep": sep,
                          "vdev": vdev, "comp": comp, "ratio": ratio,
                          "cross_route": A["route"] != B["route"]})
    return pairs


def score_pairs(pairs, res_m, cap=None):
    """Matcher score for each pair at one resolution."""
    cfg = MatchConfig(res_m=res_m)
    out = []
    for pr in (pairs[:cap] if cap else pairs):
        t = prepare_target(pr["A"]["d"], pr["A"]["e"], cfg)
        if t is None:
            continue
        ms = match_segment(pr["B"]["d"], pr["B"]["e"], t, cfg)
        if not ms:
            continue
        rec = dict(pr)
        rec["score"] = ms[0].score
        rec["shape"] = ms[0].shape
        rec["dist"] = ms[0].dist
        rec["gain_dev"] = ms[0].gain_dev
        rec["direction"] = ms[0].direction
        out.append(rec)
    return out
