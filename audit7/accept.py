"""
Acceptance checker for candidate validation data.

Hand this a directory of GPX files or cached streams and it says, per
recording and for the set as a whole, whether the evidence plan in
VALIDATION_CONTRACT.md is satisfied and which criteria are still short.

It exists because "three routes, recorded twice" turned out to be the
wrong requirement when it was checked against arithmetic rather than
asserted. An 8 km route with two recordings yields FOUR category A pairs
at a 6 km window, not the thirty the plan assumed. A requirement nobody
can evaluate mechanically is a requirement that gets asserted rather than
met.

Run:  python -m audit7.accept /path/to/gpx_or_json_dir
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from audit7.independent import haversine_m
from audit7.verify_synthetic import grade_variance_below

# --- thresholds, each traceable to a measurement in FINDINGS.md --------
MIN_TRAILS = 5
MIN_RECORDINGS_PER_TRAIL = 2
MIN_ROUTE_LENGTH_M = 12000.0
TARGET_WINDOW_M = 6000.0
MIN_A_PAIRS_PER_TRAIL = 10
DENSE_SPACING_M = 10.0
MIN_DENSE_TRAILS = 1
MAX_ELEV_STEP_M = 1.0
MIN_RELIEF_M = 100.0
SAME_TRAIL_OVERLAP = 0.70
DISTINCT_TRAIL_OVERLAP = 0.10


def read_gpx(path):
    """Distance, elevation and lat/lon from a GPX track."""
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(path).getroot()
    pts = root.findall(".//g:trkpt", ns) or root.findall(".//trkpt")
    lat, lon, ele = [], [], []
    for p in pts:
        lat.append(float(p.get("lat")))
        lon.append(float(p.get("lon")))
        e = p.find("g:ele", ns)
        if e is None:
            e = p.find("ele")
        ele.append(float(e.text) if e is not None and e.text else math.nan)
    if len(lat) < 16:
        return None
    lat = np.array(lat)
    lon = np.array(lon)
    ele = np.array(ele)
    step = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    d = np.concatenate([[0.0], np.cumsum(step)])
    good = np.isfinite(ele)
    if good.sum() < 16:
        return None
    ele = np.interp(d, d[good], ele[good])
    keep = np.concatenate([[True], np.diff(d) > 1e-6])
    return d[keep], ele[keep], np.column_stack([lat, lon])[keep]


def read_stream_json(path):
    o = json.load(open(path))
    if not o.get("latlng"):
        return None
    d = np.asarray(o["distance"], float)
    e = np.asarray(o["altitude"], float)
    ll = np.asarray(o["latlng"], float)
    n = min(len(d), len(e), len(ll))
    d, e, ll = d[:n], e[:n], ll[:n]
    keep = np.concatenate([[True], np.diff(d) > 1e-6])
    return d[keep], e[keep], ll[keep]


def elevation_step(e):
    """Smallest consistent elevation increment, i.e. the device's
    reporting precision. A stream quantized to 1 m carries a rounding
    staircase that a fine resolution can read as terrain."""
    diffs = np.abs(np.diff(np.asarray(e, float)))
    diffs = diffs[diffs > 1e-9]
    if diffs.size == 0:
        return 0.0
    q = np.round(diffs, 6)
    from math import gcd
    scaled = np.unique((q * 1000).astype(np.int64))
    scaled = scaled[scaled > 0][:400]
    if scaled.size == 0:
        return 0.0
    g = 0
    for v in scaled:
        g = gcd(g, int(v))
    return g / 1000.0


def describe(name, d, e, ll):
    span = float(d[-1] - d[0])
    spacing = float(np.median(np.diff(d)))
    n_pos = max(0, int((span - TARGET_WINDOW_M) // (TARGET_WINDOW_M / 4.0)) + 1) \
        if span >= TARGET_WINDOW_M else 0
    return {
        "name": name, "span_m": span, "spacing_m": spacing,
        "n_points": int(len(d)),
        "elev_step_m": elevation_step(e),
        "relief_m": float(np.max(e) - np.min(e)),
        "has_gps": ll is not None and len(ll) == len(d),
        "window_positions_6km": n_pos,
        "can_yield_6km_window": span >= TARGET_WINDOW_M,
        "dense": spacing <= DENSE_SPACING_M,
        "frac_grade_var_below_60m": grade_variance_below(d, e, 60.0),
        "frac_grade_var_below_120m": grade_variance_below(d, e, 120.0),
        "ll": ll,
    }


def group_into_trails(recs):
    """Cluster recordings by geography. Two recordings of one trail are
    one trail, however many files they occupy: criterion 1 counts
    locations, not activities."""
    trails = []
    for r in recs:
        placed = False
        for t in trails:
            ov = _overlap(r["ll"], t[0]["ll"])
            if ov >= SAME_TRAIL_OVERLAP:
                t.append(r)
                placed = True
                break
        if not placed:
            trails.append([r])
    return trails


def _overlap(A, B, step=5):
    A = np.asarray(A, float)[::step]
    B = np.asarray(B, float)[::step]
    best = []
    for i in range(len(A)):
        best.append(np.min(haversine_m(A[i, 0], A[i, 1], B[:, 0], B[:, 1])))
    return float(np.mean(np.asarray(best) <= 40.0))


# Measured, by truncating the one trail that can currently supply 6 km
# windows and counting the cross-recording category A pairs its two
# recordings produce. Not a fitted curve: an earlier power-law fit
# returned 6 at 12 km where the measurement says 10, which would have
# understated the data requirement.
_MEASURED_YIELD = [(6000.0, 0), (8000.0, 4), (10000.0, 7), (12000.0, 10),
                   (15000.0, 16), (18000.0, 28), (21000.0, 45),
                   (24000.0, 59)]


def expected_A_pairs(span_m):
    """Category A pairs at a 6 km window from two recordings of a route
    of this length, by interpolation on the measured table above.

    Above the measured range the yield is extrapolated linearly from the
    last two points, which is conservative: the real curve is convex.
    """
    if span_m < TARGET_WINDOW_M:
        return 0
    xs = [x for x, _ in _MEASURED_YIELD]
    ys = [y for _, y in _MEASURED_YIELD]
    if span_m >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return int(round(ys[-1] + slope * (span_m - xs[-1])))
    return int(round(float(np.interp(span_m, xs, ys))))


def evaluate(recs):
    trails = group_into_trails(recs)
    report = {"n_recordings": len(recs), "n_trails": len(trails),
              "trails": [], "criteria": {}}
    dense_trails = 0
    trails_meeting_pairs = 0
    for t in trails:
        span = max(r["span_m"] for r in t)
        pairs = expected_A_pairs(span) if len(t) >= 2 else 0
        ok_pairs = pairs >= MIN_A_PAIRS_PER_TRAIL
        # A dense recording only helps the resolution question if it can
        # actually yield the windows that question is asked at. The first
        # version of this counted a 5.2 km route sampled at 6 m as
        # satisfying the dense criterion, when it cannot produce a single
        # 6 km window.
        dense = any(r["dense"] for r in t)
        dense_usable = dense and ok_pairs
        dense_trails += bool(dense_usable)
        trails_meeting_pairs += ok_pairs
        report["trails"].append({
            "recordings": [r["name"] for r in t],
            "n_recordings": len(t), "longest_span_m": span,
            "min_spacing_m": min(r["spacing_m"] for r in t),
            "dense": dense, "dense_usable": dense_usable,
            "expected_A_pairs_6km": pairs,
            "meets_pair_floor": ok_pairs,
            "relief_m": max(r["relief_m"] for r in t),
            "elev_step_m": min(r["elev_step_m"] for r in t),
        })
    c = report["criteria"]
    c["independent_trails"] = (len(trails), MIN_TRAILS, len(trails) >= MIN_TRAILS)
    paired = sum(1 for t in trails if len(t) >= MIN_RECORDINGS_PER_TRAIL)
    c["trails_with_two_recordings"] = (paired, MIN_TRAILS, paired >= MIN_TRAILS)
    c["trails_meeting_pair_floor"] = (trails_meeting_pairs, MIN_TRAILS,
                                      trails_meeting_pairs >= MIN_TRAILS)
    c["dense_trails_usable_at_6km"] = (dense_trails, MIN_DENSE_TRAILS,
                         dense_trails >= MIN_DENSE_TRAILS)
    report["satisfied"] = all(v[2] for v in c.values())
    return report


def load_dir(path):
    recs = []
    for p in sorted(Path(path).iterdir()):
        try:
            if p.suffix.lower() == ".gpx":
                got = read_gpx(p)
            elif p.suffix.lower() == ".json":
                got = read_stream_json(p)
            else:
                continue
        except Exception as exc:                       # noqa: BLE001
            print("  skipped %s: %s" % (p.name, exc))
            continue
        if got is None:
            print("  skipped %s: unusable or no GPS" % p.name)
            continue
        recs.append(describe(p.stem, *got))
    return recs


def main(path):
    recs = load_dir(path)
    if not recs:
        print("no usable recordings found in %s" % path)
        return 1
    print("%-22s %9s %8s %9s %8s %8s %7s" % (
        "recording", "span_m", "spacing", "elev_step", "relief", "6km?", "dense"))
    for r in recs:
        print("%-22s %9.0f %8.1f %9.2f %8.0f %8s %7s" % (
            r["name"][:22], r["span_m"], r["spacing_m"], r["elev_step_m"],
            r["relief_m"], "yes" if r["can_yield_6km_window"] else "NO",
            "yes" if r["dense"] else "no"))
    rep = evaluate(recs)
    print("\n%d recordings group into %d independent trails\n"
          % (rep["n_recordings"], rep["n_trails"]))
    for t in rep["trails"]:
        print("  trail: %s" % ", ".join(t["recordings"]))
        print("     recordings %d  longest %.0f m  min spacing %.1f m  "
              "relief %.0f m  expected 6 km A pairs %d  %s"
              % (t["n_recordings"], t["longest_span_m"], t["min_spacing_m"],
                 t["relief_m"], t["expected_A_pairs_6km"],
                 "OK" if t["meets_pair_floor"] else "SHORT"))
    print("\ncriteria:")
    for k, (got, need, ok) in rep["criteria"].items():
        print("  %-28s %3d / %-3d  %s" % (k, got, need, "PASS" if ok else "FAIL"))
    print("\n%s" % ("EVIDENCE PLAN SATISFIED" if rep["satisfied"]
                    else "EVIDENCE PLAN NOT YET SATISFIED"))
    return 0 if rep["satisfied"] else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else str(Path.home() / ".strava_segment_matcher_cache"
                           / "streams")))
