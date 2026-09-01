"""
Agreement between the two instruments.

The point of writing a second instrument is wasted if it is never
compared with the first. Where they agree, a shared defect would have had
to be written twice in two different forms. Where they disagree, one of
them is wrong and the disagreement is the finding.

Nothing here concludes anything about the matcher. It concludes things
about the measuring apparatus.
"""

import numpy as np

from audit7.independent import (auc_lower_is_better, bootstrap_ci,
                                haversine_m, min_distance_to_set,
                                overlap_fraction, grade_percent,
                                resample_uniform, vertical_change_m)


def check_auc_against_broadcast(seed=0, trials=200):
    """The longhand loop and the broadcast form must agree exactly."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        p = rng.normal(0, 1, rng.integers(2, 30)).tolist()
        n = rng.normal(1, 1, rng.integers(2, 30)).tolist()
        slow = auc_lower_is_better(p, n)
        a = np.asarray(p)
        b = np.asarray(n)
        fast = float(((a[:, None] < b[None, :]).sum()
                      + 0.5 * (a[:, None] == b[None, :]).sum())
                     / (len(a) * len(b)))
        worst = max(worst, abs(slow - fast))
    return worst


def check_haversine_against_known():
    """Two fixed points with a distance known independently.

    Boulder to Denver, roughly. The tolerance is loose because the point
    is to catch a factor-of-two or a radians/degrees error, not to
    validate the ellipsoid.
    """
    d = float(haversine_m(40.0150, -105.2705, 39.7392, -104.9903))
    return d


def check_overlap_is_symmetric(seed=0, trials=100):
    """overlap_fraction must not depend on argument order."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        A = np.column_stack([40.0 + rng.normal(0, 1e-3, 30),
                             -105.3 + rng.normal(0, 1e-3, 30)])
        B = np.column_stack([40.0 + rng.normal(0, 1e-3, 30),
                             -105.3 + rng.normal(0, 1e-3, 30)])
        a, sa, _ = overlap_fraction(A, B)
        b, sb, _ = overlap_fraction(B, A)
        worst = max(worst, abs(a - b), abs(sa - sb))
    return worst


def check_grade_units():
    """A 5 percent ramp must read as 5, not 0.05 and not 500.

    This is the exact error that produced a saturated score of 793 in the
    previous audit and was briefly taken for a resolution measurement.
    """
    d = np.arange(0.0, 1000.0, 5.0)
    e = 1500.0 + 0.05 * d
    grid, elev = resample_uniform(d, e, 10.0)
    g = grade_percent(grid, elev)
    return float(np.median(g))


def check_vertical_change_is_resolution_independent():
    """Gain measured on the same hill sampled two ways must agree.

    A gain that depends on sampling density silently rewards densely
    recorded routes, which are not randomly distributed across the
    corpus.
    """
    x = np.arange(0.0, 2000.0, 1.0)
    e = 1500.0 + 60.0 * np.sin(x / 300.0)
    fine = vertical_change_m(x, e)
    coarse = vertical_change_m(x[::17], e[::17])
    return fine, coarse


def check_bench_and_audit_geometry_agree(sample=400, seed=0):
    """The production-side bench geometry and this one must give the same
    overlap on real windows.

    The bench path uses an equirectangular projection with a fixed origin
    and a bounding-box prefilter. This one uses haversine with no
    prefilter. They were written independently and one of them previously
    had a projection bug that survived 8534 pairs undetected.
    """
    from audit7.corpus import build_windows, load_streams
    import bench.semantic as S
    ws = build_windows(load_streams(), 1000.0, 500.0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ws), min(sample, len(ws) * (len(ws) - 1) // 2),
                     replace=True)
    worst_ov, worst_sep = 0.0, 0.0
    n = 0
    for i in idx:
        j = int(rng.integers(0, len(ws)))
        if j == i:
            continue
        ov_a, sep_a, _ = overlap_fraction(ws[i]["ll"], ws[j]["ll"])
        ov_b, sep_b = S.geo_relation(ws[i]["ll"], ws[j]["ll"])
        worst_ov = max(worst_ov, abs(ov_a - ov_b))
        worst_sep = max(worst_sep, abs(sep_a - sep_b))
        n += 1
    return worst_ov, worst_sep, n


def check_archetype_labels_agree(win_m=1000.0, stride_m=500.0):
    """Independent archetype classifier vs the bench one.

    They use different grade estimators, so exact agreement is not
    expected. The disagreement rate is the number that matters: if it is
    large, every category B result rests on a coin flip.
    """
    from audit7.corpus import build_windows, load_streams
    import bench.semantic as S
    ws = build_windows(load_streams(), win_m, stride_m)
    agree = 0
    disagree = []
    for w in ws:
        other, feats = S.phase_signature(w["d"], w["e"])
        if other is None:
            continue
        if other == w["label"]:
            agree += 1
        else:
            disagree.append((w["route"], round(w["start"]), w["label"],
                             other, round(w["steep"], 2)))
    return agree, disagree
