"""Regression tests for the defects found in the length audit.

Three of the four were defects in the measuring instruments rather than
in the matcher, which is the reason they are pinned here: a broken
instrument reports a clean result and nothing fails.
"""

import itertools
import math

import numpy as np
import pytest

import bench.semantic as S
from bench.length import (_subsample, auc, auc_ci, eer, operating_point,
                          recovery)
from bench.synth import gain_matched_staircase, uniform_climb
from tests.corpus_guard import requires_corpus
from segmatch.match import MatchConfig, prepare_target, match_segment
from segmatch.profile import vertical_change


def _trace(lat0, lon0, n=40, dlat=0.0, dlon=0.001):
    return np.array([[lat0 + i * dlat, lon0 + i * dlon] for i in range(n)],
                    float)


def test_bbox_gap_is_a_true_lower_bound():
    """The bounding-box gap must never exceed the real separation.

    The first version derived the longitude scale from each window's own
    mean latitude, which put every window in a different frame. Because
    longitude is multiplied by that scale before differencing, a relative
    change of 1e-6 in the factor moved a projected coordinate by metres,
    and the bound was violated on 7708 of 8534 pairs. A bound that is too
    large silently reclassifies overlapping ground as separate.
    """
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = _trace(39.9 + rng.uniform(0, 0.2), -105.5 + rng.uniform(0, 0.2),
                   dlat=rng.uniform(-2e-4, 2e-4))
        b = _trace(39.9 + rng.uniform(0, 0.2), -105.5 + rng.uniform(0, 0.2),
                   dlat=rng.uniform(-2e-4, 2e-4))
        gap = S._bbox_gap(S._bbox(a), S._bbox(b))
        _, sep = S.geo_relation(a, b)
        assert gap <= sep + 1e-6, (gap, sep)


def test_bbox_skip_never_changes_a_classification():
    """Skipping the exact computation must be inert, not just fast."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = _trace(39.9 + rng.uniform(0, 0.3), -105.5 + rng.uniform(0, 0.3))
        b = _trace(39.9 + rng.uniform(0, 0.3), -105.5 + rng.uniform(0, 0.3))
        if S._bbox_gap(S._bbox(a), S._bbox(b)) < S.MIN_SEPARATION_M:
            continue
        ov, _ = S.geo_relation(a, b)
        assert ov == 0.0


def test_operating_point_does_not_fix_the_error_rate_by_construction():
    """A threshold set at the median of the positives makes the false
    negative rate 0.5 whatever the matcher does. That number was reported
    as an accuracy figure and read as flat across window length, when it
    was flat because it could not be anything else.
    """
    rng = np.random.default_rng(2)
    pos = rng.normal(0.0, 1.0, 500).tolist()
    neg = rng.normal(4.0, 1.0, 500).tolist()
    _, fpr_good, fnr_good = operating_point(pos, neg, recall=0.90)
    assert fnr_good == pytest.approx(0.10, abs=0.02)
    # a genuinely worse separation must cost a higher false positive rate
    neg_close = rng.normal(1.0, 1.0, 500).tolist()
    _, fpr_bad, _ = operating_point(pos, neg_close, recall=0.90)
    assert fpr_bad > fpr_good


def test_auc_and_eer_agree_on_direction():
    """Lower score means more similar, so a good positive set must give
    AUC above 0.5 and a low equal error rate."""
    rng = np.random.default_rng(3)
    pos = rng.normal(0.0, 1.0, 300).tolist()
    neg = rng.normal(5.0, 1.0, 300).tolist()
    assert auc(pos, neg) > 0.99
    e, _ = eer(pos, neg)
    assert e < 0.02
    assert auc(neg, pos) < 0.01


def test_auc_ci_brackets_the_estimate_and_widens_when_starved():
    rng = np.random.default_rng(4)
    neg = rng.normal(4.0, 1.0, 400).tolist()
    wide = auc_ci(rng.normal(0.0, 1.0, 12).tolist(), neg)
    narrow = auc_ci(rng.normal(0.0, 1.0, 400).tolist(), neg)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_subsample_never_thins_a_small_category():
    pairs = ([{"cat": "A"}] * 5 + [{"cat": "D"}] * 500)
    out = _subsample(pairs, 100, 0)
    n = {}
    for p in out:
        n[p["cat"]] = n.get(p["cat"], 0) + 1
    assert n["A"] == 5
    assert n["D"] == 100
    assert [p["cat"] for p in out] == [p["cat"] for p in _subsample(pairs, 100, 0)]


def test_uniform_climb_grade_is_a_fraction_not_a_percent():
    """Guards a misuse that produced a 500 percent ramp and a saturated
    score of 793 that looked like a measurement of resolution."""
    d, e = uniform_climb(1000.0, grade=0.05)
    gain, loss = vertical_change(d, e)
    assert gain == pytest.approx(50.0, rel=0.02)
    assert loss == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("length", [1000.0, 3000.0, 6000.0])
def test_resolution_must_resolve_pitch_structure(length):
    """A 60 m pitch staircase and a ramp of identical length, gain and
    loss differ only in ordered shape below 120 m. The production
    resolution must separate them by a wide margin, and a coarse
    resolution must not.

    This is the evidence against raising the production default to 120 m
    on the strength of real-data measurements: the two long streams
    available are sampled at about 30 m and carry under 1 percent of
    their grade variance below 100 m, so they cannot see the difference
    this test makes visible.
    """
    d, e = gain_matched_staircase(length, total_gain_m=length * 0.05,
                                  pitch_m=60.0)
    d2, e2 = uniform_climb(length, grade=0.05)
    fine = match_segment(d2, e2, prepare_target(d, e, MatchConfig(res_m=70.0)),
                         MatchConfig(res_m=70.0))[0].score
    coarse = match_segment(d2, e2,
                           prepare_target(d, e, MatchConfig(res_m=150.0)),
                           MatchConfig(res_m=150.0))[0].score
    assert fine > 3.0, fine
    assert coarse < fine / 2.0, (fine, coarse)


def test_length_does_not_rescue_under_resolution():
    """Structure below the grid scale is destroyed, not averaged out, so
    a longer window cannot recover it."""
    scores = []
    for length in (1000.0, 3000.0, 6000.0):
        d, e = gain_matched_staircase(length, total_gain_m=length * 0.05,
                                      pitch_m=60.0)
        d2, e2 = uniform_climb(length, grade=0.05)
        cfg = MatchConfig(res_m=150.0)
        scores.append(match_segment(d2, e2, prepare_target(d, e, cfg),
                                    cfg)[0].score)
    assert max(scores) < 1.0, scores


@requires_corpus
def test_shape_recovery_does_not_degrade_with_length():
    """The concern that prompted the audit was that a 6 km segment might
    be too long to match reliably. Self-recovery error is measured here
    at both ends of the range; the long end must not be worse.
    """
    short = recovery(1000.0, stride_m=500.0)
    long = recovery(6000.0, stride_m=1500.0)
    assert long["n"] > 0 and short["n"] > 0
    assert long["localization"] <= short["localization"] * 1.5
    assert long["combined"] <= short["combined"] * 1.2


def test_pooled_B_is_not_reported_as_an_interpretable_number():
    """B_strict and B_loose must stay separate.

    Pooled, they produced an AUC that fell from 0.735 at 1 km to 0.574 at
    3 km and read as the matcher losing terrain similarity at long range.
    B_strict is flat at 0.75 to 0.84 over that range and B_loose sits at
    0.41 to 0.63, so the pooled value tracked only the shifting ratio
    between two populations, not anything about the matcher.
    """
    from bench.length import _bucket
    scored = ([{"cat": "B_strict", "score": 1.0}] * 3
              + [{"cat": "B_loose", "score": 9.0}] * 3
              + [{"cat": "D", "score": 5.0}])
    b = _bucket(scored)
    assert b["B_strict"] == [1.0, 1.0, 1.0]
    assert b["B_loose"] == [9.0, 9.0, 9.0]
    assert "B" not in b
    assert len(b["B_pooled_do_not_interpret"]) == 6


def test_matcher_separates_same_shape_at_different_steepness():
    """Two climbs with the same phase signature and very different
    steepness must NOT score as similar. B_loose sits below chance
    against unrelated terrain, which is the matcher declining to call a
    3 percent and a 9 percent climb the same hill. That is deliberate,
    so it is pinned rather than left to drift.
    """
    cfg = MatchConfig()
    d, e = uniform_climb(3000.0, grade=0.03)
    d2, e2 = uniform_climb(3000.0, grade=0.09)
    same = match_segment(d, e, prepare_target(d, e, cfg), cfg)[0].score
    diff = match_segment(d2, e2, prepare_target(d, e, cfg), cfg)[0].score
    assert same < 0.05
    assert diff > 2.0, diff
