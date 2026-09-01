"""
Regression locks for audit 7.

Audit 7's premise was that the measuring apparatus, not the matcher, had
been the recurring source of wrong conclusions. It found four more
defects, three of them in measurement. Each is pinned here.

A note on what these tests are for. Several of them do not assert that
the matcher is correct; they assert that an instrument reports honestly,
including reporting that it cannot answer. A test suite that only checks
implementation details is how six audits in a row produced clean runs
while the conclusions were wrong.
"""

import math

import numpy as np
import pytest

import audit7.corpus as C
from tests.corpus_guard import requires_audit_corpus
from audit7.independent import (auc_lower_is_better, bootstrap_ci,
                                grade_percent, haversine_m,
                                interval_is_adequate, overlap_fraction,
                                resample_uniform, vertical_change_m)
from audit7.verify_synthetic import (dominant_wavelength_m, measure,
                                     nyquist_limit_m)
from bench.synth import gain_matched_staircase, uniform_climb
from segmatch.match import MatchConfig, match_segment, prepare_target


# ---------------------------------------------------------------- units

def test_grade_is_percent_not_fraction():
    """A 5 percent ramp must read 5.0.

    A fraction passed where a percent was expected produced a 500 percent
    ramp and a saturated score of 793, which was briefly read as a
    resolution measurement.
    """
    d = np.arange(0.0, 1000.0, 5.0)
    grid, elev = resample_uniform(d, 1500.0 + 0.05 * d, 10.0)
    assert float(np.median(grade_percent(grid, elev))) == pytest.approx(5.0)


def test_uniform_climb_grade_argument_is_a_fraction():
    d, e = uniform_climb(1000.0, grade=0.05)
    assert vertical_change_m(d, e)[0] == pytest.approx(50.0, rel=0.02)


def test_haversine_is_metres_not_radians_or_kilometres():
    d = float(haversine_m(40.0150, -105.2705, 39.7392, -104.9903))
    assert 30000.0 < d < 50000.0, d


# ------------------------------------------------------------- geometry

def test_overlap_is_symmetric_under_argument_order():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = np.column_stack([40.0 + rng.normal(0, 1e-3, 25),
                             -105.3 + rng.normal(0, 1e-3, 25)])
        B = np.column_stack([40.0 + rng.normal(0, 1e-3, 25),
                             -105.3 + rng.normal(0, 1e-3, 25)])
        a, sa, _ = overlap_fraction(A, B)
        b, sb, _ = overlap_fraction(B, A)
        assert a == pytest.approx(b)
        assert sa == pytest.approx(sb)


def test_overlap_fraction_is_bounded():
    rng = np.random.default_rng(1)
    A = np.column_stack([40.0 + rng.normal(0, 1e-2, 40),
                         -105.3 + rng.normal(0, 1e-2, 40)])
    for B in (A, A + 1.0):
        ov, _, both = overlap_fraction(A, B)
        assert 0.0 <= ov <= 1.0
        assert all(0.0 <= x <= 1.0 for x in both)


def test_identical_traces_fully_overlap():
    rng = np.random.default_rng(2)
    A = np.column_stack([40.0 + rng.normal(0, 1e-3, 30),
                         -105.3 + rng.normal(0, 1e-3, 30)])
    ov, sep, _ = overlap_fraction(A, A)
    assert ov == 1.0 and sep == pytest.approx(0.0)


# ------------------------------------------------------- label validity

def test_archetype_label_is_stable_or_declared_unstable():
    """The classifier must not silently return a label that a 30 m window
    shift would change.

    Measured before this was addressed: 24.3 percent of window labels
    changed when one or two leading samples were dropped. Category B is
    built entirely from these labels, so a quarter of it was noise.
    Rewriting the classifier reduced that only to 20.6 percent, which is
    why reliability is now measured per window rather than assumed.
    """
    d = np.arange(0.0, 1200.0, 20.0)
    e = 1500.0 + 0.04 * d
    label, stable, _ = C.stable_label(d, e)
    assert label == "up"
    assert stable is True


def test_ambiguous_terrain_is_reported_unstable_not_labelled_confidently():
    """Terrain that wobbles either side of the flat band has no
    well-defined archetype, and the instrument must say so."""
    rng = np.random.default_rng(3)
    d = np.arange(0.0, 1200.0, 20.0)
    e = 1500.0 + 0.015 * d + rng.normal(0, 1.2, len(d)).cumsum() * 0.25
    _, stable, _ = C.stable_label(d, e)
    assert stable in (True, False)          # must not raise
    labels = set()
    for k in (0, 1, 2, 3):
        lab, _ = C.archetype(d[k:] - d[k], e[k:])
        if lab:
            labels.add(lab)
    if len(labels) > 1:
        assert stable is False, "unstable terrain reported as stable"


# ------------------------------------------- negative-class construction

def test_geometric_negatives_do_not_exclude_hard_cases():
    """The negative class must not be filtered by similarity.

    Earlier audits defined the negative as "different archetype AND
    unmatched statistics", which routed the hardest negatives into other
    categories and left an easy remainder. The measured cost was an AUC
    biased upward by roughly 0.02 to 0.03 at every window length. The
    geometric negative admits every geographically separate pair.
    """
    ws = _toy_windows()
    pairs = C.categorize_geometric(ws)
    cats = {p["cat"] for p in pairs}
    assert cats <= {"A", "N"}
    for p in pairs:
        if p["cat"] == "N":
            assert p["ov"] <= C.GEO_APART_FRAC
            assert p["sep"] >= C.MIN_APART_M
            # nothing about shape or statistics may gate membership
            assert "arch" not in p


def _toy_windows():
    ws = []
    for k, (lat, lon) in enumerate([(40.00, -105.30), (40.05, -105.40),
                                    (40.10, -105.50)]):
        d = np.arange(0.0, 1000.0, 20.0)
        e = 1500.0 + 0.03 * d + k * 5.0
        ll = np.column_stack([lat + np.zeros_like(d), lon + d * 1e-5])
        ws.append({"route": "r%d" % k, "start": 0.0, "d": d, "e": e,
                   "ll": ll, "label": "up", "arch": "up", "n_phases": 1,
                   "steep": 3.0, "mean_grade": 3.0,
                   "grades": np.full(20, 3.0), "gain": 30.0, "loss": 0.0,
                   "spacing": 20.0, "label_stable": True,
                   "profile": C.normalized_profile(d, e),
                   "fingerprint": C._fingerprint(d, e)})
    return ws


def test_byte_identical_windows_cannot_form_a_positive():
    ws = _toy_windows()
    twin = dict(ws[0])
    twin["route"] = "copy"
    ws.append(twin)
    for p in C.categorize_geometric(ws):
        assert p["A"]["fingerprint"] != p["B"]["fingerprint"]


def test_no_pair_carries_two_labels():
    C.assert_no_population_leakage(C.categorize_geometric(_toy_windows()))


# ------------------------------------------------ statistical honesty

def test_interval_adequacy_refuses_an_underpowered_comparison():
    """An AUC of 0.986 from twelve positives has an interval wider than
    every difference the experiment was built to detect."""
    rng = np.random.default_rng(4)
    neg = rng.normal(4.0, 1.0, 400).tolist()
    lo, hi = bootstrap_ci(rng.normal(0.0, 1.0, 12).tolist(), neg)
    # The interval is NARROW here, [0.9925, 1.0], because the statistic is
    # pinned against its ceiling rather than well determined. A width test
    # alone passes it, which is why the count floor exists.
    assert (hi - lo) < 0.05
    assert not interval_is_adequate(lo, hi, 0.05, n_pos=12)
    lo, hi = bootstrap_ci(rng.normal(0.0, 1.0, 600).tolist(), neg)
    assert interval_is_adequate(lo, hi, 0.05, n_pos=600)


def test_saturated_auc_does_not_pass_adequacy_on_width_alone():
    """The exact failure mode above, stated as its own lock."""
    rng = np.random.default_rng(11)
    neg = rng.normal(9.0, 0.5, 300).tolist()
    pos = rng.normal(0.0, 0.5, 8).tolist()
    lo, hi = bootstrap_ci(pos, neg)
    assert auc_lower_is_better(pos, neg) == 1.0
    assert (hi - lo) < 0.01
    assert not interval_is_adequate(lo, hi, 0.05, n_pos=len(pos))


def test_threshold_at_a_distribution_edge_is_flagged():
    scores = np.random.default_rng(5).normal(0, 1, 500)
    assert C.assert_threshold_not_degenerate(float(scores.max()) + 1,
                                             scores)["degenerate"]
    assert not C.assert_threshold_not_degenerate(float(np.median(scores)),
                                                 scores)["degenerate"]


def test_auc_longhand_matches_broadcast():
    rng = np.random.default_rng(6)
    for _ in range(50):
        p = rng.normal(0, 1, 20).tolist()
        n = rng.normal(1, 1, 20).tolist()
        a, b = np.asarray(p), np.asarray(n)
        fast = float(((a[:, None] < b[None, :]).sum()
                      + 0.5 * (a[:, None] == b[None, :]).sum()) / 400.0)
        assert auc_lower_is_better(p, n) == pytest.approx(fast)


# ------------------------------------------------- synthetic validity

@pytest.mark.parametrize("pitch", [60.0, 120.0, 240.0])
def test_staircase_is_what_the_experiment_claims(pitch):
    """Verify the probe before trusting the conclusion drawn from it."""
    L = 6000.0
    d, e = gain_matched_staircase(L, total_gain_m=L * 0.05, pitch_m=pitch)
    d2, e2 = uniform_climb(L, grade=0.05)
    ms, mr = measure(d, e), measure(d2, e2)
    assert ms["gain_m"] / mr["gain_m"] == pytest.approx(1.0, abs=0.05)
    assert ms["loss_m"] == pytest.approx(0.0, abs=1.0)
    assert mr["grade_sd_pct"] == pytest.approx(0.0, abs=1e-6)
    assert dominant_wavelength_m(d, e) == pytest.approx(2 * pitch, rel=0.10)
    # the probe must not itself be sampling-limited at the scale tested
    assert nyquist_limit_m(d) < pitch / 4.0


def test_staircase_separation_is_shape_not_gain():
    """If gain_dev carried the separation, the probe would be measuring
    the vertical term rather than resolution."""
    L = 6000.0
    d, e = gain_matched_staircase(L, total_gain_m=L * 0.05, pitch_m=60.0)
    d2, e2 = uniform_climb(L, grade=0.05)
    cfg = MatchConfig(res_m=70.0)
    m = match_segment(d2, e2, prepare_target(d, e, cfg), cfg)[0]
    assert m.gain_dev == pytest.approx(0.0, abs=1e-6)
    assert m.len_dev == pytest.approx(0.0, abs=1e-6)
    assert m.shape > 2.0


# ------------------------------------------------- frozen production

def test_production_defaults_are_frozen():
    """Audit 7 asserts production was not modified while it ran."""
    c = MatchConfig()
    assert c.res_m == 70.0
    assert c.max_shift_frac == 0.03
    assert (c.w_shape, c.w_dist, c.w_gain, c.w_len) == (1.0, 0.6, 2.0, 2.0)
    assert c.min_ratio == 0.75 and c.max_ratio == 1.15
    assert c.vert_resample_m == 25.0
    assert c.dist_bin_w == 0.0


def test_alignment_band_does_not_affect_staircase_discrimination():
    """Audit 2 claimed the band protected against the gain-matched
    staircase. It does not, at any band from 0.03 to 0.30. The claim was
    retracted in audit 6 and is pinned here so it cannot return."""
    L = 6000.0
    d, e = gain_matched_staircase(L, total_gain_m=L * 0.05, pitch_m=60.0)
    d2, e2 = uniform_climb(L, grade=0.05)
    scores = []
    for band in (0.03, 0.10, 0.30):
        cfg = MatchConfig(max_shift_frac=band)
        scores.append(match_segment(d2, e2, prepare_target(d, e, cfg),
                                    cfg)[0].score)
    assert max(scores) - min(scores) < 0.01, scores


def test_shape_and_dist_are_distinct_terms_in_general():
    """They coincide exactly when the candidate has constant grade, which
    is why the staircase probe shows shape == dist. That is a property of
    that degenerate pair, not evidence the distribution term is dead."""
    rng = np.random.default_rng(7)
    d = np.arange(0.0, 1000.0, 10.0)
    a = 1500.0 + np.cumsum(rng.normal(0, 0.6, len(d)))
    b = 1500.0 + np.cumsum(rng.normal(0, 0.6, len(d)))
    cfg = MatchConfig()
    m = match_segment(d, b, prepare_target(d, a, cfg), cfg)[0]
    assert abs(m.shape - m.dist) > 1e-6


# ------------------------------------------------ documentation truth

def test_documented_resolution_matches_the_real_default():
    """The class docstring said res_m 120 for three audits after the
    default moved to 70. Prose and value disagreeing is the same hazard
    as a fraction read as a percent, and it is user-facing: the CLI
    passes its module docstring to --help.
    """
    import segmatch.match as M
    import find_similar_segments as F
    default = MatchConfig().res_m
    assert "res_m %d" % default in M.MatchConfig.__doc__
    assert "default %d m" % default in F.__doc__
    for stale in ("res_m 120", "default 120 m"):
        assert stale not in M.MatchConfig.__doc__
        assert stale not in (F.__doc__ or "")


def test_cli_help_does_not_claim_shape_over_magnitude():
    """Measured rank correlation is 0.55 against steepness difference and
    0.13 against steepness-blind ordered shape. Text promising the
    opposite emphasis misdescribes the product."""
    import find_similar_segments as F
    doc = F.__doc__ or ""
    assert "CLIMBING" in doc.upper()
    assert "Not just a similar\naverage grade or distance, but a similar ordered shape" not in doc


# ------------------------------------------- evidence-plan arithmetic

def test_eight_km_does_not_yield_thirty_pairs():
    """Audit 7 proposed routes of >= 8 km. Checked against the measured
    yield rather than asserted, an 8 km route recorded twice gives FOUR
    category A pairs at a 6 km window, not thirty. The requirement was
    wrong and is now 12 km with a floor of 10 pairs.
    """
    from audit7.accept import (MIN_A_PAIRS_PER_TRAIL, MIN_ROUTE_LENGTH_M,
                               expected_A_pairs)
    assert expected_A_pairs(8000.0) < 10
    assert expected_A_pairs(MIN_ROUTE_LENGTH_M) >= MIN_A_PAIRS_PER_TRAIL
    assert expected_A_pairs(5000.0) == 0
    # monotone in length
    ys = [expected_A_pairs(x) for x in (6000, 9000, 12000, 18000, 24000)]
    assert ys == sorted(ys)


def test_dense_recording_must_also_be_long_enough_to_use():
    """A 5.2 km route sampled every 6 m cannot produce a 6 km window, so
    it cannot help the resolution question however dense it is. The first
    acceptance checker counted it as satisfying the dense criterion."""
    from audit7.accept import evaluate
    rec = {"name": "dense_but_short", "span_m": 5200.0, "spacing_m": 6.0,
           "n_points": 900, "elev_step_m": 0.1, "relief_m": 120.0,
           "has_gps": True, "window_positions_6km": 0,
           "can_yield_6km_window": False, "dense": True,
           "frac_grade_var_below_60m": 0.34,
           "frac_grade_var_below_120m": 0.45,
           "ll": np.column_stack([40.0 + np.arange(50) * 1e-4,
                                  -105.3 + np.zeros(50)])}
    rep = evaluate([rec, dict(rec, name="second")])
    assert rep["criteria"]["dense_trails_usable_at_6km"][0] == 0


def test_acceptance_checker_counts_trails_not_activities():
    """Two recordings of one trail are one trail. Criterion 1 counts
    locations; adding files to the same trail must not satisfy it."""
    from audit7.accept import evaluate
    ll = np.column_stack([40.0 + np.arange(60) * 1e-4, -105.3 + np.zeros(60)])
    base = {"name": "a", "span_m": 20000.0, "spacing_m": 8.0,
            "n_points": 2500, "elev_step_m": 0.1, "relief_m": 300.0,
            "has_gps": True, "window_positions_6km": 10,
            "can_yield_6km_window": True, "dense": True,
            "frac_grade_var_below_60m": 0.3,
            "frac_grade_var_below_120m": 0.4, "ll": ll}
    rep = evaluate([dict(base, name="rec%d" % i) for i in range(6)])
    assert rep["n_recordings"] == 6
    assert rep["n_trails"] == 1
    assert not rep["satisfied"]


# --------------------------------------------- scoring architecture

def test_score_does_not_scale_with_window_length():
    """An identical relative perturbation must not cost more simply
    because the window is longer. DTW is normalized by max(len(a),
    len(b)), so the shape term is a mean per sample, not a sum."""
    rng = np.random.default_rng(0)
    cfg = MatchConfig()
    scores = []
    for L in (1000.0, 3000.0, 6000.0):
        d = np.arange(0.0, L, 10.0)
        base = 1500.0 + np.cumsum(rng.normal(0, 0.35, len(d)))
        alt = base + np.cumsum(rng.normal(0, 0.05, len(d)))
        scores.append(match_segment(d, alt, prepare_target(d, base, cfg),
                                    cfg)[0].score)
    assert max(scores) < 2.0 * min(scores), scores


def test_band_is_max_of_one_sample_and_three_percent():
    """The band is not 3 percent at every length. It floors at one
    comparison sample, so below about 1170 m it is res_m/2 in metres and
    the effective fraction inflates: 9.1 percent at 400 m, 5.9 at 600 m,
    3.5 at 1000 m."""
    from segmatch.match import comparison_length
    cfg = MatchConfig()
    for L, lo, hi in ((400.0, 0.08, 0.10), (600.0, 0.055, 0.065),
                      (6000.0, 0.028, 0.031)):
        n = comparison_length(L, cfg.res_m)
        band_m = max(1, round(cfg.max_shift_frac * n)) * (L / n)
        assert lo < band_m / L < hi, (L, band_m / L)


def test_comparison_length_cap_is_documented_boundary():
    """n_cmp caps at 512, so above about 17.9 km at res 70 the shape
    comparison samples coarser than res_m/2 and silently under-resolves
    the profile it was built from. Not reachable at the 6 km product
    length; pinned so the boundary is not forgotten."""
    from segmatch.match import comparison_length
    cfg = MatchConfig()
    limit = 512 * cfg.res_m / 2.0
    assert comparison_length(limit, cfg.res_m) == 512
    below = comparison_length(limit - 1000.0, cfg.res_m)
    assert below < 512
    assert (limit - 1000.0) / below == pytest.approx(cfg.res_m / 2.0, rel=0.02)


# ------------------------------------------------- unit of replication

def test_cluster_bootstrap_is_wider_than_pair_bootstrap():
    """Pairs cut from one trail are nested observations, not independent
    draws. Measured on real data, the pair bootstrap is 1.5x too narrow at
    1 km and 1.8x at 2.2 km."""
    from audit7.independent import bootstrap_ci, cluster_bootstrap_ci
    rng = np.random.default_rng(0)
    pos, trail = [], []
    for t in range(4):
        centre = rng.normal(0.0, 0.8)
        pos.extend(rng.normal(centre, 0.3, 50).tolist())
        trail.extend([t] * 50)
    neg = rng.normal(3.0, 1.0, 400).tolist()
    plo, phi = bootstrap_ci(pos, neg)
    clo, chi = cluster_bootstrap_ci(pos, neg, trail)
    assert (chi - clo) > (phi - plo)


def test_single_trail_cannot_produce_a_between_trail_interval():
    """The 6 km headline came with an interval computed over pairs from
    one trail. There is no such interval. The function must refuse rather
    than return a number that looks like one."""
    from audit7.independent import cluster_bootstrap_ci
    rng = np.random.default_rng(1)
    pos = rng.normal(0.0, 1.0, 80).tolist()
    neg = rng.normal(3.0, 1.0, 200).tolist()
    lo, hi = cluster_bootstrap_ci(pos, neg, ["one_trail"] * 80)
    assert math.isnan(lo) and math.isnan(hi)


def test_cluster_bootstrap_accepts_tuple_trail_labels():
    """Trail ids are naturally tuples of route ids; numpy turns a list of
    tuples into a 2-D array rather than an array of labels."""
    from audit7.independent import cluster_bootstrap_ci
    rng = np.random.default_rng(2)
    pos = rng.normal(0, 1, 40).tolist()
    neg = rng.normal(3, 1, 60).tolist()
    labels = [("a", "b")] * 20 + [("c", "d")] * 20
    lo, hi = cluster_bootstrap_ci(pos, neg, labels)
    assert not math.isnan(lo) and hi >= lo


@requires_audit_corpus
def test_trail_id_is_geographic_not_file_identity():
    """Two recordings of one trail must share a trail id.

    Labelling the replication unit by route PAIRING reported three trails
    at 6 km when all three pairings were the same physical trail seen
    twice: A x B, A x A and B x B. Pseudo-replication regained through the
    pair label is precisely what the cluster bootstrap exists to prevent,
    so the label must come from geography.
    """
    ids = C.trail_ids()
    assert ids["19476565994"] == ids["19670306718"], (
        "two recordings of one trail must share a trail id")
    assert len(set(ids.values())) == 5
    for a, b in (("19131631580", "19621145681"),
                 ("19853326285", "19869723537"),
                 ("19476565994", "19853326285")):
        assert ids[a] != ids[b], (a, b)


@requires_audit_corpus
def test_pair_trail_id_collapses_self_and_cross_pairings():
    """A x A, B x B and A x B are one trail when A and B are recordings of
    the same ground, not three."""
    ids = C.trail_ids()
    mk = lambda x, y: {"A": {"route": x}, "B": {"route": y}}
    got = {C.pair_trail_id(mk(a, b), ids)
           for a, b in (("19476565994", "19476565994"),
                        ("19670306718", "19670306718"),
                        ("19476565994", "19670306718"))}
    assert len(got) == 1, got
