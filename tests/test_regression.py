"""
Regression locks.

One test per defect found in the previous implementation. Each fails if
that specific defect returns. Locks that live more naturally beside the
code they exercise are in test_distance.py and test_matching.py and are
cross-referenced here so the full list is discoverable from one place:

  test_distance.test_unbanded_would_fail
      unbounded DTW scored 0.000 between climbs with 1/5 and 2.4x the
      target's steep distance
  test_distance.test_known_value
      quantile-grid Wasserstein returned 4.505 where the answer is 5.0
  test_distance.test_admissible
      the pruning bound must never exceed the true distance
  test_matching.test_window_length_grid_contains_exactly_one
      linspace(0.75, 1.15, 7) never lands on 1.0
  test_matching.test_segment_marginally_shorter_than_target
      over-long trial windows were skipped rather than clamped
  test_matching.test_vertical_change_is_resolution_independent
      gain measured at different densities for target and window
  test_matching.test_flat_target_does_not_disable_the_gain_term
      dividing by target vertical made the term inert at zero
  test_matching.test_pruning_does_not_change_the_result
      pruning must be an optimization, never a filter
"""

import unittest
import numpy as np

from segmatch.profile import build_profile, vertical_change
from segmatch.match import MatchConfig, prepare_target, match_segment
import find_similar_segments as cli


class TestProfileRegressions(unittest.TestCase):

    def test_profile_covers_the_whole_input(self):
        """Taking int(total // dx) + 1 samples discarded the final partial
        cell, silently dropping up to dx of real terrain from the end of
        every profile: a 5992 m segment became a 5970 m one, making the
        tail unsearchable."""
        for total in (5992.0, 6000.0, 6013.7, 1234.5):
            d = np.linspace(0.0, total, 900)
            e = 1500.0 + 0.04 * d
            p = build_profile(d, e, 120.0)
            self.assertAlmostEqual(p.length, total, places=6,
                                   msg=f"total {total}")

    def test_edge_grade_uses_a_full_width_estimator(self):
        """Edge samples were computed from truncated regression windows,
        giving them a different estimator with different bias. On a pure
        linear ramp every sample, first and last included, must return
        exactly the ramp's grade."""
        d = np.arange(0.0, 3000.0, 5.0)
        for pct in (2.0, 7.0, -4.5):
            p = build_profile(d, 1000.0 + (pct / 100.0) * d, 120.0)
            self.assertLess(float(np.abs(p.grade - pct).max()), 1e-6,
                            f"grade {pct}")

    def test_duplicate_distance_samples_do_not_explode(self):
        """A stationary recorder emits repeated distances. Dividing by a
        zero distance step yields an infinite grade."""
        d = np.array([0.0, 10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        e = np.array([100.0, 101.0, 105.0, 99.0, 102.0, 103.0, 104.0, 105.0])
        p = build_profile(d, e, 20.0)
        self.assertTrue(np.all(np.isfinite(p.grade)))

    def test_gain_is_never_negative_zero(self):
        d = np.arange(0.0, 1000.0, 10.0)
        gain, loss = vertical_change(d, 1000.0 + 0.05 * d)
        self.assertEqual(str(loss), "0.0")


class TestAccessRegressions(unittest.TestCase):

    def test_unchecked_access_no_longer_beats_a_measured_one(self):
        """Scoring a failed lookup as 0.0 gave it the same penalty as a
        segment starting at the trailhead, so a segment whose lookup
        happened to fail was rewarded with the best possible access
        score."""
        rows = [{"road_dist": 100.0, "penalty": 0.0},
                {"road_dist": 900.0, "penalty": 1.04},
                {"road_dist": 1300.0, "penalty": 2.5},
                {"road_dist": cli.ACCESS_UNCHECKED, "penalty": 0.0}]
        n, med = cli.impute_access_penalties(rows)
        self.assertEqual(n, 1)
        self.assertGreater(rows[3]["penalty"], 0.0)
        self.assertAlmostEqual(rows[3]["penalty"], med)

    def test_access_samples_more_than_the_start_point(self):
        """A segment starting at a car park and climbing into the
        backcountry read as fully accessible."""
        seen = []

        def fake(lat, lon, google_api_key=None, debug=False):
            seen.append((lat, lon))
            return 100.0 if len(seen) == 1 else 2000.0

        orig = cli.road_distance_m
        cli.road_distance_m = fake
        try:
            latlng = np.column_stack([np.linspace(40.0, 40.1, 100),
                                      np.linspace(-105.0, -105.1, 100)])
            d = np.linspace(0.0, 5000.0, 100)
            worst = cli.window_access(d, latlng, 0.0, 5000.0, None)
        finally:
            cli.road_distance_m = orig
        self.assertEqual(len(seen), 3)
        self.assertEqual(worst, 2000.0)


class TestSignificanceReporting(unittest.TestCase):

    def test_weak_matches_are_labelled_weak(self):
        """The search always returns a best window, so a ranked list looks
        equally authoritative whether the area holds a twin of the target
        or nothing like it."""
        null = np.linspace(1.0, 20.0, 200)
        self.assertIn("far better", cli.describe_significance(0.2, null))
        self.assertIn("NOT DISTINGUISHABLE",
                      cli.describe_significance(12.0, null))
        self.assertIn("no null model",
                      cli.describe_significance(1.0, np.array([])))

    def test_null_model_separates_a_real_match(self):
        from segmatch.match import null_scores
        from bench import synth
        rng = np.random.default_rng(2)
        cfg = MatchConfig()
        td, te = synth.terrain(rng, 6000.0, "variable", 8.0)
        target = prepare_target(td, te, cfg)
        profiles = []
        for k in range(10):
            d, e = synth.terrain(rng, 6000.0 * rng.uniform(1.0, 2.0),
                                 synth.ARCHETYPES[k % 7], 10.0)
            profiles.append(build_profile(d, e, cfg.res_m, cfg.oversample,
                                          cfg.vert_resample_m))
        null = null_scores(profiles, target, cfg, n=200)
        exact = match_segment(td, te, target, cfg)[0].score
        self.assertGreater(len(null), 0)
        self.assertLess(exact, np.percentile(null, 1))


if __name__ == "__main__":
    unittest.main()
