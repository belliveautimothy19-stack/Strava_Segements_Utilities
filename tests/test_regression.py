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

from segmatch.profile import (build_profile, vertical_change,
                               detect_quantization)
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


class TestResolutionRegressions(unittest.TestCase):
    """Locks for defects found in the second audit."""

    def test_gain_matched_staircase_is_not_a_match(self):
        """The decisive under-resolution probe.

        A staircase of steep pitches with flat recoveries can be built to
        have exactly the same length, gain and loss as a uniform climb, so
        length, vertical and grade composition are all matched by
        construction and ONLY ordered shape can separate them. At the
        previously selected res_m of 120 m the matcher scored a 60 m-pitch
        staircase at 0.188 against a uniform climb, which is a near
        perfect match between two completely different runs.

        The old benchmark could not have caught this: its terrain
        generator summed a fixed 24 harmonics over the route length, so it
        produced nothing below about 250 m and contained nothing that a
        120 m representation could under-resolve.
        """
        from bench import synth
        length_m, grade = 4000.0, 0.06
        gain = grade * length_m
        ud, ue = synth.uniform_climb(length_m, grade, spacing=4.0)
        cfg = MatchConfig()
        target = prepare_target(ud, ue, cfg)
        exact = match_segment(ud, ue, target, cfg)[0].score
        self.assertLess(exact, 0.1)
        for pitch in (60.0, 100.0):
            sd, se = synth.gain_matched_staircase(length_m, gain, pitch,
                                                   spacing=4.0)
            m = match_segment(sd, se, target, cfg)[0]
            # Vertical is matched by construction to within about 1
            # percent, so it cannot be what separates these. The lock is
            # twofold: the pair IS separated, and SHAPE is what does it.
            self.assertLess(cfg.w_gain * m.gain_dev, 0.25,
                            "probe is not gain matched; it no longer "
                            "isolates ordered shape")
            self.assertGreater(m.score, 2.0,
                               f"{pitch:.0f} m staircase scored {m.score:.3f}, "
                               f"which means the representation cannot "
                               f"resolve it")
            self.assertGreater(cfg.w_shape * m.shape, 0.5 * m.score,
                               "shape must supply the bulk of the "
                               "separation, not vertical or length")

    def test_overlap_respected_after_refinement(self):
        """Overlap suppression used to run on grid windows, after which
        refinement moved them and nothing re-checked. Three of 47 accepted
        pairs came back overlapping, one by 59 percent."""
        from bench import synth
        from segmatch.match import _overlap_frac
        rng = np.random.default_rng(4)
        cfg = MatchConfig(top_k=4, pool_size=64)
        for trial in range(12):
            td, te = synth.terrain(rng, 5000.0,
                                   synth.ARCHETYPES[trial % 8], 8.0)
            target = prepare_target(td, te, cfg)
            d, e = synth.terrain(rng, 22000.0,
                                 synth.ARCHETYPES[(trial + 3) % 8], 10.0)
            ms = match_segment(d, e, target, cfg)
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    ov = _overlap_frac((ms[i].start_m, ms[i].end_m),
                                       (ms[j].start_m, ms[j].end_m))
                    self.assertLessEqual(ov, cfg.max_overlap + 1e-9)

    def test_vertical_term_is_bounded_and_symmetric(self):
        """Dividing by the target's vertical alone was unbounded: a 20 m
        target against a 200 m candidate gave 26, swamping a shape
        distance whose whole range is about 0 to 10. It was also
        asymmetric, so the same pair scored 16.0 or 24.3 depending on
        which one was the target, making scores incomparable between
        targets and breaking the null-model percentile."""
        from segmatch.match import vertical_deviation
        rng = np.random.default_rng(5)
        for _ in range(500):
            g, l, G, L = rng.uniform(0, 800, 4)
            v = vertical_deviation(g, l, G, L)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
            self.assertAlmostEqual(v, vertical_deviation(G, L, g, l),
                                   places=12)
        self.assertEqual(vertical_deviation(0, 0, 0, 0), 0.0)
        self.assertAlmostEqual(vertical_deviation(0, 0, 500, 0), 1.0)

    def test_full_score_is_symmetric(self):
        d = np.arange(0, 4000.0, 5.0)
        a, b = 1500 + 0.10 * d, 1500 + 0.02 * d
        cfg = MatchConfig()
        ab = match_segment(d, b, prepare_target(d, a, cfg), cfg)[0].score
        ba = match_segment(d, a, prepare_target(d, b, cfg), cfg)[0].score
        self.assertAlmostEqual(ab, ba, places=6)

    def test_realistic_quantization_still_matches(self):
        """Finer resolution costs tolerance to elevation rounding, which
        manufactures staircase structure near the grade scale. 1 m
        rounding is common in served elevation data and must remain a
        clear match."""
        from bench import synth
        rng = np.random.default_rng(12)
        cfg = MatchConfig()
        td, te = synth.terrain(rng, 5000.0, "variable", 6.0, beta=1.45)
        target = prepare_target(td, te, cfg)
        ud, ue = synth.terrain(rng, 9000.0, "rolling", 10.0, beta=1.45)
        unrelated = match_segment(ud, ue, target, cfg)[0].score
        q1 = match_segment(td, synth.quantize_elevation(te, 1.0),
                           target, cfg)[0].score
        self.assertLess(q1, 1.0)
        self.assertLess(q1, unrelated / 3.0)

    def test_quantization_is_detected(self):
        from bench import synth
        rng = np.random.default_rng(1)
        d, e = synth.terrain(rng, 4000.0, "variable", 6.0)
        self.assertEqual(detect_quantization(e), 0.0)
        for q in (1.0, 2.0, 5.0):
            self.assertAlmostEqual(
                detect_quantization(synth.quantize_elevation(e, q)), q,
                places=6)

    def test_min_ratio_boundary_is_exact(self):
        cfg = MatchConfig()
        d = np.arange(0, 4000.0, 5.0)
        e = 1500 + 0.05 * d
        target = prepare_target(d, e, cfg)
        need = target.length_m * cfg.min_ratio
        for delta, expect in ((-0.01, 0), (0.0, 1), (0.01, 1)):
            dd = np.linspace(0.0, need + delta, 900)
            ms = match_segment(dd, 1500 + 0.05 * dd, target, cfg)
            self.assertEqual(len(ms), expect, f"delta {delta}")

    def test_benchmark_generator_has_short_scale_energy(self):
        """Locks the benchmark itself. If the generator loses its
        sub-100 m content again, every resolution sweep run against it
        becomes uninformative without failing anything."""
        from bench import synth
        rng = np.random.default_rng(0)
        d, e = synth.terrain(rng, 6000.0, "variable", spacing=6.0,
                             beta=1.45)
        g = np.gradient(e, d) * 100.0
        sp = np.abs(np.fft.rfft(g - g.mean())) ** 2
        k = np.arange(sp.size)
        wl = np.where(k > 0, 6000.0 / np.maximum(k, 1), np.inf)
        frac = sp[1:][wl[1:] < 100.0].sum() / sp[1:].sum()
        self.assertGreater(frac, 0.10)


class TestCliLibraryAgreement(unittest.TestCase):

    def test_cli_defaults_match_library_defaults(self):
        """The command line must not silently override the library.

        Two defaults drifted once already: --grade-res-m stayed at 120 and
        --weight-gain at 4.0 after the library moved to 70 and 2.0. Every
        command line run then used superseded parameters while the tests
        and the benchmark used the current ones, which means the whole
        parameter selection was invisible to the actual tool.
        """
        import argparse
        import find_similar_segments as cli
        from segmatch.match import MatchConfig

        parser_defaults = {}

        real_parse = argparse.ArgumentParser.parse_args

        def capture(self, *a, **kw):
            for action in self._actions:
                for opt in action.option_strings:
                    parser_defaults[opt] = action.default
            raise SystemExit(0)

        argparse.ArgumentParser.parse_args = capture
        try:
            try:
                cli.main()
            except SystemExit:
                pass
        finally:
            argparse.ArgumentParser.parse_args = real_parse

        cfg = MatchConfig()
        for flag, attr in (("--grade-res-m", "res_m"),
                           ("--min-window-frac", "min_ratio"),
                           ("--max-window-frac", "max_ratio"),
                           ("--length-steps", "length_steps"),
                           ("--start-step-frac", "stride_frac"),
                           ("--max-shift-frac", "max_shift_frac"),
                           ("--weight-shape", "w_shape"),
                           ("--weight-distribution", "w_dist"),
                           ("--weight-gain", "w_gain"),
                           ("--weight-length", "w_len")):
            self.assertIn(flag, parser_defaults)
            self.assertAlmostEqual(
                float(parser_defaults[flag]), float(getattr(cfg, attr)),
                places=9,
                msg=f"{flag} default disagrees with MatchConfig.{attr}")
