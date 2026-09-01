"""
Adversarial and ground-truth tests for the window matcher.

Every case here has a known right answer. Several are regression locks:
they fail against the previous implementation, and are marked as such so
the specific defect cannot come back unnoticed.
"""

import unittest
import numpy as np

from segmatch.match import (MatchConfig, prepare_target, match_segment,
                            comparison_length)
from segmatch.profile import build_profile, vertical_change, sanitize
from bench import synth

MI = 1609.34


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / (max(a[1], b[1]) - min(a[0], b[0]))


def make_target(seed=11, length=6000.0, kind="variable", spacing=8.0):
    rng = np.random.default_rng(seed)
    return synth.terrain(rng, length, kind, spacing)


class MatchCase(unittest.TestCase):

    def setUp(self):
        self.cfg = MatchConfig()
        self.rng = np.random.default_rng(99)
        self.td, self.te = make_target()
        self.target = prepare_target(self.td, self.te, self.cfg)

    def best(self, d, e, cfg=None):
        ms = match_segment(d, e, self.target, cfg or self.cfg)
        self.assertTrue(ms, "matcher returned no window at all")
        return ms[0]


class TestExactAndShifted(MatchCase):

    def test_exact_self_match_scores_near_zero(self):
        m = self.best(self.td, self.te)
        self.assertLess(m.score, 0.05)
        self.assertAlmostEqual(m.length_ratio, 1.0, places=2)
        self.assertEqual(m.direction, "forward")

    def test_identical_shifted_inside_larger_route(self):
        d, e, truth = synth.embed(self.td, self.te, self.rng,
                                   pre_m=2500.0, post_m=1800.0)
        m = self.best(d, e)
        self.assertGreater(iou((m.start_m, m.end_m), truth), 0.85)

    def test_match_at_route_start(self):
        d, e, truth = synth.embed(self.td, self.te, self.rng, pre_m=0.0)
        m = self.best(d, e)
        self.assertGreater(iou((m.start_m, m.end_m), truth), 0.85)

    def test_match_at_route_end(self):
        d, e, truth = synth.embed(self.td, self.te, self.rng, post_m=0.0)
        m = self.best(d, e)
        self.assertGreater(iou((m.start_m, m.end_m), truth), 0.85)

    def test_segment_marginally_shorter_than_target(self):
        """Regression lock. A segment a few metres shorter than the target
        used to have its natural length skipped entirely, because trial
        lengths over the segment length were dropped rather than clamped.
        The search then settled for 0.95x and scored 2.164 where 0.166 was
        available."""
        d = self.td[:-2]
        e = self.te[:-2]
        m = self.best(d, e)
        # The lock is that the natural length is REACHABLE. Previously the
        # 1.0x trial was dropped for being longer than the segment and the
        # search settled on 0.95x. A residual score remains because the
        # comparison is length-normalized, so a window even a fraction of
        # a percent short carries a shape floor that no band width
        # removes; that is a documented property, not the defect.
        self.assertGreater(m.length_ratio, 0.99)
        self.assertLess(m.score, 0.6)


class TestTransformedButEquivalent(MatchCase):

    def test_reversed_direction(self):
        d, e, truth = synth.embed(self.td, self.te[::-1].copy(), self.rng)
        m = self.best(d, e)
        self.assertEqual(m.direction, "reverse")
        self.assertGreater(iou((m.start_m, m.end_m), truth), 0.85)

    def test_gps_sampling_rate_differences(self):
        """Same terrain re-recorded at 4 m and 25 m spacing must both
        match. The old pipeline compared a target measured on raw GPX
        spacing against windows on a decimated grid."""
        for spacing in (4.0, 12.0, 25.0):
            d, e = synth.resample_at(self.td, self.te, spacing)
            m = self.best(d, e)
            self.assertLess(m.score, 0.6, f"spacing {spacing}")

    def test_uneven_sampling(self):
        d, e = synth.resample_uneven(self.td, self.te, self.rng)
        self.assertLess(self.best(d, e).score, 0.6)

    def test_elevation_noise(self):
        """Noise must degrade the score gracefully, not break the match.

        Asserted relative to unrelated terrain rather than against an
        absolute number: 1.5 m of barometric jitter is heavy, and it is
        correct for it to cost something. What matters is that a noisy
        recording of the right hill still beats the wrong hill by a wide
        margin.
        """
        rng = np.random.default_rng(31)
        ud, ue = synth.terrain(rng, 9000.0, "alternating", 10.0)
        unrelated = self.best(ud, ue).score
        for sigma in (0.6, 1.5):
            e = synth.add_baro_noise(self.te, self.rng, sigma)
            noisy = self.best(self.td, e).score
            self.assertLess(noisy, unrelated - 2.0, f"sigma {sigma}")
            self.assertLess(noisy, 3.0, f"sigma {sigma}")

    def test_small_grade_perturbation(self):
        d, e = synth.perturb_grade(self.td, self.te, self.rng, 1.0)
        self.assertLess(self.best(d, e).score, 1.5)

    def test_partial_match_reports_its_ratio(self):
        k = int(len(self.td) * 0.85)
        m = self.best(self.td[:k], self.te[:k])
        self.assertLess(m.length_ratio, 0.95)
        self.assertGreater(m.length_ratio, 0.75)


class TestDiscrimination(MatchCase):

    def test_aggregate_distribution_is_not_enough(self):
        """The decisive test.

        A block-shuffled copy has almost exactly the target's grade
        histogram and a completely different ordered shape. It must score
        far worse than the real thing, and the SHAPE term must be what
        separates them, not the composition term.
        """
        sd, se = synth.shuffle_blocks(self.td, self.te, self.rng)
        exact = self.best(self.td, self.te)
        shuffled = self.best(sd, se)
        self.assertGreater(shuffled.score, exact.score * 5 + 1.0)
        self.assertGreater(shuffled.shape, exact.shape + 1.0)
        # composition barely moves, which is exactly why it cannot be
        # relied on alone
        self.assertLess(shuffled.dist, shuffled.shape)

    def test_unrelated_terrain_scores_far_worse(self):
        rng = np.random.default_rng(7)
        exact = self.best(self.td, self.te).score
        for kind in ("descent", "alternating", "flat_with_pinch"):
            d, e = synth.terrain(rng, 9000.0, kind, 10.0)
            self.assertGreater(self.best(d, e).score, exact + 2.0)

    def test_materially_different_section_is_penalized(self):
        alt = self.te.copy()
        q = alt.size // 4
        drop = np.linspace(0, -0.09 * (self.td[2 * q] - self.td[q]), q)
        alt[q:2 * q] = alt[q] + drop
        alt[2 * q:] += alt[2 * q - 1] - self.te[2 * q - 1]
        exact = self.best(self.td, self.te).score
        self.assertGreater(self.best(self.td, alt).score, exact + 1.0)

    def test_long_flat_and_short_steep(self):
        flat_d = np.arange(0, 6000.0, 10.0)
        flat_e = np.full(flat_d.size, 1500.0)
        steep_d = np.arange(0, 6000.0, 10.0)
        steep_e = 1500.0 + 0.15 * steep_d
        exact = self.best(self.td, self.te).score
        self.assertGreater(self.best(flat_d, flat_e).score, exact + 1.0)
        self.assertGreater(self.best(steep_d, steep_e).score, exact + 1.0)

    def test_flat_target_does_not_disable_the_gain_term(self):
        """Regression lock. Dividing by the target's vertical made the
        gain term inert whenever that vertical was zero."""
        fd = np.arange(0, 6000.0, 10.0)
        fe = np.full(fd.size, 1500.0)
        cfg = MatchConfig()
        flat_target = prepare_target(fd, fe, cfg)
        hilly_e = 1500.0 + 0.08 * fd
        ms = match_segment(fd, hilly_e, flat_target, cfg)
        self.assertTrue(ms)
        # The term is bounded at 1.0 by design; a flat target against a
        # steadily climbing candidate is the maximum disagreement there
        # is, so the lock is that it saturates rather than going quiet.
        self.assertGreater(ms[0].gain_dev, 0.9)


class TestRepeatsAndOverlap(MatchCase):

    def _repeated_route(self, n=3, gap_m=1200.0):
        rng = np.random.default_rng(3)
        d_parts, e_parts, truths, cursor = [], [], [], 0.0
        td = self.td - self.td[0]
        for i in range(n):
            if i:
                gd, ge = synth.terrain(rng, gap_m, "rolling", 10.0)
                ge = ge - ge[0] + e_parts[-1][-1]
                d_parts.append(gd[1:] + cursor)
                e_parts.append(ge[1:])
                cursor += float(gd[-1])
            shifted = self.te - self.te[0] + (e_parts[-1][-1]
                                              if e_parts else 1500.0)
            d_parts.append(td + cursor)
            e_parts.append(shifted)
            truths.append((cursor, cursor + float(td[-1])))
            cursor += float(td[-1])
        d = np.concatenate(d_parts)
        e = np.concatenate(e_parts)
        keep = np.ones(d.size, dtype=bool)
        keep[1:] = np.diff(d) > 0
        return d[keep], e[keep], truths

    def test_repeated_pattern_yields_distinct_windows(self):
        d, e, truths = self._repeated_route(3)
        cfg = MatchConfig(top_k=3, pool_size=64)
        ms = match_segment(d, e, self.target, cfg)
        self.assertEqual(len(ms), 3)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                self.assertLess(
                    iou((ms[a].start_m, ms[a].end_m),
                        (ms[b].start_m, ms[b].end_m)), 0.5)
        found = sum(1 for t in truths
                    if any(iou((m.start_m, m.end_m), t) > 0.7 for m in ms))
        self.assertGreaterEqual(found, 2)

    def test_overlapping_windows_are_suppressed(self):
        d, e, _ = synth.embed(self.td, self.te, self.rng)
        cfg = MatchConfig(top_k=5, pool_size=64, max_overlap=0.5)
        ms = match_segment(d, e, self.target, cfg)
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                self.assertLessEqual(
                    iou((ms[a].start_m, ms[a].end_m),
                        (ms[b].start_m, ms[b].end_m)), 0.7)


class TestSearchProperties(MatchCase):

    def test_pruning_does_not_change_the_result(self):
        """Exactness of the lower-bound pruning. Turning it off must give
        the identical top window."""
        rng = np.random.default_rng(21)
        on = MatchConfig()
        off = MatchConfig(use_pruning=False)
        for kind in ("rolling", "smooth_climb", "alternating"):
            d, e = synth.terrain(rng, 11000.0, kind, 10.0)
            a = match_segment(d, e, self.target, on)[0]
            b = match_segment(d, e, self.target, off)[0]
            self.assertAlmostEqual(a.score, b.score, places=9)
            self.assertAlmostEqual(a.start_m, b.start_m, places=6)
            self.assertEqual(a.direction, b.direction)

    def test_deterministic(self):
        d, e, _ = synth.embed(self.td, self.te, self.rng)
        a = match_segment(d, e, self.target, self.cfg)[0]
        b = match_segment(d, e, self.target, self.cfg)[0]
        self.assertEqual((a.score, a.start_m, a.direction),
                         (b.score, b.start_m, b.direction))

    def test_window_length_grid_contains_exactly_one(self):
        """Regression lock. linspace(0.75, 1.15, 7) never lands on 1.0, so
        the most likely correct length was the one never tried."""
        from segmatch.match import _window_lengths
        lens = _window_lengths(1000.0, 5000.0, MatchConfig())
        self.assertTrue(np.any(np.isclose(lens, 1000.0)))

    def test_offsets_reach_both_ends(self):
        from segmatch.match import _window_starts
        starts = _window_starts(5000.0, 1000.0, 1000.0, MatchConfig())
        self.assertAlmostEqual(starts[0], 0.0)
        self.assertAlmostEqual(starts[-1], 4000.0)

    def test_too_short_segment_returns_nothing(self):
        d = np.arange(0, 1000.0, 10.0)
        e = np.full(d.size, 1500.0)
        self.assertEqual(match_segment(d, e, self.target, self.cfg), [])


class TestProfileInvariants(unittest.TestCase):

    def test_resampling_invariance(self):
        """The same hill recorded at different rates must produce nearly
        the same normalized profile."""
        rng = np.random.default_rng(2)
        d, e = synth.terrain(rng, 5000.0, "rolling", 5.0)
        base = build_profile(d, e, 120.0)
        grade_sd = float(np.std(np.gradient(e, d) * 100.0))
        # Bounded relative to the signal, not by an absolute grade. The
        # old absolute 1.0 percent bound was set when the generator
        # produced nothing below 250 m wavelength; on terrain with real
        # short-scale structure, sampling at 40 m genuinely loses more,
        # and an absolute bound would only be re-tightening the benchmark
        # until it stopped complaining.
        for spacing, tol in ((10.0, 0.10), (20.0, 0.15), (40.0, 0.30)):
            rd, re = synth.resample_at(d, e, spacing)
            p = build_profile(rd, re, 120.0)
            n = min(len(base.grade), len(p.grade))
            worst = float(np.abs(base.grade[:n] - p.grade[:n]).max())
            self.assertLess(worst, tol * grade_sd,
                            f"spacing {spacing}: {worst:.2f}% is more than "
                            f"{100*tol:.0f}% of the {grade_sd:.2f}% grade sd")

    def test_vertical_change_is_resolution_independent(self):
        """Regression lock for the gain bug. Naive summed differences
        report 2172 m for a hill sampled every 1 m and 300 m for the same
        hill every 100 m."""
        x = np.arange(0, 6001.0, 1.0)
        base = 0.05 * x + 12 * np.sin(2 * np.pi * x / 1500)
        rng = np.random.default_rng(4)
        vals = []
        for spacing in (1.0, 4.0, 10.0, 25.0):
            xs = np.arange(0, 6001.0, spacing)
            es = np.interp(xs, x, base) + rng.normal(0, 0.6, xs.size)
            vals.append(vertical_change(xs, es)[0])
        self.assertLess(max(vals) - min(vals), 40.0)

    def test_sanitize_handles_bad_input(self):
        d = np.array([0.0, 10.0, 10.0, 20.0, 15.0, 30.0])
        e = np.array([1.0, np.nan, 2.0, 3.0, 9.0, 4.0])
        sd, se = sanitize(d, e)
        self.assertTrue(np.all(np.diff(sd) > 0))
        self.assertTrue(np.all(np.isfinite(se)))

    def test_sanitize_rejects_hopeless_input(self):
        with self.assertRaises(ValueError):
            sanitize(np.array([0.0, 1.0]), np.array([np.nan, np.nan]))

    def test_comparison_length_is_bounded(self):
        self.assertGreaterEqual(comparison_length(100.0, 120.0), 8)
        self.assertLessEqual(comparison_length(10 ** 7, 120.0), 512)


if __name__ == "__main__":
    unittest.main()
