"""Metric-level invariants.

These are the properties the matcher's correctness argument rests on. If
any of them breaks, the search is no longer exact.
"""

import unittest
import numpy as np

from segmatch.distance import (dtw_band, keogh_envelope, lb_keogh,
                                wasserstein1, grade_histogram,
                                hist_distance)


def dtw_reference(a, b, band):
    """Straightforward O(n*m) banded DTW, used only to check the fast
    implementation. Deliberately written for clarity, not speed."""
    n, m = len(a), len(b)
    w = int(max(band, abs(n - m), 1))
    c = np.full((n + 1, m + 1), np.inf)
    c[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            c[i, j] = abs(a[i - 1] - b[j - 1]) + min(
                c[i - 1, j], c[i, j - 1], c[i - 1, j - 1])
    return c[n, m] / max(n, m)


class TestDTW(unittest.TestCase):

    def test_identity_is_zero(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            a = rng.normal(0, 5, int(rng.integers(4, 60)))
            self.assertAlmostEqual(dtw_band(a, a, 3), 0.0, places=12)

    def test_symmetry(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            n = int(rng.integers(4, 40))
            a, b = rng.normal(0, 5, n), rng.normal(0, 5, n)
            self.assertAlmostEqual(dtw_band(a, b, 4), dtw_band(b, a, 4),
                                   places=12)

    def test_matches_reference(self):
        """Exactness. The fast band scan must agree bit for bit with a
        transparently correct implementation, at equal and unequal
        lengths."""
        rng = np.random.default_rng(2)
        worst = 0.0
        for _ in range(200):
            n = int(rng.integers(3, 50))
            m = max(2, n + int(rng.integers(-3, 4)))
            a, b = rng.normal(0, 4, n), rng.normal(0, 4, m)
            band = int(rng.integers(1, 9))
            worst = max(worst, abs(dtw_band(a, b, band)
                                   - dtw_reference(a, b, band)))
        self.assertEqual(worst, 0.0)

    def test_band_sees_section_length(self):
        """Regression lock for the original degeneracy.

        Unconstrained DTW scored a perfect 0.000 between a target holding
        5 bins of 8 percent and windows holding 1 bin or 12 bins of it,
        because it could stretch one sample across many. A banded DTW
        must charge for both.
        """
        target = np.array([8.0] * 5 + [2.0] * 10)
        too_short = np.array([8.0] * 1 + [2.0] * 14)
        too_long = np.array([8.0] * 12 + [2.0] * 3)
        self.assertAlmostEqual(dtw_band(target, target, 2), 0.0, places=12)
        self.assertGreater(dtw_band(target, too_short, 2), 0.5)
        self.assertGreater(dtw_band(target, too_long, 2), 1.0)

    def test_unbanded_would_fail(self):
        """The same comparison with an unbounded band collapses to zero,
        which is precisely the behaviour this design rejects."""
        target = np.array([8.0] * 5 + [2.0] * 10)
        too_short = np.array([8.0] * 1 + [2.0] * 14)
        self.assertAlmostEqual(dtw_band(target, too_short, 999), 0.0,
                               places=12)


class TestLowerBound(unittest.TestCase):

    def test_admissible(self):
        """LB_Keogh must never exceed the true banded DTW.

        This is what licenses pruning. If it can overshoot, the search
        silently discards windows that would have won.
        """
        rng = np.random.default_rng(3)
        for _ in range(400):
            n = int(rng.integers(6, 60))
            q, c = rng.normal(0, 5, n), rng.normal(0, 5, n)
            band = int(rng.integers(1, 8))
            up, dn = keogh_envelope(q, band)
            lb = lb_keogh(c, up, dn, n)
            self.assertLessEqual(lb, dtw_band(q, c, band) + 1e-9)

    def test_bound_is_tight_enough_to_prune(self):
        """A useless bound is admissible but prunes nothing. On clearly
        dissimilar pairs the bound should be well above zero."""
        q = np.concatenate([np.full(30, 8.0), np.full(30, 1.0)])
        c = np.full(60, -6.0)
        up, dn = keogh_envelope(q, 2)
        self.assertGreater(lb_keogh(c, up, dn, 60), 5.0)


class TestWasserstein(unittest.TestCase):

    def test_known_value(self):
        """Half at 0 and half at 10 against a constant 5 is exactly 5."""
        a = np.array([0.0] * 50 + [10.0] * 50)
        b = np.full(100, 5.0)
        self.assertAlmostEqual(wasserstein1(a, b), 5.0, places=9)

    def test_order_independent(self):
        rng = np.random.default_rng(4)
        a = rng.normal(3, 4, 60)
        b = rng.normal(1, 2, 60)
        self.assertAlmostEqual(wasserstein1(a, b),
                               wasserstein1(rng.permutation(a), b),
                               places=12)

    def test_identity_and_symmetry(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 3, 40), rng.normal(1, 2, 55)
        self.assertAlmostEqual(wasserstein1(a, a), 0.0, places=12)
        self.assertAlmostEqual(wasserstein1(a, b), wasserstein1(b, a),
                               places=12)

    def test_scales_with_shift(self):
        a = np.linspace(-5, 5, 100)
        for shift in (0.5, 2.0, 7.0):
            self.assertAlmostEqual(wasserstein1(a, a + shift), shift,
                                   places=6)


class TestHistogram(unittest.TestCase):

    def test_identity(self):
        rng = np.random.default_rng(6)
        g = rng.normal(4, 3, 200)
        h = grade_histogram(g, 1.0)
        self.assertAlmostEqual(hist_distance(h, h, 1.0), 0.0, places=12)
        self.assertAlmostEqual(h.sum(), 1.0, places=12)

    def test_approximates_exact_emd(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(4, 3, 400), rng.normal(6, 3, 400)
        exact = wasserstein1(a, b)
        binned = hist_distance(grade_histogram(a, 0.25),
                               grade_histogram(b, 0.25), 0.25)
        self.assertLess(abs(exact - binned), 0.3)


if __name__ == "__main__":
    unittest.main()
