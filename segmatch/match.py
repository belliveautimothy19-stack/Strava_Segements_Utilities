"""
Window matching core.

SEMANTICS
=========
The old implementation left most of these undefined. They are stated here
because every one of them changes what "a match" means.

segment
    A candidate route, given as distance and elevation samples. It is
    converted once to a Profile (uniform arc-length grid, grade estimated
    at scale res_m) and never re-derived per window.

candidate match (window)
    A contiguous interval [s, s + L] of a segment, with the length ratio
    r = L / L_target inside [min_ratio, max_ratio]. Nothing outside that
    band of lengths is considered a match at all.

window generation
    Lengths: `length_steps` values spanning [min_ratio, max_ratio],
    always including exactly 1.0, each clamped to the segment length and
    then de-duplicated. Clamping rather than skipping matters: a segment
    a few metres shorter than the target would otherwise never be tried
    at its natural length.
    Offsets: a uniform grid of step `stride_frac` * L_target, always
    including both s = 0 and s = total - L so the head and tail of every
    segment are reachable.

alignment
    Both sequences are resampled to a common comparison length n_cmp and
    aligned by dynamic time warping under a Sakoe-Chiba band of
    `max_shift_frac` * n_cmp samples. The band is a physical tolerance: a
    feature may occur up to that fraction of the target length early or
    late and still match. Unbounded warping is what let the old code score
    0.000 between a 0.25 mi climb and a 1.25 mi one.

length differences
    Handled by resampling the window to n_cmp and charging an explicit
    len_dev = |L - L_target| / L_target term. Shape similarity and length
    similarity are therefore separate, named contributions rather than
    being tangled together inside the shape distance.

comparison representation
    Ordered grade sequences, never histograms. A histogram can screen
    candidates but cannot establish that two ordered profiles agree; two
    segments with identical grade composition in opposite order have
    identical histograms and completely different shapes.

tolerances
    Every threshold is a named field of MatchConfig with a default chosen
    by the sweep in bench/optimize.py, not by intuition.

boundary conditions
    Offsets include both endpoints (above). Grade near a profile's ends is
    computed from a truncated but honest regression window rather than
    zero-padded, which would fabricate a large slope at both ends.

overlapping windows
    Suppressed greedily. Windows are considered in ascending score, and
    one is rejected if its overlap with an already-accepted window exceeds
    `max_overlap` as a fraction of the shorter window.

duplicate matches
    A direct consequence of overlap suppression: a repeated pattern
    occurring three times in a route yields three separate accepted
    windows, while three near-identical offsets of the same occurrence
    collapse to one.

partial matches
    A window shorter than the target is a legitimate match carrying a
    len_dev penalty. Its length ratio is reported so the caller can tell a
    full match from a partial one.

noisy measurements
    Absorbed by the representation: grade is an OLS slope over res_m
    rather than a two-endpoint secant, and gain is measured at a fixed
    spatial interval.

final score
    score = w_shape * shape + w_dist * dist + w_gain * gain_dev
            + w_len * len_dev
    shape and dist are in percent-grade units; gain_dev and len_dev are
    dimensionless fractions, which is why their weights are larger.

EXACTNESS
=========
The search returns the true best `pool_size` windows under this scoring
function. Windows are skipped only when their Keogh lower bound already
exceeds the worst score currently in the pool, which by admissibility of
the bound means their exact score would have too. Pruning changes runtime,
never the result. test_matching.py asserts this by running with pruning
disabled and comparing.

Complexity: O(W * n_cmp) for screening plus O(P * n_cmp * band) for the
P windows that survive it, where W is the number of windows and P is
typically a few percent of W.
"""

import heapq
import numpy as np

from .profile import Profile, build_profile, vertical_change
from .distance import (dtw_band, keogh_envelope, lb_keogh, wasserstein1,
                        grade_histogram, hist_distance)

__all__ = ["MatchConfig", "Match", "TargetSpec", "prepare_target",
           "match_segment", "null_scores", "comparison_length"]

# Minimum vertical (metres) used as the gain-term denominator, so a flat
# target cannot drive it to a division by zero or make it inert. A target
# with under this much total vertical is flat for training purposes, and
# windows are then judged on how little vertical they have.
MIN_VERT_DENOM_M = 10.0


class MatchConfig:
    """Every tolerance and weight the matcher uses.

    Defaults were selected by the sweep in bench/optimize.py over a
    labelled synthetic dataset. See PARAMETERS.md for the measurements
    behind each one. In summary:

      res_m 120        grade resolution. Swept from 25 m to 600 m; F1
                       peaks on a 90 to 120 m plateau (0.955) against
                       0.903 at the old 0.25 mi / 402 m value, driven by
                       false positives roughly halving. 120 is the
                       coarsest value on that plateau.
      max_shift_frac   0.03. Tighter alignment tolerance was both more
                       accurate and faster than 0.05, 0.10 or 0.18.
      dist_bin_w 0     no binning. Histogram bin widths from 0.25 to 4.0
                       percent were indistinguishable from each other and
                       from the exact binless distance at every
                       resolution, so the parameter was removed rather
                       than guessed.
      w_shape/w_dist/w_gain  1.0 / 0.6 / 4.0, unchanged. The sweep found
                       no significant difference across the ranges tried,
                       so the existing values were kept.
      w_len 2.0        smallest weight that makes the reported window
                       land on the true extent rather than a clipped one.
    """

    __slots__ = ("res_m", "oversample", "min_ratio", "max_ratio",
                 "length_steps", "stride_frac", "max_shift_frac",
                 "w_shape", "w_dist", "w_gain", "w_len", "vert_resample_m",
                 "max_overlap", "pool_size", "top_k", "use_pruning",
                 "dist_bin_w")

    def __init__(self, res_m=120.0, oversample=4, min_ratio=0.75,
                 max_ratio=1.15, length_steps=7, stride_frac=0.02,
                 max_shift_frac=0.03, w_shape=1.0, w_dist=0.6, w_gain=4.0,
                 w_len=2.0, vert_resample_m=25.0, max_overlap=0.5,
                 pool_size=16, top_k=1, use_pruning=True,
                 dist_bin_w=0.0):
        self.res_m = float(res_m)
        self.oversample = int(oversample)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.length_steps = int(length_steps)
        self.stride_frac = float(stride_frac)
        self.max_shift_frac = float(max_shift_frac)
        self.w_shape = float(w_shape)
        self.w_dist = float(w_dist)
        self.w_gain = float(w_gain)
        self.w_len = float(w_len)
        self.vert_resample_m = float(vert_resample_m)
        self.max_overlap = float(max_overlap)
        self.pool_size = int(pool_size)
        self.top_k = int(top_k)
        self.use_pruning = bool(use_pruning)
        # Bin width in grade percent for the composition term. 0 means the
        # exact binless Wasserstein-1, which is the default because no bin
        # width beat it in the sweep. Any positive value switches to a
        # histogram earth mover's distance at that resolution.
        self.dist_bin_w = float(dist_bin_w)

    def replace(self, **kw):
        vals = {s: getattr(self, s) for s in self.__slots__}
        vals.update(kw)
        return MatchConfig(**vals)


class Match:
    """One accepted window."""

    __slots__ = ("score", "shape", "dist", "gain_dev", "len_dev",
                 "start_m", "end_m", "direction", "length_ratio",
                 "gain_m", "loss_m", "grade_seq")

    def __init__(self, score, shape, dist, gain_dev, len_dev, start_m,
                 end_m, direction, length_ratio, gain_m, loss_m, grade_seq):
        self.score = score
        self.shape = shape
        self.dist = dist
        self.gain_dev = gain_dev
        self.len_dev = len_dev
        self.start_m = start_m
        self.end_m = end_m
        self.direction = direction
        self.length_ratio = length_ratio
        self.gain_m = gain_m
        self.loss_m = loss_m
        self.grade_seq = grade_seq

    def __repr__(self):
        return (f"Match(score={self.score:.3f}, {self.start_m:.0f}-"
                f"{self.end_m:.0f}m, {self.direction}, "
                f"r={self.length_ratio:.2f})")


class TargetSpec:
    """Everything about the target that is reused across every candidate.

    Built once. This is the "avoid repeatedly recomputing normalized
    representations" requirement made structural: nothing here can be
    rebuilt per window even by accident.
    """

    __slots__ = ("profile", "length_m", "seq", "n_cmp", "band", "env_up",
                 "env_dn", "gain_m", "loss_m", "vert_denom", "cfg", "hist")

    def __init__(self, profile, length_m, seq, n_cmp, band, env_up, env_dn,
                 gain_m, loss_m, vert_denom, cfg, hist=None):
        self.profile = profile
        self.length_m = length_m
        self.seq = seq
        self.n_cmp = n_cmp
        self.band = band
        self.env_up = env_up
        self.env_dn = env_dn
        self.gain_m = gain_m
        self.loss_m = loss_m
        self.vert_denom = vert_denom
        self.cfg = cfg
        self.hist = hist


def comparison_length(length_m, res_m):
    """Number of samples used for shape comparison.

    Two samples per res_m. The representation carries no detail finer than
    res_m by construction, so sampling faster than Nyquist adds cost and
    no information. Bounded below so very short targets still have enough
    samples for a warping path to mean anything, and above so a very long
    target cannot make the DTW quadratic in route length.
    """
    return int(min(512, max(8, round(length_m / (res_m / 2.0)))))


def prepare_target(cum_dist, elev, cfg):
    """Build the reusable target representation. Returns None if the
    target is too short to represent at cfg.res_m."""
    prof = build_profile(cum_dist, elev, cfg.res_m, cfg.oversample,
                         cfg.vert_resample_m)
    if prof is None:
        return None
    length_m = prof.length
    n_cmp = comparison_length(length_m, cfg.res_m)
    seq = prof.grade_at(np.linspace(0.0, length_m, n_cmp))
    band = max(1, int(round(cfg.max_shift_frac * n_cmp)))
    up, dn = keogh_envelope(seq, band)
    gain, loss = vertical_change(cum_dist, elev, cfg.vert_resample_m)
    denom = max(gain + loss, MIN_VERT_DENOM_M)
    hist = (grade_histogram(seq, cfg.dist_bin_w)
            if cfg.dist_bin_w > 0 else None)
    return TargetSpec(prof, length_m, seq, n_cmp, band, up, dn, gain, loss,
                      denom, cfg, hist)


def _window_lengths(target_len, total_len, cfg):
    fr = np.linspace(cfg.min_ratio, cfg.max_ratio, cfg.length_steps)
    if cfg.min_ratio <= 1.0 <= cfg.max_ratio:
        fr = np.append(fr, 1.0)
    lens = np.minimum(target_len * fr, total_len)
    return np.unique(np.round(lens, 6))


def _window_starts(total_len, win_len, target_len, cfg):
    step = max(target_len * cfg.stride_frac, 10.0)
    span = total_len - win_len
    if span <= 1e-9:
        return np.array([0.0])
    starts = np.arange(0.0, span + 1e-9, step)
    if starts[-1] < span - 1e-9:
        starts = np.append(starts, span)
    return starts


def _overlap_frac(a, b):
    """Overlap of two intervals as a fraction of the shorter one."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (hi - lo) / shorter if shorter > 0 else 0.0


def _score_window(prof, target, cfg, start, win_len, direction, unit):
    """Score one (start, length, direction) triple. Returns a tuple of
    (score, shape, dist, gain_dev, len_dev, gain, loss, seq)."""
    end = start + win_len
    grades = prof.grade_at(start + unit * win_len)
    gain, loss = prof.window_vertical(start, end)
    if direction == "forward":
        seq, g_dir, l_dir = grades, gain, loss
    else:
        seq, g_dir, l_dir = -grades[::-1], loss, gain
    len_dev = abs(win_len - target.length_m) / target.length_m
    gain_dev = ((abs(g_dir - target.gain_m) + abs(l_dir - target.loss_m))
                / target.vert_denom)
    shape = dtw_band(target.seq, seq, target.band)
    if cfg.dist_bin_w > 0:
        dist = hist_distance(target.hist,
                             grade_histogram(seq, cfg.dist_bin_w),
                             cfg.dist_bin_w)
    else:
        dist = wasserstein1(target.seq, seq)
    score = (cfg.w_shape * shape + cfg.w_dist * dist
             + cfg.w_gain * gain_dev + cfg.w_len * len_dev)
    return score, shape, dist, gain_dev, len_dev, g_dir, l_dir, seq


def _refine(prof, target, cfg, start, win_len, direction, unit, total_len):
    """Local coordinate descent on (start, length) around a grid winner.

    The offset and length grids exist to generate candidates cheaply, but
    they quantize the answer, and the score is genuinely sensitive to that
    quantization: because DTW warps by whole samples it cannot represent a
    fractional stretch, so a window even 0.5 percent off the target length
    carries a shape cost of about 0.45 that no band width removes. With
    seven trial lengths spanning 0.75 to 1.15, neighbouring lengths differ
    by 5.7 percent, which leaves real score on the table for a true match.

    Rather than pay for a much denser grid everywhere, the grid stays
    coarse and the few best candidates are refined here by halving steps.
    Roughly 24 extra evaluations per refined window, against thousands
    screened. The returned window is therefore NOT quantized to the grid.
    """
    step_s = max(target.length_m * cfg.stride_frac * 0.5, 1.0)
    span = cfg.max_ratio - cfg.min_ratio
    step_l = max(target.length_m * span / max(1, cfg.length_steps - 1)
                 * 0.5, 1.0)
    best = _score_window(prof, target, cfg, start, win_len, direction,
                         unit)
    lo_len = target.length_m * cfg.min_ratio
    hi_len = target.length_m * cfg.max_ratio

    for _ in range(7):
        improved = False
        for d_s, d_l in ((-step_s, 0.0), (step_s, 0.0),
                         (0.0, -step_l), (0.0, step_l)):
            ns = start + d_s
            nl = win_len + d_l
            if nl < lo_len or nl > hi_len or nl > total_len:
                continue
            if ns < 0.0 or ns + nl > total_len:
                continue
            cand = _score_window(prof, target, cfg, ns, nl, direction,
                                 unit)
            if cand[0] < best[0] - 1e-12:
                best, start, win_len, improved = cand, ns, nl, True
        if not improved:
            step_s *= 0.5
            step_l *= 0.5
    return best, start, win_len


def match_segment(seg_dist, seg_elev, target, cfg=None, profile=None):
    """Find the best non-overlapping windows of one segment against the
    target.

    Returns a list of Match, ascending by score, at most cfg.top_k long.
    Empty if the segment cannot hold any window in the allowed length
    range.
    """
    cfg = cfg or target.cfg
    # Callers that need the profile for other work (a null model, say) can
    # build it once and hand it in rather than paying for it twice.
    prof = profile if profile is not None else build_profile(
        seg_dist, seg_elev, cfg.res_m, cfg.oversample, cfg.vert_resample_m)
    if prof is None:
        return []
    total_len = prof.length
    if total_len < target.length_m * cfg.min_ratio:
        return []

    n_cmp = target.n_cmp
    tgt_seq = target.seq
    band = target.band
    unit = np.linspace(0.0, 1.0, n_cmp)

    # Bounded max-heap of the best pool_size windows, keyed by -score so
    # heap[0] is the worst kept. Pruning compares against that worst.
    # The pool must hold at least the top_k that will survive overlap
    # suppression, with headroom for the ones it discards. A smaller pool
    # prunes harder (the bound compares against the pool's worst), so this
    # is kept as tight as correctness allows.
    pool_size = max(cfg.pool_size, 8 * cfg.top_k)
    pool = []
    worst = float("inf")
    n_eval = 0
    n_pruned = 0

    for win_len in _window_lengths(target.length_m, total_len, cfg):
        ratio = win_len / target.length_m
        len_dev = abs(win_len - target.length_m) / target.length_m
        for start in _window_starts(total_len, win_len, target.length_m,
                                    cfg):
            end = start + win_len
            grades = prof.grade_at(start + unit * win_len)
            gain, loss = prof.window_vertical(start, end)

            for direction in ("forward", "reverse"):
                if direction == "forward":
                    seq = grades
                    g_dir, l_dir = gain, loss
                else:
                    seq = -grades[::-1]
                    g_dir, l_dir = loss, gain

                # Direction-aware gain term. Comparing signed gain and
                # loss separately, rather than max(gain, loss), means a
                # rolling window with 300 m up and 300 m down no longer
                # scores identically to a pure 300 m climb, and the term
                # now discriminates direction instead of being invariant
                # to it by construction.
                gain_dev = ((abs(g_dir - target.gain_m)
                             + abs(l_dir - target.loss_m))
                            / target.vert_denom)

                fixed = (cfg.w_gain * gain_dev + cfg.w_len * len_dev)

                if cfg.use_pruning and len(pool) >= pool_size:
                    lb = lb_keogh(seq, target.env_up, target.env_dn, n_cmp)
                    if cfg.w_shape * lb + fixed >= worst:
                        n_pruned += 1
                        continue

                shape = dtw_band(tgt_seq, seq, band)
                if cfg.dist_bin_w > 0:
                    dist = hist_distance(
                        target.hist,
                        grade_histogram(seq, cfg.dist_bin_w),
                        cfg.dist_bin_w)
                else:
                    dist = wasserstein1(tgt_seq, seq)
                score = cfg.w_shape * shape + cfg.w_dist * dist + fixed
                n_eval += 1

                item = (-score, start, end, direction, shape, dist,
                        gain_dev, len_dev, ratio, g_dir, l_dir, seq)
                if len(pool) < pool_size:
                    heapq.heappush(pool, item)
                elif score < worst:
                    heapq.heapreplace(pool, item)
                if len(pool) >= pool_size:
                    worst = -pool[0][0]

    if not pool:
        return []

    # Deterministic ordering: score, then start, then direction.
    ranked = sorted(pool, key=lambda it: (-it[0], it[1], it[3]))

    accepted = []
    for it in ranked:
        span = (it[1], it[2])
        if any(_overlap_frac(span, (a.start_m, a.end_m)) > cfg.max_overlap
               for a in accepted):
            continue
        res, rs, rl = _refine(prof, target, cfg, it[1], it[2] - it[1],
                              it[3], unit, total_len)
        (score, shape, dist, gain_dev, len_dev, g_dir, l_dir, seq) = res
        accepted.append(Match(score, shape, dist, gain_dev, len_dev, rs,
                              rs + rl, it[3], rl / target.length_m,
                              g_dir, l_dir, seq))
        if len(accepted) >= cfg.top_k:
            break
    # Refinement can reorder the accepted windows, so sort once more.
    accepted.sort(key=lambda m: (m.score, m.start_m))
    return accepted


def null_scores(profiles, target, cfg, n=240, seed=0):
    """Scores of randomly chosen windows drawn from the fetched segments.

    This is the reference distribution the matcher previously lacked. The
    search always returns a best window for every segment, so a ranked
    list looks equally authoritative whether the area holds a twin of the
    target or nothing resembling it. Sampling windows at random from the
    same candidate pool gives an empirical distribution of what "no
    relationship" scores like, against which a real match can be placed.

    Random windows are used rather than each segment's best window,
    because the best window of a segment is by construction the extreme
    of its own distribution and would give an optimistically low null.

    Returns a sorted array of scores (possibly shorter than n if the
    candidate pool is small).
    """
    rng = np.random.default_rng(seed)
    usable = [p for p in profiles
              if p is not None and p.length >= target.length_m * cfg.min_ratio]
    if not usable:
        return np.array([])
    unit = np.linspace(0.0, 1.0, target.n_cmp)
    lo, hi = cfg.min_ratio, cfg.max_ratio
    out = []
    for _ in range(int(n)):
        prof = usable[int(rng.integers(len(usable)))]
        win_len = min(target.length_m * rng.uniform(lo, hi), prof.length)
        start = rng.uniform(0.0, max(0.0, prof.length - win_len))
        direction = "forward" if rng.random() < 0.5 else "reverse"
        out.append(_score_window(prof, target, cfg, start, win_len,
                                 direction, unit)[0])
    return np.sort(np.asarray(out))
