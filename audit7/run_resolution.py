"""
Challenge the 70 m default on every axis, not just the one that favours it.

The standing recommendation is to keep 70 m. It was reached from a single
synthetic probe, and a recommendation supported by one experiment is
exactly the pattern that has failed repeatedly in this project. Six axes
are measured here. If 70 m wins only on the synthetic probe, that has to
be stated plainly rather than presented as a general result.
"""

import time

import numpy as np

from audit7.corpus import build_windows, load_streams
from audit7.independent import auc_lower_is_better, bootstrap_ci
from bench.synth import gain_matched_staircase, uniform_climb
from segmatch.match import MatchConfig, match_segment, prepare_target

RESOLUTIONS = (50.0, 70.0, 90.0, 120.0, 150.0)


def structure_discrimination(res_m, pitches=(60.0, 90.0, 120.0, 180.0),
                             length=6000.0):
    """Separation between a staircase and a gain-matched ramp.

    Higher is better: it is the score the matcher gives two profiles that
    share length, gain and loss and differ only in ordered shape below
    the cycle. A resolution that cannot see the difference reports a
    small number, which in production means calling them the same hill.
    """
    out = {}
    for p in pitches:
        d, e = gain_matched_staircase(length, total_gain_m=length * 0.05,
                                      pitch_m=p)
        d2, e2 = uniform_climb(length, grade=0.05)
        cfg = MatchConfig(res_m=res_m)
        m = match_segment(d2, e2, prepare_target(d, e, cfg), cfg)[0]
        out[p] = m.score
    return out


def noise_sensitivity(res_m, sigma=0.6, n=40, seed=0, win_m=1500.0):
    """Self-match score after adding barometric-scale noise.

    Lower is better. A fine resolution that chases noise pays here.
    """
    rng = np.random.default_rng(seed)
    ws = build_windows(load_streams(), win_m, win_m)
    cfg = MatchConfig(res_m=res_m)
    out = []
    for w in ws[:n]:
        t = prepare_target(w["d"], w["e"], cfg)
        if t is None:
            continue
        ms = match_segment(w["d"], w["e"] + rng.normal(0, sigma, len(w["e"])),
                           t, cfg)
        if ms:
            out.append(ms[0].score)
    return float(np.median(out)) if out else float("nan")


def quantization_sensitivity(res_m, step=1.0, n=40, win_m=1500.0):
    """Self-match score after rounding elevation to a coarse step.

    Many devices report elevation to the nearest metre or foot. A
    resolution too fine to average over the rounding staircase will read
    it as terrain.
    """
    ws = build_windows(load_streams(), win_m, win_m)
    cfg = MatchConfig(res_m=res_m)
    out = []
    for w in ws[:n]:
        t = prepare_target(w["d"], w["e"], cfg)
        if t is None:
            continue
        ms = match_segment(w["d"], np.round(w["e"] / step) * step, t, cfg)
        if ms:
            out.append(ms[0].score)
    return float(np.median(out)) if out else float("nan")


def sampling_density_robustness(res_m, n=40, win_m=1500.0):
    """Self-match score after decimating to every third sample.

    Lower is better, but this axis is the one that most easily misleads:
    a coarse resolution scores well partly because it sees less to
    disagree about. It is reported next to structure_discrimination for
    that reason, never alone.
    """
    ws = build_windows(load_streams(), win_m, win_m)
    cfg = MatchConfig(res_m=res_m)
    out = []
    for w in ws[:n]:
        t = prepare_target(w["d"], w["e"], cfg)
        if t is None:
            continue
        ms = match_segment(w["d"][::3], w["e"][::3], t, cfg)
        if ms:
            out.append(ms[0].score)
    return float(np.median(out)) if out else float("nan")


def runtime_ms(res_m, win_m=6000.0, reps=3):
    ws = build_windows(load_streams(), win_m, win_m)
    if not ws:
        return float("nan")
    cfg = MatchConfig(res_m=res_m)
    w = ws[0]
    t = prepare_target(w["d"], w["e"], cfg)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        match_segment(w["d"], w["e"], t, cfg)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def real_data_accuracy(res_m, win_m=1000.0, cap=500, seed=0):
    """AUC on the geometric A/N split, with its interval."""
    from audit7.corpus import categorize_geometric
    from audit7.run_physical import score_pairs
    ws = build_windows(load_streams(), win_m, win_m / 4.0)
    pairs = categorize_geometric(ws)
    scored = score_pairs(pairs, res_m, cap_per_cat=cap, seed=seed)
    a = [r["score"] for r in scored if r["cat"] == "A"]
    n = [r["score"] for r in scored if r["cat"] == "N"]
    lo, hi = bootstrap_ci(a, n, seed=seed)
    return auc_lower_is_better(a, n), lo, hi, len(a), len(n)
