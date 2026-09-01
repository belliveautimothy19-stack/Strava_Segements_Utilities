"""Alignment band tradeoff, and whether the score tracks shape."""

import numpy as np

import audit7.corpus as C
from audit7.independent import auc_lower_is_better, bootstrap_ci
from audit7.run_physical import score_pairs
from bench.synth import gain_matched_staircase, uniform_climb
from segmatch.match import MatchConfig, match_segment, prepare_target

BANDS = (0.03, 0.06, 0.10, 0.18, 0.30)


def band_staircase(band, length=6000.0, pitch=60.0):
    """Does widening the band erode the structure discrimination that
    resolution provides? Audit 2 claimed the band protected against the
    staircase; audit 6 showed it does not. Re-checked here independently."""
    d, e = gain_matched_staircase(length, total_gain_m=length * 0.05,
                                  pitch_m=pitch)
    d2, e2 = uniform_climb(length, grade=0.05)
    cfg = MatchConfig(max_shift_frac=band)
    return match_segment(d2, e2, prepare_target(d, e, cfg), cfg)[0].score


def band_real(band, win_m=1000.0, cap=600, seed=0):
    ws = C.build_windows(C.load_streams(), win_m, win_m / 4.0)
    pairs = C.categorize_geometric(ws)
    cfg_scored = _score_with_band(pairs, band, cap, seed)
    a = [r["score"] for r in cfg_scored if r["cat"] == "A"]
    n = [r["score"] for r in cfg_scored if r["cat"] == "N"]
    lo, hi = bootstrap_ci(a, n, seed=seed)
    return (auc_lower_is_better(a, n), lo, hi, float(np.median(a)),
            float(np.median(n)), float(np.min(n)), len(a), len(n))


def _score_with_band(pairs, band, cap, seed):
    rng = np.random.default_rng(seed)
    by = {}
    for i, p in enumerate(pairs):
        by.setdefault(p["cat"], []).append(i)
    keep = []
    for c, idx in by.items():
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.extend(int(i) for i in idx)
    cfg = MatchConfig(max_shift_frac=band)
    out = []
    for i in sorted(keep):
        p = pairs[i]
        t = prepare_target(p["A"]["d"], p["A"]["e"], cfg)
        if t is None:
            continue
        ms = match_segment(p["B"]["d"], p["B"]["e"], t, cfg)
        if ms:
            r = dict(p)
            r["score"] = ms[0].score
            out.append(r)
    return out


def monotonicity(win_m=1000.0, cap=1500, seed=0):
    """Does the score rise with an independently defined shape difference?

    The shape axis is the mean absolute difference between the two
    windows' normalized elevation profiles. It is not a grade sequence,
    it is not compared by dynamic time warping, and it is normalized by
    each window's own vertical range so it is blind to steepness. It
    therefore shares no machinery with the matcher and cannot be accused
    of restating it.

    Only geographically separate pairs are used, so physical identity
    cannot masquerade as shape agreement.
    """
    ws = C.build_windows(C.load_streams(), win_m, win_m / 4.0)
    pairs = [p for p in C.categorize_geometric(ws) if p["cat"] == "N"]
    rng = np.random.default_rng(seed)
    if len(pairs) > cap:
        pairs = [pairs[i] for i in
                 sorted(rng.choice(len(pairs), cap, replace=False))]
    scored = score_pairs(pairs, 70.0, cap_per_cat=cap, seed=seed)
    sd = np.array([r["shape_d"] for r in scored])
    sc = np.array([r["score"] for r in scored])
    edges = np.quantile(sd, np.linspace(0, 1, 9))
    rows = []
    for i in range(len(edges) - 1):
        m = (sd >= edges[i]) & (sd <= edges[i + 1])
        if m.sum() < 5:
            continue
        rows.append((float(edges[i]), float(edges[i + 1]), int(m.sum()),
                     float(np.median(sc[m]))))
    # Spearman without scipy
    r_s = _spearman(sd, sc)
    return rows, r_s, len(scored)


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def five_way(win_m=1000.0, cap=400, seed=0):
    """The five populations, labelled without the matcher and using only
    windows whose archetype label survives perturbation."""
    ws = C.build_windows(C.load_streams(), win_m, win_m / 4.0)
    stable = [w for w in ws if w["label_stable"]]
    pairs = C.categorize(stable)
    scored = score_pairs(pairs, 70.0, cap_per_cat=cap, seed=seed)
    out = {}
    for r in scored:
        out.setdefault(r["cat"], []).append(r["score"])
    return out, len(ws), len(stable)
