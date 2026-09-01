"""Experiment A: can the matcher tell the same ground from different
ground? Labels come from geometry alone."""

import json
import sys

import numpy as np

from audit7.corpus import (assert_sample_adequate, assert_no_population_leakage,
                           build_windows, categorize_geometric, load_streams,
                           provenance)
from audit7.independent import auc_lower_is_better, bootstrap_ci
from segmatch.match import MatchConfig, match_segment, prepare_target


def score_pairs(pairs, res_m, cap_per_cat=700, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i, p in enumerate(pairs):
        by.setdefault(p["cat"], []).append(i)
    keep = []
    for c, idx in by.items():
        if len(idx) > cap_per_cat:
            idx = rng.choice(idx, cap_per_cat, replace=False)
        keep.extend(int(i) for i in idx)
    cfg = MatchConfig(res_m=res_m)
    out = []
    for i in sorted(keep):
        p = pairs[i]
        t = prepare_target(p["A"]["d"], p["A"]["e"], cfg)
        if t is None:
            continue
        ms = match_segment(p["B"]["d"], p["B"]["e"], t, cfg)
        if not ms:
            continue
        r = dict(p)
        r["score"] = ms[0].score
        out.append(r)
    return out


def report(win_m, res_m=70.0, stride_frac=0.25, seed=0):
    ws = build_windows(load_streams(), win_m, win_m * stride_frac)
    pairs = categorize_geometric(ws)
    assert_no_population_leakage(pairs)
    scored = score_pairs(pairs, res_m, seed=seed)
    a = [r["score"] for r in scored if r["cat"] == "A"]
    n = [r["score"] for r in scored if r["cat"] == "N"]
    a_cross = [r["score"] for r in scored
               if r["cat"] == "A" and r["cross_route"]]
    a_same = [r["score"] for r in scored
              if r["cat"] == "A" and not r["cross_route"]]
    auc = auc_lower_is_better(a, n)
    lo, hi = bootstrap_ci(a, n, seed=seed)
    adeq = assert_sample_adequate("AUC(A|N)", lo, hi, 0.05, n_pos=len(a))
    res = {
        "win_m": win_m, "res_m": res_m,
        "n_windows": len(ws), "n_routes": len({w["route"] for w in ws}),
        "n_A": len(a), "n_N": len(n),
        "n_A_cross_recording": len(a_cross), "n_A_same_recording": len(a_same),
        "auc": auc, "ci": [lo, hi], "adequate": adeq,
        "auc_cross_only": auc_lower_is_better(a_cross, n) if a_cross else None,
        "auc_same_only": auc_lower_is_better(a_same, n) if a_same else None,
        "median_A": float(np.median(a)) if a else None,
        "median_N": float(np.median(n)) if n else None,
        "p90_A": float(np.quantile(a, 0.90)) if a else None,
        "p10_N": float(np.quantile(n, 0.10)) if n else None,
        "nearest_negative": float(np.min(n)) if n else None,
        "prov_A": {"%s x %s" % k: v
                   for k, v in provenance(scored, "A").items()},
        "n_route_pairings_A": len(provenance(scored, "A")),
    }
    return res


if __name__ == "__main__":
    out = []
    for L in [float(x) for x in sys.argv[1:]] or [1000.0]:
        r = report(L)
        out.append(r)
        print("L=%5.0f  windows=%3d/%dr  A=%4d (cross %3d, same %3d)  N=%4d"
              % (L, r["n_windows"], r["n_routes"], r["n_A"],
                 r["n_A_cross_recording"], r["n_A_same_recording"], r["n_N"]))
        print("    AUC(A|N)=%.4f  95%% CI [%.4f, %.4f]  width %.4f  %s"
              % (r["auc"], r["ci"][0], r["ci"][1],
                 r["adequate"]["ci_width"],
                 "ADEQUATE" if r["adequate"]["adequate"] else "TOO WIDE"))
        print("    cross-recording only AUC=%s   same-recording only AUC=%s"
              % (round(r["auc_cross_only"], 4) if r["auc_cross_only"] else "-",
                 round(r["auc_same_only"], 4) if r["auc_same_only"] else "-"))
        print("    median A=%.3f  N=%.3f   p90(A)=%.3f  p10(N)=%.3f  "
              "nearest negative=%.3f"
              % (r["median_A"], r["median_N"], r["p90_A"], r["p10_N"],
                 r["nearest_negative"]))
        print("    A from %d route pairing(s): %s"
              % (r["n_route_pairings_A"], r["prov_A"]))
        sys.stdout.flush()
    json.dump(out, open("audit7/physical.json", "w"), indent=1)
