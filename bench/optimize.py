"""
Parameter sweep and Pareto selection.

Runs the evaluation over a grid of parameter settings and reports the
Pareto-efficient region on (discrimination, runtime). Selection rule,
applied in this order:

  1. Discard settings whose AUC is below the best AUC minus a tolerance
     that reflects evaluation noise, rather than chasing the maximum.
  2. Among the survivors, prefer the coarsest resolution, since coarser
     is cheaper, more robust to noise, and less prone to overfitting the
     evaluation set.
  3. Break remaining ties on runtime.

This deliberately picks the simplest setting on the performance plateau
instead of the single best-scoring one.
"""

import itertools
import multiprocessing as mp
import numpy as np

from segmatch.match import MatchConfig
from bench.evaluate import build_dataset, evaluate

_DS = None


def _init(seed, n_targets):
    global _DS
    _DS = build_dataset(seed=seed, n_targets=n_targets)


def _run(kw):
    cfg = MatchConfig(**kw)
    r = evaluate(cfg, _DS)
    r.update(kw)
    return r


def sweep(grid, seed=20240501, n_targets=4, processes=4):
    """grid: dict of parameter name -> list of values. Returns list of
    result dicts, one per combination."""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(
        *(grid[k] for k in keys))]
    with mp.Pool(processes, initializer=_init,
                 initargs=(seed, n_targets)) as pool:
        return pool.map(_run, combos)


def pareto(results, auc_tol=0.005):
    """Return (selected, frontier). Selection favours the coarsest
    res_m within auc_tol of the best AUC."""
    best_auc = max(r["auc"] for r in results)
    ok = [r for r in results if r["auc"] >= best_auc - auc_tol]
    ok.sort(key=lambda r: (-r.get("res_m", 0.0), r["runtime_s"]))
    frontier = []
    for r in sorted(results, key=lambda r: r["runtime_s"]):
        if not frontier or r["auc"] > frontier[-1]["auc"]:
            frontier.append(r)
    return ok[0], frontier


def report(results, keys, sort_by="auc"):
    hdr = "".join(f"{k:>16}" for k in keys)
    print(f"{hdr}{'AUC':>8}{'F1':>7}{'prec':>7}{'rec':>7}{'FPR':>7}"
          f"{'FNR':>7}{'loc':>7}{'ms':>8}")
    for r in sorted(results, key=lambda r: -r[sort_by]):
        vals = "".join(f"{r[k]:>16.4g}" for k in keys)
        print(f"{vals}{r['auc']:>8.4f}{r['f1']:>7.3f}{r['precision']:>7.3f}"
              f"{r['recall']:>7.3f}{r['fpr']:>7.3f}{r['fnr']:>7.3f}"
              f"{r['localization']:>7.3f}{r['ms_per_match']:>8.1f}")
