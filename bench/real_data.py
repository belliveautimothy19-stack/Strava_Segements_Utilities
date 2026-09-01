"""
Validation against real route data.

Everything in bench/evaluate.py is synthetic. This module is the bridge to
real terrain, and it is deliberately built so that it needs no labels: it
derives ground truth from the data itself, so it can run against whatever
the tool has already cached without anyone hand-annotating matches.

RUN IT

    python3 -m bench.real_data --cache
    python3 -m bench.real_data --gpx-dir ./routes
    python3 -m bench.real_data --cache --res-sweep

--cache reads ~/.strava_segment_matcher_cache/streams, which
find_similar_segments.py populates as a side effect of ordinary use. No
credentials are read and no network calls are made.

WHAT IT ESTABLISHES

  spectrum      How much grade energy real terrain carries below 100 m.
                This is the single most valuable measurement here,
                because the resolution parameter was selected on
                synthetic terrain whose roughness spread across scales
                was assumed rather than measured. If real terrain is
                smoother than assumed, a coarser res_m is safe; if
                rougher, res_m must come down. The recommendation is
                printed.

  quantization  The elevation step in real streams. Coarse rounding
                manufactures staircase structure near the grade
                resolution and is the dominant false-negative risk at
                fine res_m.

  self_match    Every segment must match itself at essentially zero.
                Catches any asymmetry or representation bug that
                synthetic data happens not to excite.

  subwindow     Real ground truth without labels: cut a known interval
                out of a real segment, use it as the target, and require
                the matcher to find it back in the full segment. Real
                terrain, real sampling, real noise, and the answer is
                known by construction.

  robustness    The same real segment re-recorded: decimated, jittered,
                quantized, reversed. All must still match.

  cross_null    All-pairs scores across real segments give a real null
                distribution, and the gap between self-match and
                cross-match scores is the real separation the synthetic
                AUC only estimates.

WHAT IT DOES NOT ESTABLISH

  Whether two DIFFERENT real segments that a runner would call "the same
  kind of hill" score close together. That needs human labels and cannot
  be derived from the data. Everything here measures self-consistency,
  localization and separation, which is necessary but not sufficient.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from segmatch.match import MatchConfig, prepare_target, match_segment
from segmatch.profile import build_profile, detect_quantization

CACHE_DIR = Path.home() / ".strava_segment_matcher_cache" / "streams"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_cache(cache_dir=CACHE_DIR, limit=None):
    """Load (name, dist, elev) from the tool's own stream cache."""
    out = []
    if not Path(cache_dir).is_dir():
        return out
    for path in sorted(Path(cache_dir).glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("_miss") or "distance" not in data:
            continue
        d = np.asarray(data["distance"], dtype=float)
        e = np.asarray(data["altitude"], dtype=float)
        if d.size < 20 or not np.isfinite(e).any():
            continue
        out.append((path.stem, d, e))
        if limit and len(out) >= limit:
            break
    return out


def load_gpx_dir(directory, limit=None):
    """Load (name, dist, elev) from a directory of GPX files."""
    import gpxpy
    from find_similar_segments import haversine_m
    out = []
    for path in sorted(Path(directory).glob("*.gpx")):
        try:
            with open(path) as f:
                gpx = gpxpy.parse(f)
        except Exception:
            continue
        pts = [p for t in gpx.tracks for s in t.segments for p in s.points]
        if len(pts) < 20:
            continue
        d = [0.0]
        for i in range(1, len(pts)):
            d.append(d[-1] + haversine_m(pts[i - 1].latitude,
                                          pts[i - 1].longitude,
                                          pts[i].latitude,
                                          pts[i].longitude))
        e = [p.elevation for p in pts]
        e = np.asarray([np.nan if v is None else v for v in e], dtype=float)
        if not np.isfinite(e).any():
            continue
        out.append((path.stem, np.asarray(d, dtype=float), e))
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Audits
# --------------------------------------------------------------------------

def grade_energy_by_scale(dist, elev, bands=(400.0, 200.0, 100.0, 50.0)):
    """Fraction of grade variance below each wavelength.

    Only bands the stream can actually resolve are reported. This matters
    more than it sounds: an earlier version of this function interpolated
    every stream onto a fixed 5 m grid regardless of its native sampling,
    which manufactures high-frequency content out of linear interpolation.
    A stream sampled every 42 m carries no information below 84 m by
    Nyquist, but interpolating it to 5 m and running an FFT happily
    reports energy there, all of it artifact. That would have biased the
    single measurement this whole audit turns on.

    Returns (fractions, grade_sd, length, native_spacing, resolvable_limit)
    where fractions omits any band below the resolvable limit, or None if
    the stream is unusable.
    """
    d = np.asarray(dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    good = np.isfinite(e)
    if good.sum() < 16:
        return None
    e = np.interp(d, d[good], e[good])
    # Drop stalled samples before measuring spacing; a stationary recorder
    # emits repeated distances that would understate the real interval.
    step = np.diff(d)
    moving = step[step > 1e-9]
    if moving.size < 16:
        return None
    spacing = float(np.median(moving))
    total = float(d[-1] - d[0])
    if total < 400.0:
        return None
    # Resolve no finer than 2.5 sample intervals; below that the FFT is
    # reporting the interpolator, not the terrain.
    limit = 2.5 * spacing
    dx = max(spacing, 2.0)
    n = int(total / dx) + 1
    if n < 32:
        return None
    grid = np.linspace(d[0], d[-1], n)
    ge = np.interp(grid, d, e)
    g = np.gradient(ge, dx) * 100.0
    sp = np.abs(np.fft.rfft(g - g.mean())) ** 2
    k = np.arange(sp.size)
    wl = np.where(k > 0, total / np.maximum(k, 1), np.inf)
    tot = sp[1:].sum()
    if tot <= 0:
        return None
    fr = {b: float(sp[1:][wl[1:] < b].sum() / tot)
          for b in bands if b >= limit}
    return fr, float(np.std(g)), total, spacing, limit


def resolution_noise_budget(dist, elev, values=(150.0, 120.0, 90.0, 70.0,
                                                  50.0, 35.0, 25.0)):
    """Measure, per resolution, how much of the represented grade is real
    terrain and how much is elevation-rounding artifact.

    This replaces an earlier recommendation that keyed off raw sub-100 m
    grade energy, which is not a safe basis: rounding manufactures exactly
    that energy. Measured on a real 2 km slice of Boulder trail recorded
    at 0.1 m precision, quantizing it to 1 m took the apparent sub-100 m
    share from 37 percent to 80 percent and nearly doubled apparent
    roughness. A recommender reading that number would demand a finer
    resolution the coarser the source data got, which is backwards.

    So the quantity measured here is the ratio the choice actually turns
    on: perturb the profile by the rounding the provider already applied,
    and compare the resulting change in grade against the real grade
    variation at that scale.

    Returns a list of (res_m, terrain_sd, quant_err, ratio).
    """
    d = np.asarray(dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    good = np.isfinite(e)
    if good.sum() < 32:
        return []
    e = np.interp(d, d[good], e[good])
    # Probe by PERTURBING at the rounding scale, not by re-rounding.
    # Most served streams are already quantized, so re-applying the same
    # rounding is a no-op and measures nothing: an earlier version of this
    # reported a noise ratio of 0.00 at every resolution for exactly that
    # reason. Adding a uniform error of the same magnitude measures the
    # quantity that matters, namely how far the representation moves when
    # elevation is wrong by about one rounding step.
    #
    # The probe is at least 1 m because that is what the provider was
    # observed to deliver on most activities; a stream that happens to
    # carry finer precision still has to survive being compared against
    # ones that do not.
    step = detect_quantization(e)
    probe = max(step, 1.0)
    rng = np.random.default_rng(0)
    out = []
    for res in values:
        pn = build_profile(d, e, res, 8)
        pq = build_profile(d, e + rng.uniform(-probe / 2, probe / 2, e.size),
                           res, 8)
        if pn is None or pq is None:
            continue
        m = min(len(pn.grade), len(pq.grade))
        if m < 8:
            continue
        sd = float(np.std(pn.grade[:m]))
        err = float(np.std(pn.grade[:m] - pq.grade[:m]))
        out.append((res, sd, err, err / sd if sd > 0 else float("inf")))
    return out


def recommend_res_m(routes, max_noise_ratio=0.20):
    """Finest resolution whose rounding-induced grade error stays under
    max_noise_ratio of real grade variation, across the supplied routes.

    Finer than that and the representation is increasingly describing the
    provider's rounding rather than the hill.
    """
    per_res = {}
    for _, d, e in routes:
        for res, sd, err, ratio in resolution_noise_budget(d, e):
            per_res.setdefault(res, []).append(ratio)
    if not per_res:
        return None, {}
    med = {r: float(np.median(v)) for r, v in per_res.items()}
    ok = [r for r in sorted(med) if med[r] <= max_noise_ratio]
    return (min(ok) if ok else max(med)), med


def run_audits(routes, cfg=None, verbose=True):
    """Run every label-free protocol. Returns a results dict."""
    cfg = cfg or MatchConfig()
    rng = np.random.default_rng(0)
    res = {"n_routes": len(routes)}
    if not routes:
        return res

    def say(*a):
        if verbose:
            print(*a)

    # ---- spectrum --------------------------------------------------------
    fr100, sds, lens, spac, skipped = [], [], [], [], 0
    for name, d, e in routes:
        r = grade_energy_by_scale(d, e)
        if not r:
            continue
        fracs, sd, total, spacing, limit = r
        sds.append(sd); lens.append(total); spac.append(spacing)
        if 100.0 in fracs:
            fr100.append(fracs[100.0])
        else:
            skipped += 1
    if sds:
        say(f"\nSPECTRUM  ({len(sds)} routes)")
        say(f"  median grade sd            {np.median(sds):6.2f} %")
        say(f"  median route length        {np.median(lens):6.0f} m")
        say(f"  median sample spacing      {np.median(spac):6.1f} m")
        if skipped:
            say(f"  NOTE: {skipped} of {len(sds)} routes are sampled too "
                f"coarsely to resolve 100 m and are excluded from the "
                f"figure below rather than contributing interpolation "
                f"artifact to it.")
    if fr100:
        res["spectrum"] = {"median_frac_below_100m": float(np.median(fr100)),
                           "n_usable": len(fr100)}
        say(f"  grade energy below 100 m   {100*np.median(fr100):6.1f} %  "
            f"(from {len(fr100)} routes that can resolve it)")
        say(f"  per route: " + ", ".join(f"{100*f:.0f}%" for f in fr100))
        say("  NOTE: this figure is inflated by elevation rounding and must "
            "not be used on its own to pick a resolution. See the noise "
            "budget below, which is what the choice actually turns on.")
    else:
        say("  NO route is sampled finely enough to measure sub-100 m "
            "energy.")

    # ---- resolution noise budget ----------------------------------------
    rec, med = recommend_res_m(routes)
    if med:
        res["noise_budget"] = {"per_res": med, "recommended_res_m": rec}
        say(f"\nRESOLUTION NOISE BUDGET")
        say("  rounding-induced grade error as a fraction of real grade "
            "variation")
        say("  " + "  ".join(f"{int(r)}m:{med[r]:.2f}" for r in sorted(med,
                                                                reverse=True)))
        say(f"  finest resolution under a 0.20 noise ratio: {rec:.0f} m "
            f"(current default {cfg.res_m:.0f} m)")

    # ---- quantization ----------------------------------------------------
    steps = [detect_quantization(e) for _, _, e in routes]
    coarse = [s for s in steps if s >= 2.0]
    res["quantization"] = {"median_step_m": float(np.median(steps)),
                           "n_coarse": len(coarse)}
    say(f"\nQUANTIZATION")
    say(f"  median detected step       {np.median(steps):6.2f} m")
    say(f"  routes quantized >= 2 m    {len(coarse)} of {len(steps)}"
        + ("  (these risk false negatives at fine res_m)" if coarse else ""))

    # ---- self match, robustness, reversal --------------------------------
    def score_of(target, d, e):
        ms = match_segment(d, e, target, cfg)
        return ms[0].score if ms else float("inf")

    selfs, decim, jitter, quant, revs, rev_ok = [], [], [], [], [], 0
    for _, d, e in routes:
        t = prepare_target(d, e, cfg)
        if t is None:
            continue
        selfs.append(score_of(t, d, e))
        dd = np.arange(d[0], d[-1], 25.0)
        decim.append(score_of(t, dd, np.interp(dd, d, e)))
        jitter.append(score_of(t, d, e + rng.normal(0, 0.6, e.size)))
        quant.append(score_of(t, d, np.round(e)))
        ms = match_segment(d, e[::-1].copy(), t, cfg)
        if ms:
            revs.append(ms[0].score)
            rev_ok += int(ms[0].direction == "reverse")
    for key, vals in (("self_match", selfs), ("decimated_25m", decim),
                      ("jitter_0.6m", jitter), ("quantized_1m", quant),
                      ("reversed", revs)):
        if vals:
            res[key] = {"median": float(np.median(vals)),
                        "p90": float(np.percentile(vals, 90)),
                        "max": float(np.max(vals))}
    say(f"\nSELF-CONSISTENCY  (lower is better; self-match should be ~0)")
    for key in ("self_match", "decimated_25m", "jitter_0.6m",
                "quantized_1m", "reversed"):
        if key in res:
            r = res[key]
            say(f"  {key:<16} median {r['median']:7.3f}  p90 {r['p90']:7.3f}"
                f"  max {r['max']:7.3f}")
    if revs:
        res["reverse_direction_correct"] = rev_ok / len(revs)
        say(f"  reversed runs reported as 'reverse': "
            f"{rev_ok}/{len(revs)}")

    # ---- sub-window recovery: real ground truth --------------------------
    hits, ious = 0, []
    trials = 0
    for name, d, e in routes:
        total = float(d[-1] - d[0])
        if total < 1600.0:
            continue
        win = total * 0.5
        start = d[0] + rng.uniform(0.0, total - win)
        mask = (d >= start) & (d <= start + win)
        if mask.sum() < 20:
            continue
        t = prepare_target(d[mask] - d[mask][0], e[mask], cfg)
        if t is None:
            continue
        ms = match_segment(d, e, t, cfg)
        trials += 1
        if not ms:
            continue
        lo = max(ms[0].start_m + d[0], start)
        hi = min(ms[0].end_m + d[0], start + win)
        inter = max(0.0, hi - lo)
        union = (max(ms[0].end_m + d[0], start + win)
                 - min(ms[0].start_m + d[0], start))
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)
        hits += int(iou >= 0.5)
    if trials:
        res["subwindow"] = {"trials": trials, "iou_ge_0.5": hits / trials,
                            "median_iou": float(np.median(ious)) if ious else 0.0}
        say(f"\nSUB-WINDOW RECOVERY  (real ground truth, {trials} trials)")
        say(f"  localized with IoU >= 0.5  {hits}/{trials} "
            f"({100*hits/trials:.1f}%)")
        say(f"  median IoU                 {np.median(ious):.3f}")

    # ---- cross-segment null ---------------------------------------------
    cross = []
    for i, (_, d, e) in enumerate(routes[:40]):
        t = prepare_target(d, e, cfg)
        if t is None:
            continue
        for j, (_, d2, e2) in enumerate(routes[:40]):
            if i == j:
                continue
            cross.append(score_of(t, d2, e2))
    if cross and selfs:
        cross = np.asarray(cross)
        cross = cross[np.isfinite(cross)]
        res["cross_null"] = {"n": int(cross.size),
                             "p1": float(np.percentile(cross, 1)),
                             "median": float(np.median(cross))}
        say(f"\nCROSS-SEGMENT NULL  ({cross.size} pairs)")
        say(f"  self-match median          {np.median(selfs):7.3f}")
        say(f"  cross-match 1st percentile {np.percentile(cross,1):7.3f}")
        say(f"  cross-match median         {np.median(cross):7.3f}")
        # Floor the denominator at a score that is already
        # indistinguishable from a perfect match. Without it a self-match
        # median that rounds to zero produces a meaningless four-digit
        # ratio that looks like a spectacular result.
        floor = 0.01
        sep = np.percentile(cross, 1) / max(np.median(selfs), floor)
        capped = " (self-match at or below the 0.01 floor)" \
            if np.median(selfs) < floor else ""
        say(f"  separation ratio           {sep:7.1f}x{capped}")
        res["separation_ratio"] = float(sep)
    return res


def res_sweep(routes, values=(120.0, 90.0, 70.0, 50.0), cfg=None):
    """Re-run the self-consistency and sub-window audits per resolution.

    This is how a resolution is chosen on real data rather than assumed:
    the value that keeps self-consistency tight AND localizes sub-windows
    is the one the terrain supports.
    """
    print(f"\n{'res_m':>7}{'self med':>10}{'quant 1m':>10}{'decim':>9}"
          f"{'subwin IoU':>12}{'sep ratio':>11}")
    rows = []
    for v in values:
        c = (cfg or MatchConfig()).replace(res_m=v)
        r = run_audits(routes, c, verbose=False)
        rows.append((v, r))
        print(f"{v:>7.0f}{r.get('self_match',{}).get('median',float('nan')):>10.3f}"
              f"{r.get('quantized_1m',{}).get('median',float('nan')):>10.3f}"
              f"{r.get('decimated_25m',{}).get('median',float('nan')):>9.3f}"
              f"{r.get('subwindow',{}).get('median_iou',float('nan')):>12.3f}"
              f"{r.get('separation_ratio',float('nan')):>11.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache", action="store_true",
                     help="Read the tool's own stream cache")
    src.add_argument("--gpx-dir", help="Directory of GPX files")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use at most this many routes")
    ap.add_argument("--res-sweep", action="store_true",
                    help="Also sweep --grade-res-m over the real data")
    ap.add_argument("--json-out", default=None,
                    help="Write the results dict to this path")
    args = ap.parse_args()

    routes = (load_cache(limit=args.limit) if args.cache
              else load_gpx_dir(args.gpx_dir, limit=args.limit))
    if not routes:
        where = str(CACHE_DIR) if args.cache else args.gpx_dir
        print(f"No usable routes found in {where}.")
        print("The stream cache fills up as a side effect of running "
              "find_similar_segments.py normally; run a search first, then "
              "re-run this.")
        return 1

    print(f"Loaded {len(routes)} real routes.")
    res = run_audits(routes)
    if args.res_sweep:
        res_sweep(routes)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
