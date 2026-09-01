"""
Empirical evaluation of the matcher.

Builds a labelled dataset of routes that either do or do not contain a
known target, runs the matcher over it, and reports discrimination and
localization quality. Parameters are then chosen from these numbers rather
than from intuition.

Labelling rules, stated so the metrics mean something:

  POSITIVE  the route contains the target, possibly re-recorded (noise,
            sampling rate, uneven sampling), run in the opposite
            direction, or mildly rescaled in length. All of these are the
            same terrain and must be found.

  NEGATIVE  the route does not contain the target. Includes the decisive
            control: a block-shuffled copy of the target, which has
            almost exactly the target's grade histogram but a different
            ordered shape. A matcher that leans on aggregate composition
            passes everything else and fails this.

  HARD      the route contains the target with one section materially
            replaced. Reported separately and excluded from precision and
            recall, because whether it "should" match is a judgement call
            rather than ground truth.

Discrimination is reported primarily as mean per-target AUC, which is
threshold-free and therefore comparable across parameter settings whose
score scales differ. Precision, recall, FPR and FNR are then reported at
the single global threshold maximizing F1.
"""

import time
import numpy as np

from segmatch.match import MatchConfig, prepare_target, match_segment
from bench import synth

POS, NEG, HARD = "positive", "negative", "hard"


def build_dataset(seed=20240501, n_targets=6, target_lengths=(2400.0,
                  4000.0, 6200.0, 9000.0), beta=1.45,
                  resolution_probes=True):
    """Return a list of (target_d, target_e, cases) tuples.

    beta controls how the synthetic terrain spreads grade energy across
    scales; see synth._broadband. resolution_probes adds the gain-matched
    staircase groups described below.
    """
    rng = np.random.default_rng(seed)
    kinds = list(synth.ARCHETYPES)
    dataset = []
    for t in range(n_targets):
        kind = kinds[t % len(kinds)]
        L = target_lengths[t % len(target_lengths)]
        td, te = synth.terrain(rng, L, kind, spacing=8.0, beta=beta)
        cases = []

        def add(label, d, e, truth=None, tag=""):
            cases.append({"label": label, "d": d, "e": e, "truth": truth,
                          "tag": tag})

        # ---- positives -------------------------------------------------
        d, e, tr = synth.embed(td, te, rng)
        add(POS, d, e, tr, "exact")

        d, e, tr = synth.embed(td, synth.add_baro_noise(te, rng, 0.6), rng)
        add(POS, d, e, tr, "baro_noise")

        d, e, tr = synth.embed(td, synth.add_baro_noise(te, rng, 1.5), rng)
        add(POS, d, e, tr, "heavy_noise")

        d, e, tr = synth.embed(td, te, rng)
        rd, re = synth.resample_at(d, e, 25.0)
        add(POS, rd, re, tr, "coarse_gps")

        d, e, tr = synth.embed(td, te, rng)
        rd, re = synth.resample_at(d, e, 4.0)
        add(POS, rd, re, tr, "fine_gps")

        d, e, tr = synth.embed(td, te, rng)
        rd, re = synth.resample_uneven(d, e, rng)
        add(POS, rd, re, tr, "uneven")

        pd_, pe = synth.perturb_grade(td, te, rng, 1.0)
        d, e, tr = synth.embed(pd_, pe, rng)
        add(POS, d, e, tr, "small_perturb")

        d, e, tr = synth.embed(td, te[::-1].copy(), rng)
        add(POS, d, e, tr, "reversed")

        for f in (0.90, 1.10):
            sd, se = synth.scale_length(td, te, f)
            d, e, tr = synth.embed(sd, se, rng)
            add(POS, d, e, tr, f"scaled_{f:.2f}")

        # Genuinely shorter and longer copies, i.e. real partial matches.
        # scale_length compresses distance while keeping elevation, which
        # makes a climb STEEPER rather than shorter, so it exercises the
        # shape term and not the length handling. Truncation and extension
        # are what actually test length tolerance, and their absence from
        # an earlier version of this set let the alignment band be tuned
        # against near-exact-length positives only.
        for frac in (0.85, 0.95):
            k = max(4, int(td.size * frac))
            d, e, tr = synth.embed(td[:k], te[:k], rng)
            add(POS, d, e, tr, f"truncated_{frac:.2f}")

        ext_d = np.concatenate([td, td[-1] + (td[1:] - td[0]) * 0.12])
        ext_e = np.concatenate([te, te[-1] + (te[1:] - te[0]) * 0.12])
        d, e, tr = synth.embed(ext_d, ext_e, rng)
        add(POS, d, e, tr, "extended_1.12")

        d, e, tr = synth.embed(td, te, rng)
        add(POS, d, synth.quantize_elevation(e, 1.0), tr, "dem_quantized")

        d, e, tr = synth.embed(td, te, rng)
        add(POS, d, synth.smooth_elevation(d, e, 30.0), tr,
            "provider_smoothed")

        d, e, tr = synth.embed(td, te, rng, pre_m=0.0)
        add(POS, d, e, tr, "at_route_start")

        d, e, tr = synth.embed(td, te, rng, post_m=0.0)
        add(POS, d, e, tr, "at_route_end")

        # ---- negatives -------------------------------------------------
        for k in kinds:
            if k == kind:
                continue
            nd, ne = synth.terrain(rng, L * rng.uniform(1.1, 2.2), k,
                                    spacing=10.0, beta=beta)
            add(NEG, nd, ne, None, f"unrelated_{k}")

        sd, se = synth.shuffle_blocks(td, te, rng)
        d, e, _ = synth.embed(sd, se, rng)
        add(NEG, d, e, None, "block_shuffled")

        sd, se = synth.shuffle_blocks(td, te, rng, n_blocks=4)
        d, e, _ = synth.embed(sd, se, rng)
        add(NEG, d, e, None, "block_shuffled_4")

        # ---- hard ------------------------------------------------------
        alt = te.copy()
        q = alt.size // 4
        seg = np.linspace(0, -0.09 * (td[2 * q] - td[q]), q)
        alt[q:2 * q] = alt[q] + seg
        alt[2 * q:] += (alt[2 * q - 1] - te[2 * q - 1])
        d, e, tr = synth.embed(td, alt, rng)
        add(HARD, d, e, tr, "section_replaced")

        dataset.append((td, te, cases))

    if resolution_probes:
        dataset.extend(_resolution_probe_groups(rng))
    return dataset


def _resolution_probe_groups(rng):
    """Groups whose negatives differ from the target ONLY below a known
    physical scale.

    The target is a uniform climb. Each negative is a staircase with
    exactly the same length, gain and loss, so length, vertical and grade
    composition are all matched by construction and only ordered shape at
    scales below the pitch can separate them. A representation coarser
    than the pitch cannot, and will score these as near-perfect matches.

    These exist because the original benchmark could not have detected
    under-resolution: its generator produced no structure below about
    250 m, so every resolution it tried was adequate for the data.
    """
    groups = []
    for length_m, grade in ((4000.0, 0.06), (6000.0, 0.045)):
        gain = grade * length_m
        ud, ue = synth.uniform_climb(length_m, grade, spacing=4.0)
        cases = []

        for tag, mut in (
                ("uniform_exact", lambda d, e: (d, e)),
                ("uniform_noisy",
                 lambda d, e: (d, synth.add_baro_noise(e, rng, 0.6))),
                ("uniform_coarse_gps",
                 lambda d, e: synth.resample_at(d, e, 25.0)),
                ("uniform_quantized",
                 lambda d, e: (d, synth.quantize_elevation(e, 1.0)))):
            d, e = mut(ud, ue)
            d2, e2, tr = synth.embed(d, e, rng, pre_m=900.0, post_m=700.0)
            cases.append({"label": POS, "d": d2, "e": e2, "truth": tr,
                          "tag": tag})

        for pitch in (40.0, 60.0, 100.0):
            sd, se = synth.gain_matched_staircase(length_m, gain, pitch,
                                                   spacing=4.0)
            d2, e2, _ = synth.embed(sd, se, rng, pre_m=900.0, post_m=700.0)
            cases.append({"label": NEG, "d": d2, "e": e2, "truth": None,
                          "tag": f"staircase_{pitch:.0f}m"})
        groups.append((ud, ue, cases))
    return groups


def _iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return (hi - lo) / union if union > 0 else 0.0


def _auc(pos_scores, neg_scores):
    """Rank AUC with lower-is-better scores. 1.0 = perfect separation."""
    if not pos_scores or not neg_scores:
        return float("nan")
    p = np.asarray(pos_scores)[:, None]
    n = np.asarray(neg_scores)[None, :]
    return float(((p < n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def evaluate(cfg, dataset, iou_thresh=0.5, collect=False):
    """Run the matcher over the dataset and return a metrics dict."""
    per_target_auc, rows = [], []
    t0 = time.perf_counter()
    n_calls = 0

    for td, te, cases in dataset:
        target = prepare_target(td, te, cfg)
        if target is None:
            continue
        pos, neg = [], []
        for c in cases:
            ms = match_segment(c["d"], c["e"], target, cfg)
            n_calls += 1
            score = ms[0].score if ms else float("inf")
            ok_loc = None
            if ms and c["truth"] is not None:
                ok_loc = _iou((ms[0].start_m, ms[0].end_m),
                              c["truth"]) >= iou_thresh
            rows.append({"label": c["label"], "tag": c["tag"],
                         "score": score, "localized": ok_loc})
            if c["label"] == POS:
                pos.append(score)
            elif c["label"] == NEG:
                neg.append(score)
        per_target_auc.append(_auc(pos, neg))
    runtime = time.perf_counter() - t0

    pos_s = [r["score"] for r in rows if r["label"] == POS]
    neg_s = [r["score"] for r in rows if r["label"] == NEG]
    finite = [s for s in pos_s + neg_s if np.isfinite(s)]
    best = {"f1": -1.0}
    for t in np.unique(np.round(finite, 4)) if finite else []:
        tp = sum(1 for s in pos_s if s <= t)
        fn = len(pos_s) - tp
        fp = sum(1 for s in neg_s if s <= t)
        tn = len(neg_s) - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "threshold": float(t), "precision": prec,
                    "recall": rec,
                    "fpr": fp / (fp + tn) if fp + tn else 0.0,
                    "fnr": fn / (fn + tp) if fn + tp else 0.0}

    loc = [r["localized"] for r in rows
           if r["label"] == POS and r["localized"] is not None]
    out = {
        "auc": float(np.nanmean(per_target_auc)) if per_target_auc else 0.0,
        "localization": float(np.mean(loc)) if loc else 0.0,
        "runtime_s": runtime,
        "ms_per_match": 1000.0 * runtime / max(1, n_calls),
        "n_cases": len(rows),
    }
    out.update({k: v for k, v in best.items() if k != "f1"})
    out["f1"] = best["f1"]
    if collect:
        out["rows"] = rows
    return out
