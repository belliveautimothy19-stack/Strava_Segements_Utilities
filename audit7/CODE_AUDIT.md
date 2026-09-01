# Code audit of the validation framework

The framework, not the matcher, has been the recurring source of wrong
conclusions. This inspects it as software.

## Defects found in audit 7

### 1. Archetype labels were unstable at 20 to 24 percent

`bench/semantic.phase_signature` and the first version of
`audit7.corpus.archetype` both dropped a phase shorter than
`int(MIN_PHASE_FRAC * n_samples)`. The floor is in SAMPLES. When a window
gains or loses one leading sample the floor moves between 2 and 3, and a
two-sample phase is admitted or dropped accordingly.

Measured: 24.3 percent of 1 km window labels changed when one or two
leading samples were dropped, a 30 to 60 m perturbation. Category B is
built entirely from these labels.

Rewriting the classifier with a physical floor, an anchored grid and an
iterative merge reduced churn only to 20.6 percent, and the residual
failures included reversals (flat against up). The correct conclusion is
that a discrete archetype is not a well-defined property of about a fifth
of these windows. Reliability is now measured per window by
`stable_label` and unstable windows are excluded from label-dependent
categories and counted. 66 to 79 percent survive.

### 2. The negative class excluded its hardest members

Category D required "different archetype AND unmatched statistics". Pairs
that were geographically separate but shared an archetype or matched on
statistics went to B or C instead. The negative class was therefore the
easy remainder, and every AUC(A|D) was biased upward.

Measured by re-running against a purely geometric negative:

    window   AUC(A|D) as reported   AUC(A|N) honest
      1000            0.914                0.889
      3000            0.929                0.912
      6000            0.986                0.962

The bias is 0.02 to 0.03 at every length. `categorize_geometric` admits
every geographically separate pair and consults no shape or statistic.

### 3. Interval width alone is not a test of adequacy

With 12 positives cleanly separated from 400 negatives the bootstrap AUC
interval is [0.9925, 1.0]. It is narrow because the statistic is pinned
against its ceiling, not because it is well determined. A width test
alone passes that case, which is precisely the "AUC 0.986 from twelve
pairs" claim the gate exists to refuse. `interval_is_adequate` now also
requires a minimum positive count that does not depend on the data.

### 4. Stale parameter documentation in production

`segmatch/match.py`'s class docstring states `res_m 120`. The actual
default is 70.0. The value was changed in audit 2 and the prose was not.
This is the same class of hazard as a fraction read as a percent: the
authoritative statement of a parameter and its real value disagree.

NOT FIXED during this audit, deliberately. Production is frozen and the
source hash is the evidence of that. Recorded as a recommendation.

## Structural changes made

- **Two independent instruments.** `audit7/` recomputes windows, grades,
  geometry, labels and statistics from the raw streams, importing nothing
  from `segmatch` or `bench` except the matcher entry point under test.
  Where a different algorithm was possible it was used: haversine rather
  than an equirectangular projection, central differences rather than a
  windowed OLS slope, an explicit double loop for AUC rather than a
  broadcast comparison.
- **Cross-checks rather than assumptions.** `cross_check.py` compares the
  two paths. Overlap agrees to 0.0000 on 397 real window pairs. Archetype
  labels disagree on 21.3 percent, which is how defect 1 was found.
- **Assertions at every boundary.** Coordinate frames, array lengths,
  grade units, overlap bounds, byte-identical windows, pair
  disjointness, population leakage, sample adequacy, degenerate
  thresholds.
- **The instrument may refuse to answer.** The dense sub-corpus
  resolution comparison returns TOO WIDE rather than a number, because 13
  positives cannot resolve the effect. Six audits of reporting an
  unqualified point estimate is what made this necessary.

## Remaining known weaknesses

- `bench/semantic._pair_distances` uses each pair's own mean latitude
  while `_bbox` uses a fixed origin. The disagreement is bounded at 3.1 m
  and covered by a 50 m margin, so it cannot change a classification, but
  the two frames are still not the same frame. Audit 7 does not use that
  path.
- The category C population is small (29 pairs at 1 km with stable
  labels) and its median should not be read as precise.
- `stable_label` costs roughly 8 archetype evaluations per window. It is
  correct and slow, which is the intended tradeoff for measurement code.
