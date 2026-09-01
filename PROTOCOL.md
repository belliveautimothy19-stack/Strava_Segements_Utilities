# Pre-registered analysis protocol for the GREEN corpus

Written **before the corpus exists**, so that the analysis cannot be
chosen after seeing the data. Seven audits produced nine measurement
defects; several were found only because a number looked wrong, which is
not a method.

Nothing here may be revised once the first new recording is in hand,
except to record a deviation and its reason in a `DEVIATIONS` section.

## 0. Which thresholds are derived and which are conventions

Stated up front because "derived after looking at the data" is the exact
failure this document exists to prevent.

| Threshold | Basis | Kind |
|---|---|---|
| 5 trails | between-trail AUC sd >= 0.038, t interval, half-width <= 0.05 | derived, from a lower-bound variance |
| >= 12 km per trail | measured pair yield: 4 pairs at 8 km, 10 at 12 km | derived, from a measured curve |
| >= 10 A pairs per trail | the point at which a per-trail AUC is estimable at all | convention |
| 2 recordings per trail | the minimum that yields any cross-recording positive | forced, not chosen |
| <= 10 m sampling on one trail | Nyquist: to test 60 m structure needs <= 24 m; 10 m gives margin | derived |
| >= 100 m relief | below this the vertical term's 10 m denominator floor distorts | derived from the code |
| <= 1.0 m elevation step | coarser quantization masks the structure under test | convention |
| >= 3 terrain classes | judgement about the product's range | convention, declared |
| 0.05 CI width | the differences the audit must resolve are 0.02 to 0.03 | derived |
| 30 positives | a saturated AUC has a narrow interval regardless of n | derived from the failure |

Two of these are conventions, not statistics, and are labelled so. The
5-trail figure rests on a variance that is a **lower bound**: only one
distinct trail currently supplies enough pairs, so most of the measured
spread is within-trail. If the new corpus shows a larger between-trail
sd, 5 trails will not be enough and the requirement must be recomputed by
the same formula. That recomputation is pre-authorized here; choosing a
different formula afterwards is not.

## 1. Inclusion and exclusion

A recording is included if and only if:

- it carries distance, elevation and GPS of equal length
- distance is strictly increasing after removing stalled samples
- usable continuous span >= 12,000 m with no gap exceeding 200 m
- median sampling interval <= 35 m
- elevation step <= 1.0 m
- relief >= 100 m

Excluded recordings are listed with the criterion they failed. No
recording may be excluded for its score.

## 2. Geographic independence

Two recordings belong to the same trail if their GPS overlap fraction at
40 m is >= 0.70. Trails are the unit of replication. A trail requires two
recordings on **different days**, so barometric calibration and GPS fix
are independent; same-day repeats count as one recording. Direction is
irrelevant, since both directions are scored and reversal is verified
separately.

Two trails are independent if their mutual overlap is <= 0.10 and their
minimum separation is >= 150 m. Trails failing this are merged and
counted once.

## 3. Window and pair construction

- Window length 6000 m, stride 1500 m.
- A window requires >= 25 samples and >= 0.9 of the nominal length.
- Positives: cross-recording pairs within one trail with overlap >= 0.70.
- Negatives: **every** pair with overlap <= 0.10 and separation >= 150 m.
  Nothing is removed for being hard. This is the defect that inflated
  audits 4 to 6.
- Same-recording pairs are admitted only if their starts differ by at
  least the window length.
- Byte-identical windows are rejected.

## 4. Metrics

**Primary.** AUC(A|N) at 6000 m, res_m 70, band 0.03, production defaults
otherwise.

**Unit of replication: the trail.** The interval is a **cluster
bootstrap over trails**, not over pairs. Pairs within a trail are nested
observations. A pair-level bootstrap is reported alongside only to show
how much narrower it wrongly is.

**Secondary.** Per-trail AUC and the spread across trails; p90(A) and
p10(N); the nearest negative; median A as the realistic same-terrain
baseline; forward and reverse self-match; self-match under 1 m
quantization and under 3x decimation.

**Resolution.** Paired: the same pairs scored at 50, 70, 90, 120, 150 m,
reporting the mean paired difference in retrieval percentile with a
cluster bootstrap over trails, and the effect size before any
significance statement.

## 5. Adequacy gates

The analysis reports NO PRIMARY CONCLUSION unless all hold:

- >= 5 independent trails contribute positives
- >= 30 positives in total and >= 10 per trail
- cluster-bootstrap CI width <= 0.05
- no category contains a byte-identical or route-overlapping pair
- the negative class was constructed geometrically, unfiltered

If a gate fails, the output is the failed gate, not a number.

## 6. Failure criteria

The product objective is **not met** if, with the gates satisfied:

- AUC(A|N) at 6 km < 0.85 across trails, or
- the lower cluster-bootstrap bound < 0.80, or
- p90(A) > p10(N) at 6 km, or
- any single trail scores below 0.75 while others exceed 0.90, which
  would indicate the result is a property of terrain rather than of the
  matcher.

## 7. GREEN

GREEN requires the gates satisfied, no failure criterion triggered, and
no production change made on the basis of this corpus.

## 8. If a parameter is changed

Changing `res_m`, the band, or any weight on the strength of this corpus
converts it from validation into tuning, and it can no longer validate
anything.

Required if that happens:

1. Split by **trail**, never by pair: 3 trails to tune, 2 held out,
   assigned by a seeded shuffle recorded before any scoring.
2. Tune only on the tuning trails.
3. Report the primary metric on the holdout trails, once. Not iterated.
4. If the holdout result is used to reject the change, the holdout is
   spent and GREEN requires a further independent trail.

The default expectation is that no parameter changes and the corpus is
spent entirely on validation.
