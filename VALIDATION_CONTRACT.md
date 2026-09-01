# Validation contract

What counts as sufficient validation for this project. It exists so the
audit treadmill does not restart: seven audits found nine defects in the
measuring apparatus, and the way out is a standing definition of done
rather than another round of judgement.

## 1. Production objective

> Given a segment the athlete names, find other segments in a fetched
> candidate set whose **climbing demand** is comparable: similar ordered
> profile AND similar steepness and vertical, at comparable length. Rank
> them, and place each against a null distribution so the athlete can
> distinguish a real twin from the closest thing nearby.

This is magnitude-first. Measured rank correlation of the score against
axes defined outside the matcher, over 1500 real pairs: steepness
difference 0.55, magnitude-preserving shape 0.47, net vertical 0.42,
ordered shape with steepness normalized away **0.13**. Ordered shape
refines the ranking within comparable magnitude; it does not substitute
for it. A 3 percent and a 9 percent climb of identical profile are not
similar under this objective, and that is deliberate.

Anything describing the tool as matching "similar terrain" or ordered
shape in preference to magnitude contradicts this contract.

## 2. Required evidence

| Question | Instrument | Standard |
|---|---|---|
| Physical identity | geometric A/N split, labels from GPS only | AUC(A\|N) with a cluster-aware interval |
| Long-window behavior | the same, swept over window length | no degradation with length |
| Resolution | paired comparison, same pairs at each resolution | effect size, not only significance |
| Reversal | forward and reverse scored for every window | reversal must not change a self-match |
| Quantization robustness | elevation rounded to the device step | self-match must not move materially |
| Statistics-only negatives | matched gain, loss and grade composition, different shape | must score worse than same ground |
| Same-terrain baseline | two independent recordings of one trail | the realistic floor, never 0.0 |

## 3. Required data diversity

Counts are of **trails**, not activities. Two recordings of one trail are
one trail.

| Requirement | Value | Derivation |
|---|---|---|
| Independent trails | >= 5 | Between-trail AUC sd is at least 0.038. With a t interval, 5 trails give a 95 percent half-width of 0.048 on the across-trail mean; 3 give 0.095. |
| Recordings per trail | >= 2 | Two is the minimum that yields any cross-recording positive. A third would let recording-level variance be separated from trail-level, which is not needed for the primary claim. |
| Continuous length per trail | >= 12 km | Measured, not assumed: two recordings of a route of length L yield 4 category A pairs at 8 km, 10 at 12 km, 16 at 15 km, 28 at 18 km, 59 at 24 km. 12 km is where a per-trail estimate becomes possible. |
| Trails sampled at <= 10 m AND >= 12 km | >= 1 | The only way to test whether fine resolution helps where fine structure exists. |
| Elevation step | <= 1.0 m | Coarser quantization masks the structure under test. |
| Vertical relief per trail | >= 100 m | Below this the vertical term's denominator floor dominates. |
| Terrain classes | >= 3 distinct | Sustained climb, rolling, and mixed. Diversity matters more than activity count: five trails of 12 km beat one trail of 60 km. |

Run `python -m audit7.accept <dir>` on candidate GPX or stream files. It
reports per-trail status and which criteria are short.

## 4. Required metrics

- **Primary retrieval metric.** AUC(A|N), positives being the same ground
  seen twice, negatives being every geographically separate pair with
  nothing removed for being hard.
- **Acceptable uncertainty.** 95 percent interval width below 0.05, AND
  at least 30 positives, AND at least 5 contributing trails. Width alone
  is not sufficient: with 12 cleanly separated positives the bootstrap
  interval is [0.9925, 1.0], narrow only because the statistic is pinned
  at its ceiling.
- **False positives.** p10 of the negative distribution and the single
  nearest negative, both reported. The nearest negative is what a user
  actually sees at the top of a ranked list.
- **False negatives.** p90 of the positive distribution. The operating
  requirement is p90(A) < p10(N).
- **Same-terrain baseline.** Median A. It is never 0; at 6 km it is 2.99.
  Any claim quoted against a baseline of 0 is wrong.
- **Resolution comparison.** Paired, same pairs at every resolution.
  Report the effect size with the significance: on 745 real positives,
  res 150 beats res 70 by 0.0057 of retrieval percentile at t = 5.1 —
  detectable and negligible.

## 5. Adequacy rules

The framework must refuse a definitive conclusion when any of these hold.
Refusing is a result; a point estimate in these conditions is not.

1. **n too small.** Fewer than 30 positives, or fewer than 5 trails for a
   generalization claim.
2. **Degenerate positives.** Byte-identical windows, or windows
   overlapping along the route, admitted as an independent positive pair.
3. **Misleading intervals.** A metric pinned near 0 or 1, where interval
   width understates uncertainty. Report the count alongside.
4. **Unstable labels.** Any category built on archetype labels, which are
   20 percent unstable under a 30 to 60 m window shift. Restrict to
   perturbation-stable windows and report how many were excluded.
5. **Contaminated categories.** A pair carrying two labels, or a positive
   also appearing as a negative.
6. **An artificially easy negative class.** A negative defined by
   excluding similar cases. This inflated every AUC in audits 4 to 6 by
   0.02 to 0.03.
7. **Pseudo-replication.** Pairs drawn from one trail are not independent
   observations. A bootstrap over pairs understates uncertainty; cluster
   over trails.
8. **Physical scale unresolvable.** Any spectral or resolution claim
   below 2.5 sample intervals of the stream. There the FFT reports the
   interpolator, not the terrain.

## 6. What GREEN means

> The defined product objective has been validated to a degree
> appropriate to the actual risk, using a validation apparatus that has
> demonstrated resistance to the classes of error that previously
> corrupted the audits.

GREEN is not "all tests pass" and it is not "we have enough data". A
remaining geographic generalization gap requiring more real-world samples
is materially different from an unresolved algorithmic correctness
problem, and this document keeps the two apart.
