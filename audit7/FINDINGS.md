# Audit 7: adversarial re-validation

Production was frozen for the whole audit. `segmatch/*.py` hashes to
`0f7dfc3d...` before and after. Nothing in this round changes matching
behavior.

## Every claim from audits 1 to 7

| Claim | Status | Evidence | Vulnerability | Confidence |
|---|---|---|---|---|
| Unbounded DTW scores 0.000 between a 0.25 mi and a 1.25 mi climb | DEMONSTRATED | test_distance.test_unbanded_would_fail | none, it is a unit test on a closed form | high |
| Quantile-grid Wasserstein was biased (4.505 where the answer is 5.0) | DEMONSTRATED | test_distance.test_known_value | none | high |
| The window-length grid never included 1.0x | DEMONSTRATED | test_matching, arithmetic | none | high |
| Vertical term was unbounded and asymmetric (16.0 vs 24.3 on the same pair) | DEMONSTRATED | closed form in vertical_deviation | none | high |
| Gain must be measured at a fixed spatial interval | DEMONSTRATED | audit7 check, fine 137.7 vs coarse 137.7 | none | high |
| res_m 402 m under-resolves; a 90 to 120 m plateau beat it on synthetic F1 | SUPPORTED | bench/optimize sweep | synthetic benchmark only; the generator was later found to lack sub-250 m energy and was rebuilt | medium |
| The alignment band protects against the gain-matched staircase | **RETRACTED** | audit 6 found identical scores at every band; audit 7 reconfirms 4.095 at bands 0.03, 0.06, 0.10, 0.18, 0.30 | the original claim was never measured | n/a |
| Band 0.03 is the best value | **RETRACTED as "best"** | audit 7: real AUC(A|N) rises monotonically 0.890 to 0.913 from band 0.03 to 0.30 | CIs overlap pairwise; the monotone trend across 5 values is the evidence | medium |
| res_m 120 under-resolves a 60 m staircase | DEMONSTRATED | audit 7 verified probe: separation 4.10 at res 70, 1.40 at 120, 0.20 at 150, gain_dev exactly 0 | none; the probe was independently verified before use | high |
| Shape recovery degrades with window length | **RETRACTED** | audit 6, reconfirmed audit 7: AUC(A|N) rises 0.889 (1 km) to 0.962 (6 km) | none | n/a |
| The apparent length degradation was a sampling-rate and route-selection confound | DEMONSTRATED | decimation error tracks native spacing (0.54 at 6 m, 1.26 at 30 m); with the route held fixed, error falls 1.268 at 1 km to 0.799 at 8 km | none | high |
| Archetype similarity collapses with length | **RETRACTED** | audit 6: pooled B tracked its own composition; B_strict flat at 0.75 to 0.84 | n/a | n/a |
| Category A was contaminated by trivial shifted copies | DEMONSTRATED | fixed and locked | none | high |
| The archetype classifier mislabels gentle wobble as rolling | DEMONSTRATED | 14 of 16 pairs were the artifact | none | high |
| Category B labels are trustworthy | **RETRACTED** | audit 7: 24.3 percent of labels flip under a 30 to 60 m window shift; rewriting reduces it only to 20.6 percent | the instability is a property of the terrain, not the code | n/a |
| AUC(A|D) values reported in audits 4 to 6 | **RETRACTED as biased** | audit 7: the D class excluded its hardest members; honest AUC(A|N) is 0.02 to 0.03 lower at every length | none | n/a |
| 6 km physical matching AUC 0.986 | **RETRACTED, corrected to 0.962 [0.934, 0.982]** | audit 7 geometric negative, n_A=81 | evidence is one trail | n/a |
| The matcher is fundamentally a shape matcher | **RETRACTED** | audit 7 rank correlation of score against independent axes: steepness difference 0.551, magnitude-preserving shape 0.469, net vertical 0.415, steepness-blind ordered shape **0.128** | none | n/a |
| The matcher requires magnitude agreement, with ordered shape refining | DEMONSTRATED | five populations: A 2.36 < B_strict 4.16 < C 4.78 < D 6.14 < B_loose 6.41 | labels restricted to perturbation-stable windows | high |
| Optimal resolution tracks the structure scale to be discriminated | DEMONSTRATED | discrimination-to-noise ratio peaks at res 50 for a 60 m pitch, 90 for 90 m, 120 for 120 m, 150 for 180 m | synthetic | high |
| Real data can decide the resolution question | **RETRACTED** | audit 7: all five resolutions give AUC 0.872 to 0.879 with CIs overlapping almost entirely; the dense sub-corpus gives CI width 0.21 on 13 positives | n/a | n/a |
| Shape recovery improves with length | SUPPORTED | AUC(A|N) 0.825, 0.889, 0.897, 0.909, 0.912, 0.923, 0.914, 0.962 | 600 m and 5000 m rows fail the adequacy gate; the 6 km point is one trail | medium-high |
| 6 km operation works | SUPPORTED, not DEMONSTRATED generally | see below | one trail | medium |

## Physical identity, honest instrument

Negatives are every geographically separate pair, with nothing filtered
out for being hard.

| window | n_A (cross/same) | n_N | AUC(A\|N) | 95% CI | adequate | p90(A) | p10(N) | nearest neg |
|---|---|---|---|---|---|---|---|---|
| 600 | 44 (0/44) | 700 | 0.825 | [0.760, 0.884] | no, width 0.125 | 6.301 | 3.215 | 1.600 |
| 1000 | 700 (452/248) | 700 | 0.889 | [0.872, 0.905] | yes | 4.943 | 3.266 | 1.622 |
| 1500 | 467 (307/160) | 700 | 0.897 | [0.879, 0.914] | yes | 5.130 | 3.694 | 2.102 |
| 2200 | 277 (195/82) | 700 | 0.909 | [0.890, 0.927] | yes | 4.808 | 3.805 | 2.389 |
| 3000 | 200 (129/71) | 700 | 0.912 | [0.893, 0.932] | yes | 4.530 | 3.617 | 2.569 |
| 4000 | 135 (95/40) | 700 | 0.923 | [0.903, 0.942] | yes | 4.194 | 3.716 | 2.301 |
| 5000 | 100 (72/28) | 231 | 0.914 | [0.884, 0.941] | no, width 0.057 | 3.851 | 3.383 | 2.530 |
| 6000 | 81 (61/20) | 112 | 0.962 | [0.934, 0.982] | yes | 3.584 | 3.794 | 2.841 |

Cross-recording and same-recording positives agree closely (0.957 vs
0.976 at 6 km), so the result is not carried by same-file retraces.

**6 km is the only length where p90(A) < p10(N).** Below it the
distributions overlap at the operating edges.

## Terrain diversity behind the 6 km claim

Pairwise route overlap gives five independent locations:

    {19476565994, 19670306718}   one trail, 0.987 overlap, two recordings
    {19131631580}                steep Boulder climb, 4.9 km
    {19621145681}                Boulder trail run, 9.9 km
    {19853326285}                Betasso Preserve, 5.5 km
    {19869723537}                Boulder rolling, 5.2 km

Only the first is long enough to yield 6 km windows. **Every 6 km
category A pair comes from that one trail.** The multi-location evidence
exists only at 1000 to 4000 m, where four to six route pairings
contribute.

So the length TREND is supported across five locations; the 6 km POINT
ESTIMATE is one trail.

## Resolution

Three separate questions, with three different answers.

**What synthetic physics demonstrates.** The probe was verified before
use: measured dominant wavelength matches 2x pitch to within 10 percent,
sampling is 2 m so the probe is not itself sampling-limited, the ramp has
exactly 5.000 percent grade with zero variance, and `gain_dev` and
`len_dev` are exactly 0 so the separation is shape alone.

    pitch    res 50  res 70  res 90  res 120  res 150
      60 m     5.22    4.09    2.97     1.40     0.20
      90 m     6.16    5.41    4.63     3.60     2.39
     120 m     6.63    6.06    5.54     4.78     3.88
     180 m     7.10    6.71    6.32     5.81     5.21

Combined with noise sensitivity, the discrimination-to-noise ratio peaks
where the resolution matches the structure scale: res 50 for a 60 m
pitch, 90 for 90 m, 120 for 120 m, 150 for 180 m. There is no single
optimum. Resolution is a choice of the finest scale the product must
resolve.

**What real data demonstrates.** Nothing. All five resolutions give
AUC(A|N) between 0.872 and 0.879 with almost entirely overlapping
intervals. Coarser is nominally better on noise (1.427 to 0.593),
decimation (1.149 to 0.750) and runtime (152.6 ms to 35.8 ms at 6 km).
Elevation quantization at 1 m is negligible at every resolution
(0.003 to 0.012).

**What is unresolvable with the served streams.** Whether fine
resolution helps where fine structure exists. Independently measured
grade variance below 60 m: 0.341 on the 6 m-sampled route, 0.090 on the
13.5 m route, and refused as below Nyquist on the 30 to 33 m routes. The
terrain has the structure; the long streams do not record it. The dense
sub-corpus has 13 category A pairs and returns a CI width of 0.21, so it
cannot answer.

## The 70 m recommendation, restated honestly

70 m is **not** empirically optimal. It is not the best value on any
real-data axis measured, and on noise, decimation and runtime it is
beaten by every coarser value.

The correct statement is the conservative one:

> 70 m is the finest resolution that is defensible under the current
> data and physics constraints. It preserves discrimination of terrain
> structure down to roughly 60 to 90 m, which real terrain in this region
> demonstrably contains (34 percent of grade variance below 60 m on a
> 6 m-sampled route). Moving to 120 m would discard two thirds of the
> discrimination at 60 m for a runtime gain, on the strength of real-data
> evidence that is statistically incapable of detecting the loss. Keeping
> 70 m is therefore the choice that fails safe.

## Alignment band

Independently reconfirmed: the band does not affect staircase
discrimination at all (4.095 at every band from 0.03 to 0.30). Audit 2's
claim stays retracted.

On real data, widening improves separation monotonically:

    band   AUC(A|N)   95% CI            median A   median N   nearest neg
    0.03    0.8896    [0.8723, 0.9063]    2.539      5.838       1.920
    0.06    0.8958    [0.8791, 0.9118]    2.278      5.604       1.699
    0.10    0.9010    [0.8845, 0.9168]    2.134      5.454       1.582
    0.18    0.9081    [0.8925, 0.9231]    2.034      5.229       1.439
    0.30    0.9125    [0.8977, 0.9267]    1.997      4.992       1.438

Positives improve faster than negatives degrade, but negatives do move
closer: the nearest negative falls from 1.920 to 1.438. Adjacent
intervals overlap; the evidence is the monotone trend across five values.

This is a genuine Pareto tradeoff and 0.03 is a defensible point on it,
but 0.03 is **not** the accuracy-optimal value on real data. Production
value unchanged, as instructed. Recorded as a recommendation.

## What the matcher actually matches

Rank correlation of the score against axes defined outside it, on 1500
geographically separate 1 km pairs:

    steepness difference                 0.551
    magnitude-preserving shape           0.469
    net vertical difference              0.415
    steepness-blind ordered shape        0.128

Median score against steepness-blind shape rises from 4.29 to 6.32 and
then TURNS OVER to 5.73 in the top two bins. Against
magnitude-preserving shape it rises monotonically, 3.96 to 7.25.

Five populations, labelled without the matcher, restricted to windows
whose archetype label survives perturbation:

    A         same ground                          2.361
    B_strict  same archetype, similar steepness    4.155
    C         matched statistics, different shape  4.780
    D         different everything                 6.142
    B_loose   same archetype, different steepness  6.411

B_strict beats C, so ordered shape contributes beyond aggregate
statistics. B_loose is worse than D, so ordered shape contributes
essentially nothing once steepness disagrees.

**The matcher is not fundamentally a shape matcher.** Magnitude
agreement is close to a precondition; given it, ordered shape refines the
ranking. Whether that is the right product behavior is a product
question, and the current answer is defensible: most runners would not
call a 3 percent and a 9 percent climb the same hill.
