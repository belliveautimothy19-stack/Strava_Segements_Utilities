# Parameter selection

Every tolerance, weight and resolution in `segmatch` was chosen by
sweeping it against a labelled dataset. This file records the
measurements. Reproduce any of them with `bench/optimize.py`.

It also records a mistake, because the mistake is the most useful thing
in here: the first version of this benchmark could not have detected the
matcher's worst failure mode, and the resolution it selected was wrong by
a factor of nearly two.

## Honest limits of this evidence

There are no Strava credentials in the environment these sweeps run in,
so the evaluation set is **synthetic**. Nothing here has been validated
against real route data. `bench/real_data.py` exists to do exactly that
and needs no labels; see "Real-world validation" at the end.

Three claims should be kept separate when reading this file:

- **Mathematically verified.** Proved or checked against a reference
  implementation: DTW exactness, lower-bound admissibility, symmetry,
  boundedness, edge estimator correctness. These hold regardless of data.
- **Synthetic benchmark results.** Everything with a precision or recall
  number attached. These are only as good as the generator.
- **Real-world empirical validation.** None yet.

## How the benchmark was wrong, and what it cost

The original generator summed a fixed 24 harmonics over the route length,
so the shortest wavelength it could produce was `length / 24`: 250 m on a
6 km route. It therefore contained **no structure that a 120 m
representation could under-resolve**, and a resolution sweep run against
it could not have detected under-resolution however badly the matcher
suffered from it. The sweep duly reported a broad plateau at 90 to 120 m
and the coarsest value on it was selected.

The failure that hid behind this is easy to state. Build a staircase of
60 m pitches at 12 percent with flat recoveries, and a uniform 6 percent
climb. Over every 120 m both rise by exactly 7.2 m, so the two routes
have identical length, identical gain, identical loss and an identical
grade histogram. Only their ordered shape differs, and only below the
cycle length. At `res_m = 120` the matcher scored them at **0.188**,
which is a near-perfect match between 33 twelve-percent pitches and a
steady grind: two completely different training sessions.

The generator now builds terrain by inverse FFT with a power-law
spectrum down to a specified minimum wavelength, and is normalized on the
standard deviation of its **grade** rather than its elevation, so the
spectral slope `beta` varies how energy is spread across scales without
also changing how steep the route is. Gain-matched staircase probes were
added as negatives. `tests/test_regression.py` locks both the matcher
behaviour and the generator's short-scale content, so this cannot quietly
regress.

## Method

`bench/evaluate.py` builds routes that either contain a known target or
do not:

- **positive** the route contains the target, possibly re-recorded at a
  different GPS rate, unevenly sampled, noisy, DEM-quantized,
  provider-smoothed, reversed, truncated or extended.
- **negative** the route does not contain it. Two kinds matter most: a
  **block-shuffled** copy, which has almost exactly the target's grade
  histogram and a different ordered shape, and a **gain-matched
  staircase**, which matches on every scalar and differs only below a
  known physical scale.
- **hard** the target with one section materially replaced. Reported
  separately and excluded from precision and recall.

Discrimination is reported as mean per-target AUC (threshold-free, so
comparable across settings whose score scales differ), plus precision,
recall, FPR and FNR at the single global threshold maximising F1.

Selection rule: discard settings below the best AUC by more than
evaluation noise, then prefer the **coarsest** resolution among
survivors, then break ties on runtime.

## Shape resolution: `res_m = 70`, `oversample = 8`

Swept 40 to 600 m, across three roughness regimes, three seeds, with
results split into ordinary cases and the under-resolution probes:

| res_m | os | ALL AUC | F1 | FPR | MAIN AUC | PROBE AUC | ms |
|-------|----|---------|----|-----|----------|-----------|-----|
| 120 | 4 | 0.9152 | 0.912 | 0.254 | 0.9639 | 0.7917 | 71 |
| 120 | 8 | 0.9540 | 0.909 | 0.286 | 0.9620 | 0.9167 | 73 |
| 90  | 8 | 0.9592 | 0.921 | 0.190 | 0.9600 | 0.9583 | 111 |
| **70** | **8** | **0.9735** | **0.929** | **0.167** | **0.9631** | **1.0000** | **157** |
| 60  | 8 | 0.9710 | 0.931 | 0.222 | 0.9563 | 1.0000 | 187 |
| 50  | 8 | 0.9717 | 0.926 | 0.254 | 0.9559 | 1.0000 | 240 |

The split is the whole story. On ordinary terrain (MAIN) resolution
barely matters anywhere between 50 and 120: AUC 0.956 to 0.964, flat. The
entire case for finer resolution is the probes, where 120 m scores 0.79
and 70 m scores 1.000.

**70 m is the coarsest value that fully separates the probes.** The
choice is physical rather than fitted: separation follows Nyquist, so
distinguishing a pitch of length `p` needs `res_m` at or below `p`:

| res_m | 100 m pitch | 60 m pitch | 40 m pitch |
|-------|-------------|------------|------------|
| 200 | 0.368 | 0.421 | 0.398 |
| 120 | 4.106 | 0.561 | 0.417 |
| 90  | 5.596 | 2.755 | 0.342 |
| 70  | 6.481 | 4.334 | 1.775 |
| 50  | 7.362 | 5.824 | 3.910 |

(scores against a gain-matched uniform climb; low means fooled)

### Why not go finer

Two counterweights, both measured.

**Elevation noise.** Grade is a derivative, so jitter is amplified by
`1/res_m`. From 0.6 m of barometric noise at `oversample = 8`: 0.42
percent grade noise at 120 m, 0.74 at 70 m, 1.05 at 50 m, 1.98 at 30 m.
Real grade differences of interest are 2 to 5 percent, so below about
50 m the noise approaches the signal.

**Quantization.** Elevation rounding manufactures a staircase near the
grade scale, and finer resolution resolves more of that artifact as if it
were terrain. Same route, quantized, scored against its unquantized self:

| res_m | q=1 m | q=2 m | q=3 m | q=5 m | unrelated terrain |
|-------|-------|-------|-------|-------|-------------------|
| 120 | 0.268 | 0.606 | 0.842 | 1.484 | 3.198 |
| 70  | 0.388 | 1.122 | 1.688 | 2.518 | 3.106 |
| 50  | 0.620 | 1.587 | 2.442 | **3.821** | 3.030 |

At 50 m, 5 m quantization scores **worse than unrelated terrain**: a
total false negative. 70 m tolerates realistic rounding (1 to 2 m)
comfortably. `find_similar_segments.py` now detects quantization in the
target and warns with a suggested coarser resolution.

`oversample = 8` costs about 5 percent runtime and improves the probes
materially at coarse resolutions (probe AUC 0.79 to 0.92 at 120 m), by
reducing interpolation error in the comparison sequence.

## Grade-distribution resolution: `dist_bin_w = 0` (no binning)

Re-verified at `res_m = 70` on the harder benchmark:

| dist_bin_w | AUC | F1 | precision | recall | FPR | ms |
|------------|-----|----|-----------|--------|-----|-----|
| **0 (binless)** | **0.9815** | **0.936** | 0.920 | 0.956 | 0.159 | **153** |
| 0.25 | 0.9818 | 0.936 | 0.920 | 0.956 | 0.159 | 160 |
| 0.5 | 0.9815 | 0.934 | 0.943 | 0.930 | 0.111 | 156 |
| 1.0 | 0.9808 | 0.934 | 0.943 | 0.930 | 0.111 | 163 |
| 2.0 | 0.9779 | 0.934 | 0.916 | 0.956 | 0.167 | 161 |
| 4.0 | 0.9797 | 0.932 | 0.910 | 0.961 | 0.183 | 158 |

Unchanged conclusion: bin width makes no difference, and the exact
binless distance is tied for best while being fastest. **The optimal
binning is no binning**, so the parameter stays at 0 and no constant
needs justifying.

## Alignment tolerance: `max_shift_frac = 0.03`

F1 is flat from 0.02 to 0.12. The real choice is a precision/recall
trade: 0.03 gives FPR 0.062 and FNR 0.100; 0.05 gives FPR 0.156 and FNR
0.050. 0.03 was chosen because a false positive sends you to run a hill
that does not match, while a false negative loses one candidate among
many, and because it is 30 percent faster.

## Weights: `1.0 / 0.6 / 2.0 / 2.0`

`w_gain` was **2.0**, down from 4.0, because the vertical term changed
scale. It previously divided by the target's vertical alone, which made
it unbounded: a 20 m-gain target against a 200 m candidate produced 26,
against a shape distance whose whole range is about 0 to 10. It is now
normalized by the sum of both profiles' vertical, which is symmetric and
lies in [0, 1], so the weight had to be recalibrated to match.

Beyond that, **the benchmark cannot adjudicate `w_gain` or `w_len`**, and
this should be said plainly rather than dressed up as a result:

| configuration | AUC | F1 | precision | recall |
|---------------|-----|----|-----------|--------|
| w_gain 0 (term removed) | 0.9812 | 0.937 | 0.933 | 0.944 |
| w_gain 2 | 0.9785 | 0.934 | 0.904 | 0.970 |
| w_gain 2, w_len 0 | 0.9804 | 0.934 | 0.944 | 0.928 |
| w_gain 0, w_len 0 | 0.9812 | 0.936 | 0.893 | 0.984 |

All four are statistically identical. The reason is structural: identical
grade sequences over an identical length force identical vertical, so the
vertical term can only differ when grades or length already differ, and
both of those are charged elsewhere. Its independent contribution is
confined to sub-`res_m` structure that the grade representation smooths
away.

The benchmark is also **biased against `w_len`**, because it labels
truncated copies as positives, so any term charging for length mismatch
reduces recall on them.

Both weights are therefore **semantic choices** that the benchmark
confirms cost nothing, not values the benchmark selected. A complete
vertical mismatch costs 2.0, comparable to a moderate shape difference; a
25 percent length mismatch costs 0.5. `w_dist = 0.6` keeps composition
weighted below shape.

## Search grid: `length_steps = 5`, `stride_frac = 0.06`

Deliberately coarse. The grid only has to find the right basin, because
the winner is refined off-grid afterwards.

| stride | lengths | AUC | F1 | localization | ms |
|--------|---------|-----|----|--------------|-----|
| 0.02 | 7 | 0.9804 | 0.936 | 1.000 | 279 |
| 0.04 | 7 | 0.9786 | 0.936 | 1.000 | 202 |
| **0.06** | **5** | **0.9815** | **0.936** | **1.000** | **158** |
| 0.08 | 5 | 0.9782 | 0.936 | 1.000 | 145 |

Quality is flat and localization is perfect at every setting while
runtime nearly halves. Paying for a fine grid buys nothing that
refinement does not already supply.

Refinement matters because the score is genuinely sensitive to length
quantization: DTW warps by whole samples, so it cannot represent a
fractional stretch, and a window even 0.5 percent off the target length
carries a shape cost of about 0.465 that no band width removes.

## Vertical sampling interval: `vert_resample_m = 25`

Set by where measured gain stops depending on sampling density, not by
match quality. Same synthetic hill, true gain 300 m: 2172 m measured at
1 m spacing, 668 at 4 m, 409 at 10 m, 326 at 25 m, 301 at 100 m. What
matters more than the value is that target and window use the same one.

## Result

Same harder benchmark, 4 targets, 3 seeds:

| implementation | AUC | F1 | precision | recall | FPR | FNR | loc | ms |
|----------------|-----|----|-----------|--------|-----|-----|-----|-----|
| original | 0.9052 | 0.884 | 0.792 | 1.000 | 0.476 | 0.000 | 1.000 | 123 |
| original plus correctness fixes | 0.8279 | 0.893 | 0.844 | 0.947 | 0.317 | 0.053 | 0.996 | 80 |
| rewrite, previous parameters | 0.9406 | 0.920 | 0.888 | 0.956 | 0.222 | 0.044 | 1.000 | 78 |
| **rewrite, current parameters** | **0.9815** | **0.936** | **0.920** | **0.956** | **0.159** | **0.044** | **1.000** | **151** |

Most of the residual false-positive rate is one deliberately strict
label. Excluding the block-shuffled negatives, which have the same
pitches in a different order and which a runner might reasonably call
similar terrain, the same configuration gives F1 0.974, precision 0.954,
FPR 0.108 and FNR 0.004.

Runtime roughly doubled against the previous parameters, which is the
price of resolving terrain the old settings could not see. Pass
`--grade-res-m 120` to trade that back, understanding what it costs.

## Real-world validation

**Status: performed.** Seven real routes from the athlete's own Strava
account (Boulder and Roosevelt/White River National Forest, Colorado)
were audited via `bench/real_data.py`. No route data is stored in this
repository; it lives in the local cache outside the tree.

### What real Strava streams actually look like

- **Elevation is quantized**, median step **1.00 m**. Precision varies
  between activities in the same account: most carry whole metres, one
  device generation carries 0.1 m. Any matcher must tolerate both.
- Native sampling runs about 3.6 to 4 m on newer activities.
- Distance streams contain long stalls, the same value repeated for
  hundreds of samples where the athlete stopped.
- Median grade standard deviation about 6 percent, though on the one
  route available at 0.1 m precision the true figure is 3.0 percent and
  the rest is rounding, see below.

### The measurement that decided the question

Real grade energy below 100 m wavelength **cannot be read off served
streams directly**, because two artifacts inflate it and one suppresses
it:

Decimation suppresses it. The same real route measured at its native
6 m sampling and then decimated by hand:

| sampled every | energy below 100 m |
|---------------|--------------------|
| 6 m | 37% |
| 10 m | 22% |
| 15 m | 12% |
| 25 m | 3% |
| 40 m | 0% |

Rounding inflates it. The same real 2 km slice, recorded at 0.1 m
precision, then quantized by hand:

| elevation step | grade sd | energy below 100 m |
|----------------|----------|--------------------|
| native 0.1 m | 3.03% | 37% |
| 0.5 m | 3.58% | 55% |
| **1.0 m (what Strava serves)** | **5.43%** | **80%** |
| 2.0 m | 10.32% | 94% |

So most of the short-scale content in a served stream is the provider's
rounding, not the hill. Genuine sub-100 m terrain content on this
athlete's routes is about **37 percent**, which is real and substantial,
but less than half what the raw served data suggests.

### Resolution noise budget on real data

The quantity the choice actually turns on is how far the representation
moves when elevation is wrong by about one rounding step, relative to
real grade variation at that scale:

| res_m | 150 | 120 | 90 | **70** | 50 | 35 | 25 |
|-------|-----|-----|----|--------|----|----|----|
| noise / signal | 0.03 | 0.05 | 0.07 | **0.09** | 0.13 | 0.17 | 0.22 |

The finest resolution staying under a 0.20 ratio is 35 m. **70 m sits at
0.09, comfortably inside the budget.**

### Discrimination on real data

Eleven cross-route window pairs were selected from real terrain to match
within 6 percent on vertical and 0.9 percent-grade on composition: same
length, same gain, same loss, same grade mix, different hills. Median
scores:

| res_m | self-match | matched-statistics pair | separation |
|-------|-----------|-------------------------|------------|
| 150 | 0.001 | 2.753 | 2.752 |
| 120 | 0.000 | 3.041 | 3.041 |
| 90 | 0.000 | 3.161 | 3.160 |
| **70** | 0.000 | 2.829 | 2.829 |
| 50 | 0.000 | 3.121 | 3.121 |

Aggregate statistics do not establish a match on real data either. But
note what this table does **not** show: separation is flat from 50 to
150 m. **The real data does not prefer 70 m over 120 m.** The synthetic
staircase probe that originally drove the change from 120 to 70 has no
counterpart in these seven routes.

### Self-consistency on real data

| res_m | self-match | 1 m rounded | decimated 25 m | sub-window IoU | separation |
|-------|-----------|-------------|----------------|----------------|------------|
| 120 | 0.001 | 0.004 | 0.185 | 0.955 | 88.6x |
| 90 | 0.001 | 0.002 | 0.213 | 0.932 | 92.8x |
| **70** | 0.000 | 0.001 | 0.294 | 0.948 | 100.4x |
| 50 | 0.000 | 0.000 | 0.436 | 0.980 | 106.5x |

Reversed routes were identified as reversed in 7 of 7 cases. Sub-window
recovery, which is real ground truth obtained by cutting a known interval
out of a real route and requiring the matcher to find it back, localizes
at IoU 0.93 to 0.98 at every resolution. Finer resolution is marginally
better on separation and slightly worse on decimation robustness.

### Verdict on the resolution

**70 m is confirmed as safe on real data but is not shown to be
necessary.** It sits well inside the measured noise budget, separates
real matched-statistics negatives cleanly, and recovers real sub-windows.
It is retained because the staircase failure mode it protects against is
physically real even though it does not occur in these seven routes, and
because the cost of retaining it is small. A reader who prefers the
faster, quieter option can set `--grade-res-m 120` and lose nothing
measurable on this athlete's terrain.

What would change the answer: elevation rounded to 2 m or coarser, which
pushes the noise ratio past the budget and argues for 120 m; or terrain
with genuine short pitches, which argues for 50 m.

### Two defects found in the audit tool itself

Both would have driven the resolution choice the wrong way, and both now
have regression tests.

- `grade_energy_by_scale` interpolated every stream onto a fixed 5 m grid
  regardless of native sampling, so it reported spectral energy at
  wavelengths the data could not resolve, all of it interpolation
  artifact. It now refuses bands below 2.5 sample intervals.
- The resolution recommender probed quantization sensitivity by
  re-rounding the elevation. Most served streams are already rounded, so
  this was a no-op that reported a noise ratio of 0.00 at every
  resolution, which would have licensed an arbitrarily fine resolution on
  the noisiest possible data. It now perturbs at the rounding scale
  instead.

### What is still not established

Whether two *different* real segments that a runner would call the same
kind of hill score close together. That needs human labels. Everything
above measures self-consistency, localization and separation, which is
necessary but not sufficient.


## Semantic similarity: does the matcher find a different hill of the same kind?

Every earlier experiment measured self-consistency (the same terrain
re-measured) or separation (unrelated terrain rejected). Neither answers
the question the tool is actually for. `bench/semantic.py` addresses it
using four categories whose boundaries come from GPS geometry and from an
ordered-shape classifier, both independent of the matcher and of each
other. Aggregate statistics never define a positive; they define only
category C, which is a negative.

  A  same physical terrain, second pass (out-and-back retrace)
  B  same archetype, geographically distinct hill
  C  matched statistics, different archetype
  D  clearly different

87 windows of 1 km from 4 GPS-bearing routes; 3244 pairs after excluding
trivial shifted copies; 660 scored at each of five resolutions.

### The realistic baseline is not zero

Category A, genuine second passes over the same ground, scores a median
of **2.47**, not 0. Earlier audits quoted a self-match of 0.000, which
compares identical bytes rather than the same hill measured twice. Every
separation claim in this file should be read against 2.47, and the usable
dynamic range is 2.47 to about 6.6, not 0 to 6.6.

### Result, by archetype

Position 0 is the same-hill baseline, 1 is arbitrary negatives:

| archetype | n | 50 m | 70 m | 90 m | 120 m | 150 m |
|-----------|---|------|------|------|-------|-------|
| sustained_climb | 121 | 0.30 | 0.29 | 0.32 | 0.34 | 0.34 |
| sustained_descent | 48 | 0.39 | 0.34 | 0.36 | 0.37 | 0.36 |
| descent_then_climb | 16 | 0.54 | 0.46 | 0.40 | 0.38 | 0.32 |
| rolling | 24 | 0.81 | 0.84 | 0.84 | 0.75 | 0.70 |
| climb_then_descent | 11 | 0.99 | 0.93 | 0.76 | 0.76 | 0.66 |

The split is sharp and it has a mechanism. A sustained climb is a
near-constant grade sequence, so any two instances of similar steepness
look alike whatever the alignment. Rolling terrain differs between hills
in the PHASE of its oscillation, and a bounded warping band cannot bring
two different phases into correspondence.

Confirmed by widening the band at fixed resolution: rolling moves from
0.67 to 0.42 as the band goes 0.03 to 0.30, while sustained barely moves.

### Retrieval tells a more favourable story than pairwise medians

Ranking every other window against a target, which is how the tool is
used:

| archetype | best same-archetype partner | same-archetype in top 10 |
|-----------|---------------------------|--------------------------|
| sustained_climb | top 0% | 4.9 |
| descent_then_climb | top 0% | 4.6 |
| rolling | top 0% | 3.2 |
| sustained_descent | top 4% | 2.2 |
| climb_then_descent | top 1% | 1.2 |

For every archetype including rolling, the best same-archetype hill
surfaces in the top few percent. A mediocre median does not matter if the
top of the list is right, and the top of the list is what a user sees.

### Length is the sharper limitation

| window | 600 m | 1000 m | 1500 m | 2200 m |
|--------|-------|--------|--------|--------|
| B position | 0.06 | 0.22 | 0.36 | 0.73 |

Archetype similarity is strong for short targets and nearly gone by
2.2 km. Longer windows carry more idiosyncratic detail, so two different
hills of the same kind diverge. Targets of several km, which is the
documented use case, sit at the weak end of this trend, and no route in
this corpus is long enough to measure past 2.2 km.

### A correction to the second audit

Audit 2 tightened `max_shift_frac` to 0.03 and the write-up implied it
protected against the gain-matched staircase. It does not. The staircase
scores 2.490, 4.777 and 6.806 for 40, 60 and 100 m pitches at **every**
band from 0.03 to 0.30. That protection comes from resolution, not from
the band. The band's real effect is the tradeoff above: wider recognises
rolling archetypes better, and moves real matched-statistics negatives
from 0.36 to 0.25 above the same-hill baseline.

The band was left at 0.03. This is a product decision, not a defect, and
it is recorded rather than taken silently.

### Two defects found in the experiment itself

- Category A initially admitted windows cut from overlapping stretches of
  the same route, which share most of their samples. That is a slice of
  itself, not a second pass, and it flattered the positive baseline.
- The archetype classifier labelled gentle wobble around a steady climb
  as "rolling" because it keyed on sign changes alone. 14 of 16
  supposedly different-archetype pairs were that artifact. A phase now
  has to average at least 2 percent grade to count as a phase.

Both have regression tests.
