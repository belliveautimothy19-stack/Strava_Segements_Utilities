# Parameter selection

Every tolerance, weight and resolution in `segmatch` was chosen by
sweeping it against a labelled dataset, not by intuition. This file
records the measurements. Reproduce any of them with `bench/optimize.py`.

## Honest limits of this evidence

There are no Strava credentials in the environment these sweeps were run
in, so the evaluation set is **synthetic**. Elevation is generated from a
power-law spectrum (amplitude falling as `k^-1.7`), which is the standard
first-order model of natural topography, with barometric noise, GPS
resampling and uneven sampling applied as separate transformations.

That is good enough to rank parameter settings against each other, and
good enough to expose the failure modes the old implementation had. It is
**not** a substitute for a validation run against real Strava streams.
Treat the selected values as well-founded defaults rather than as
universal optima, and re-run the sweep if you gather a real labelled set.

## Method

`bench/evaluate.py` builds routes that either contain a known target or
do not:

- **positive** the route contains the target, possibly re-recorded at a
  different GPS rate, unevenly sampled, noisy, reversed, truncated or
  extended. All of these are the same terrain and must be found.
- **negative** the route does not contain it. Includes the decisive
  control: a **block-shuffled** copy of the target, which has almost
  exactly the target's grade histogram and a completely different ordered
  shape. Any matcher leaning on aggregate composition passes every other
  case and fails this one.
- **hard** the target with one section materially replaced. Reported
  separately and excluded from precision and recall, because whether it
  "should" match is a judgement call rather than ground truth.

Discrimination is reported as mean per-target AUC (threshold-free, so it
is comparable across settings whose score scales differ), plus precision,
recall, FPR and FNR at the single global threshold maximising F1.
Localization is the fraction of positives whose reported window overlaps
the true interval with IoU >= 0.5.

Selection rule, applied in order:

1. Discard settings whose AUC falls below the best by more than
   evaluation noise.
2. Among survivors prefer the **coarsest** resolution, since coarser is
   cheaper, more robust to noise, and less prone to overfitting the
   evaluation set.
3. Break remaining ties on runtime.

## Shape resolution: `res_m = 120`

The old code used 0.25 mi (402 m) grade bins. That value was treated as a
baseline, not an assumption. Swept from 25 m to 600 m; 4 targets, 3
seeds, 378 cases per setting:

| res_m | AUC | F1 | precision | recall | FPR | FNR | ms |
|-------|-----|----|-----------|--------|-----|-----|-----|
| 90    | 0.9855 | 0.955 | 0.923 | 0.991 | 0.125 | 0.009 | 321 |
| **120** | **0.9867** | **0.955** | **0.927** | **0.986** | **0.118** | **0.014** | **234** |
| 160   | 0.9838 | 0.944 | 0.910 | 0.981 | 0.146 | 0.019 | 164 |
| 200   | 0.9884 | 0.947 | 0.908 | 0.991 | 0.153 | 0.009 | 131 |
| 300   | 0.9896 | 0.923 | 0.864 | 0.991 | 0.236 | 0.009 | 91 |
| 402 (old) | 0.9803 | 0.903 | 0.862 | 0.949 | 0.229 | 0.051 | 79 |

AUC is flat within one standard deviation (about 0.005) across 90 to 300,
so it does not discriminate here. F1 and FPR do: finer resolution mainly
buys **precision**, because a finer shape representation separates
near-misses. F1 plateaus at 90 to 120 (0.955 both) against 0.903 at the
old value, with false positives roughly halved.

**120 m is the coarsest value on that plateau**, and 27 percent cheaper
than 90 m. It is 3.4x finer than the old 0.25 mi. Below 60 m the metrics
degrade again as elevation noise starts to dominate the derivative.

## Grade-distribution resolution: `dist_bin_w = 0` (no binning)

Swept jointly with `res_m`, since individually optimal values need not be
jointly optimal. Histogram bin widths of 0.25, 0.5, 1.0, 2.0 and 4.0
percent were compared against the exact binless Wasserstein-1:

| res_m | binless | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|-------|---------|------|-----|-----|-----|-----|
| 90  | 0.959 | 0.959 | 0.959 | 0.959 | 0.959 | 0.949 |
| 120 | 0.949 | 0.949 | 0.949 | 0.949 | 0.959 | 0.959 |
| 200 | 0.941 | 0.949 | 0.949 | 0.949 | 0.939 | 0.941 |
| 402 | 0.920 | 0.929 | 0.929 | 0.929 | 0.929 | 0.929 |

(F1; AUC was identical to four decimal places within each row.)

Bin width makes no difference at any resolution, and the `res_m` ranking
is unchanged by it, so the two axes do not interact. The composition term
is a genuine but minor contributor to discrimination.

**The empirically optimal binning is therefore no binning.** The exact
distance is not slower, and removing the parameter removes a constant
that would otherwise have to be justified. `hist_distance` is retained in
`segmatch/distance.py` for cheap screening, but nothing uses it by
default.

## Alignment tolerance: `max_shift_frac = 0.03`

How far a feature may sit from where the target has it and still match,
as a fraction of target length.

| max_shift_frac | AUC | F1 | precision | recall | FPR | FNR | ms |
|----------------|-----|----|-----------|--------|-----|-----|-----|
| 0.02 | 0.9708 | 0.931 | 0.964 | 0.900 | 0.062 | 0.100 | 54 |
| **0.03** | **0.9729** | **0.931** | **0.964** | **0.900** | **0.062** | **0.100** | **66** |
| 0.05 | 0.9729 | 0.934 | 0.919 | 0.950 | 0.156 | 0.050 | 85 |
| 0.08 | 0.9729 | 0.934 | 0.919 | 0.950 | 0.156 | 0.050 | 107 |
| 0.12 | 0.9750 | 0.926 | 0.918 | 0.933 | 0.156 | 0.067 | 137 |

This sweep was re-run after discovering that the first version of the
evaluation set had no genuine length-varied positives: `scale_length`
compresses distance while keeping elevation, which makes a climb
*steeper* rather than shorter, so it exercised the shape term and not the
length handling. Truncated and extended copies were added and the band
re-swept.

F1 is flat. The real choice is a **precision/recall trade**: 0.03 gives
FPR 0.062 and FNR 0.100; 0.05 gives FPR 0.156 and FNR 0.050. 0.03 was
chosen because a false positive here sends you to run a hill that does
not match, while a false negative loses one candidate out of many, and
because it is 30 percent faster. Raise it with `--max-shift-frac` if you
would rather miss less.

Note that a wide band is not merely slower: it is what lets DTW ignore
how physically long each section is. Unbounded, a target of 1.25 mi at 8
percent then 2.5 mi at 2 percent scores a perfect 0.000 against a window
holding 0.25 mi at 8 percent, and also against one holding 3.0 mi.

## Scoring weights: 1.0 / 0.6 / 4.0 / 2.0

`w_shape` is fixed at 1.0 as the scale anchor. Swept `w_gain` over
{2, 4, 6} and `w_dist` over {0.3, 0.6, 1.0}: AUC 0.987 to 0.992 and F1
0.947 to 0.969 across all nine combinations, which is inside evaluation
noise.

**The existing weights were kept.** The sweep found no evidence to move
them, and changing parameters that the evidence does not move is how
tuning turns into overfitting. `w_dist = 0.6` also preserves the
deliberate choice to weight composition below shape.

`w_len = 2.0` is new. Swept over {0, 1, 2, 4}, all within noise on F1.
The deciding evidence is not F1 but the reported window: with `w_len = 0`
nothing charges for length, and a perfect match is reported at a clipped
`length_ratio` of 0.95 rather than 1.00. 2.0 is the smallest weight
tested that makes the reported extent land on the true one.

## Vertical sampling interval: `vert_resample_m = 25`

Not swept against match quality, because it does not affect ranking much.
It is set by a different criterion: the interval past which measured gain
stops depending on sampling density. Same synthetic hill, true gain
300 m:

| sample spacing | measured gain |
|----------------|---------------|
| 1 m   | 2172 m |
| 4 m   | 668 m  |
| 10 m  | 409 m  |
| 25 m  | 326 m  |
| 50 m  | 311 m  |
| 100 m | 301 m  |

Gain is not a property of terrain until the interval is fixed. 25 m is
where the curve has substantially flattened while still resolving real
short pitches. What matters far more than the exact value is that the
target and every candidate window use the **same** one: the old code
measured the target on raw GPX spacing and windows on a decimated grid,
and reported a 50 percent gain deviation for identical terrain.

## Search grid: `length_steps = 7`, `stride_frac = 0.02`

Left coarse on purpose. The grid only has to find the right basin,
because the winning window is then refined off-grid by local coordinate
descent (`_refine` in `segmatch/match.py`).

This matters more than it looks. Because DTW warps by whole samples it
cannot represent a fractional stretch, so a window even 0.5 percent off
the target length carries a shape cost of about 0.45 that no band width
removes:

| length stretch | band=1 | band=3 | band=8 | band=12 |
|----------------|--------|--------|--------|---------|
| 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.995 | 0.465 | 0.465 | 0.465 | 0.465 |
| 0.980 | 0.729 | 0.531 | 0.531 | 0.531 |
| 0.950 | 2.275 | 0.961 | 0.514 | 0.514 |

With seven trial lengths spanning 0.75 to 1.15, neighbouring lengths
differ by 5.7 percent, which would leave real score on the table for a
true match. Refining the winner costs about 24 extra evaluations against
thousands screened, and is far cheaper than a denser grid everywhere.

## Result

Same dataset, 6 targets, 3 seeds:

| implementation | AUC | F1 | precision | recall | FPR | FNR | localization | ms |
|----------------|-----|----|-----------|--------|-----|-----|--------------|-----|
| original | 0.9704 | 0.906 | 0.833 | 0.993 | 0.375 | 0.007 | 1.000 | 116.6 |
| original plus correctness fixes | 0.9801 | 0.935 | 0.916 | 0.956 | 0.167 | 0.044 | 1.000 | 87.7 |
| rewritten core | 0.9824 | 0.950 | 0.943 | 0.959 | 0.111 | 0.041 | 1.000 | 62.2 |

False positives fall by a factor of 3.4, and the rewrite is 1.9x faster
despite running at 3.4x finer resolution, because the admissible lower
bound prunes most windows before any expensive comparison.
