# Production freeze, audit 7

Recorded from the running code, not from documentation. Every value below
is what `MatchConfig()` actually returns; the module docstring disagrees
with one of them and that disagreement is itself a finding (see
CODE_AUDIT.md).

## Scoring parameters

    res_m              70.0      grade estimation scale (metres)
    oversample         8         profile grid points per res_m
    vert_resample_m    25.0      fixed interval for gain/loss
    min_ratio          0.75      shortest admissible window / target
    max_ratio          1.15      longest admissible window / target
    length_steps       5         window lengths tried, always incl. 1.0
    stride_frac        0.06      offset grid step, fraction of target
    max_shift_frac     0.03      Sakoe-Chiba band, fraction of n_cmp
    max_overlap        0.5       overlap suppression threshold
    pool_size          16        grid winners kept for refinement
    top_k              1         matches returned per segment
    use_pruning        True      LB_Keogh screening (result-invariant)
    dist_bin_w         0.0       exact binless Wasserstein

## Score

    score = 1.0 * shape + 0.6 * dist + 2.0 * gain_dev + 2.0 * len_dev

`shape` and `dist` are in percent-grade units. `gain_dev` and `len_dev`
are dimensionless fractions in [0, 1], which is why their weights are
larger. Lower is more similar; 0 is identical.

## Distribution representation

Exact binless Wasserstein-1 between the two ordered grade sequences,
computed as the integral of |F_a - F_b|. `dist_bin_w = 0` means no
histogram binning is applied.

## Reversal handling

Both directions are always tried. A reverse window uses the negated,
reversed grade sequence with gain and loss swapped. Direction is reported
on the match, not collapsed away.

## Thresholding

There is no absolute threshold in production. `null_scores` samples
random windows from the same candidate pool to give an empirical
"no relationship" distribution, and a match is placed against it as a
percentile. All fixed thresholds in this audit belong to the audit, not
to production.

## Constants

    MIN_VERT_DENOM_M   10.0
    LB_PROBE_WINDOWS   120
    LB_MIN_HIT_RATE    0.08

## Rule for this audit

Production behavior is frozen. Nothing in `segmatch/` is modified. Any
change judged necessary is recorded as a recommendation only.

## Source hash

    segmatch/*.py sha256 = 0f7dfc3defb0f331821b2fd84058241ce84e3a25775b65ed913aab3b64af80c9

Re-checked at the end of the audit. If it differs, production was
modified and the audit is void.
