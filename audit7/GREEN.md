# What GREEN requires

GREEN means no remaining known uncertainty that is both material to the
defined product behavior and reasonably addressable with available
experiments. It is not "all tests pass". 100 tests pass right now and the
verdict below is not GREEN.

## The product objective, stated so it can be tested

> Given a segment the athlete names, find other segments in a fetched
> candidate set whose **climbing demand** is comparable: similar ordered
> profile AND similar steepness and vertical, at comparable length.
> Rank them, and place each against a null distribution so the athlete
> can tell "this is a real twin" from "this is the closest thing nearby,
> and it is not close".

Two things follow, and both are now measured rather than assumed. It is
a magnitude-first matcher. Ordered shape refines within comparable
magnitude; it does not substitute for it.

## Criteria

Each threshold is tied to a product risk, not chosen to be hard.

| # | Criterion | Threshold | Why this number | Now | Met |
|---|---|---|---|---|---|
| 1 | Independent terrain locations with routes long enough for 6 km windows | >= 4 | One trail cannot show that the result is not a property of that trail. Four gives a visible spread without demanding a survey. | 1 | no |
| 2 | Independent recordings per such location | >= 2 | Separates matcher repeatability from barometer and GPS repeatability. | 2 | yes |
| 3 | Category A pairs at 6 km per location | >= 30 | Below this the AUC interval exceeds the effect being claimed, as the 13-pair dense corpus showed at width 0.21. | 81 total, one location | partly |
| 4 | CI width on AUC(A\|N) at 6 km | < 0.05 | The differences the audit must resolve between configurations are 0.02 to 0.03. | 0.048 | yes |
| 5 | Operating separation at the working length | p90(A) < p10(N) | The product ranks and thresholds; overlap at the edges is what produces a confident wrong answer. | holds at 6 km only | partly |
| 6 | At least one 6 km-capable route sampled at <= 10 m | >= 1 route | Without it the resolution question is permanently unanswerable on real data, and 70 m rests on synthetic physics alone. | 0 | no |
| 7 | Archetype label stability, if semantic claims are made | >= 90% stable | Category B currently rests on labels that are 20% noise. | 66 to 79% | no |
| 8 | Production behavior unchanged during validation | source hash fixed | An audit that edits what it measures proves nothing. | verified | yes |

## Smallest set of evidence that would close it

Not "more data". Specifically:

1. **Three more long routes on distinct trails**, each >= 8 km so it
   yields multiple non-overlapping 6 km windows, each recorded twice.
   That is 6 activities. It closes criteria 1 and 3 and re-tests 5.
2. **One of those recorded at <= 10 m sampling** (a watch set to
   1-second recording, or a GPX export rather than a decimated Strava
   stream). It closes criterion 6 and settles resolution on real data
   for the first time.
3. **Nothing else.** Criteria 2, 4 and 8 are already met. Criterion 7
   matters only if archetype claims are wanted; the physical-identity
   product claim does not depend on it.

Six activities on three trails, one of them densely recorded. That is a
weekend of running, not a research program.

## What would NOT close it

- More recordings of the Roosevelt NF trail. Criterion 1 counts
  locations, not files, and that trail already has two.
- More synthetic experiments. The synthetic side is verified and its
  answer is known.
- Tightening any threshold above. They were set from product risk before
  the current numbers were looked at.
