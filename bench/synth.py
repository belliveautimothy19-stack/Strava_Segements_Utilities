"""
Ground truth generation for matcher evaluation.

No Strava credentials are available in this environment, so the evaluation
set is synthetic. It is built to be realistic rather than convenient:
elevation is generated from a power-law spectrum (amplitude falling as
k^-beta), which is the standard first-order model of natural topography,
plus an archetype trend. Barometric noise, GPS resampling and uneven
sampling are then applied as separate, individually controllable
transformations so their effects can be attributed.

Every generator is seeded, so the whole evaluation is reproducible.
"""

import numpy as np

ARCHETYPES = ("smooth_climb", "rolling", "alternating", "flat_with_pinch",
              "steep_short", "variable", "descent")


def terrain(rng, length_m, kind="variable", spacing=10.0, beta=1.7):
    """Return (cum_dist, elev) for one synthetic route."""
    n = max(16, int(length_m / spacing) + 1)
    x = np.linspace(0.0, length_m, n)

    # Power-law roughness: natural terrain has more energy at long
    # wavelengths, falling off as roughly k^-1.7.
    e = np.zeros(n)
    for k in range(1, 25):
        amp = rng.normal(0, 1.0) * (k ** -beta) * 60.0
        e += amp * np.sin(2 * np.pi * k * x / length_m
                          + rng.uniform(0, 2 * np.pi))

    if kind == "smooth_climb":
        e = 0.4 * e + 0.06 * x
    elif kind == "descent":
        e = 0.4 * e - 0.055 * x
    elif kind == "rolling":
        e = 1.2 * e + 0.005 * x
    elif kind == "alternating":
        e = 0.5 * e + 35 * np.sin(2 * np.pi * x / max(400.0, length_m / 6))
    elif kind == "flat_with_pinch":
        mid = length_m * 0.55
        pinch = 45 * np.exp(-0.5 * ((x - mid) / (length_m * 0.06)) ** 2)
        e = 0.25 * e + pinch + 0.004 * x
    elif kind == "steep_short":
        e = 0.3 * e + 0.11 * x
    # "variable" keeps the raw power-law shape
    return x, e + 1500.0


def add_baro_noise(elev, rng, sigma=0.6):
    """Barometric jitter. 0.6 m is typical of a consumer altimeter."""
    return elev + rng.normal(0.0, sigma, np.asarray(elev).size)


def resample_at(cum_dist, elev, spacing):
    """Re-record the same route at a different GPS sampling rate."""
    d = np.arange(cum_dist[0], cum_dist[-1] + 1e-9, spacing)
    return d, np.interp(d, cum_dist, elev)


def resample_uneven(cum_dist, elev, rng, mean_spacing=12.0, jitter=0.7):
    """Unevenly sampled recording, as produced by a watch that drops
    points on switchbacks or under tree cover."""
    d = [float(cum_dist[0])]
    while d[-1] < cum_dist[-1]:
        step = mean_spacing * (1.0 + rng.uniform(-jitter, jitter))
        d.append(d[-1] + max(1.0, step))
    d = np.array(d)
    d = d[d <= cum_dist[-1]]
    if d.size < 2 or d[-1] < cum_dist[-1]:
        d = np.append(d, cum_dist[-1])
    return d, np.interp(d, cum_dist, elev)


def scale_length(cum_dist, elev, factor):
    """Same shape, different total length."""
    return np.asarray(cum_dist, dtype=float) * factor, np.asarray(elev)


def perturb_grade(cum_dist, elev, rng, pct):
    """Perturb elevation so grades shift by roughly `pct` percent, without
    changing the route's endpoints."""
    d = np.asarray(cum_dist, dtype=float)
    span = d[-1] - d[0]
    bump = np.zeros_like(d)
    for _ in range(4):
        c = rng.uniform(d[0], d[-1])
        w = span * rng.uniform(0.08, 0.25)
        bump += rng.normal(0, 1) * np.exp(-0.5 * ((d - c) / w) ** 2)
    if np.abs(bump).max() > 0:
        bump = bump / np.abs(bump).max()
    return d, np.asarray(elev) + bump * (pct / 100.0) * span * 0.05


def embed(target_d, target_e, rng, pre_m=2000.0, post_m=1500.0,
          spacing=10.0, kind="rolling"):
    """Splice a target into the middle of a longer route.

    Returns (route_d, route_e, (true_start, true_end)). The joins are made
    continuous in elevation so the splice does not create a step that the
    matcher could key on.
    """
    td = np.asarray(target_d, dtype=float) - target_d[0]
    te = np.asarray(target_e, dtype=float)
    out_d, out_e, start = [], [], 0.0

    if pre_m > 0:
        pd, pe = terrain(rng, pre_m, kind, spacing)
        pe = pe - pe[-1] + te[0]
        out_d.append(pd[:-1])
        out_e.append(pe[:-1])
        start = float(pd[-1])
    out_d.append(td + start)
    out_e.append(te)
    end = start + float(td[-1])

    if post_m > 0:
        sd, se = terrain(rng, post_m, kind, spacing)
        se = se - se[0] + te[-1]
        out_d.append(sd[1:] + end)
        out_e.append(se[1:])

    d = np.concatenate(out_d)
    e = np.concatenate(out_e)
    keep = np.ones(d.size, dtype=bool)
    keep[1:] = np.diff(d) > 0
    return d[keep], e[keep], (start, end)


def shuffle_blocks(cum_dist, elev, rng, n_blocks=6):
    """Reorder the route's blocks.

    This preserves the grade HISTOGRAM almost exactly while destroying the
    ordered shape, so it is the decisive negative control: any matcher
    that leans on aggregate grade composition will call this a match, and
    it must not be one.
    """
    d = np.asarray(cum_dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    idx = np.array_split(np.arange(d.size), n_blocks)
    order = rng.permutation(n_blocks)
    grades, lengths = [], []
    for b in idx:
        if b.size < 2:
            continue
        grades.append(np.diff(e[b]) / np.diff(d[b]))
        lengths.append(np.diff(d[b]))
    g = np.concatenate([grades[i] for i in order if i < len(grades)])
    L = np.concatenate([lengths[i] for i in order if i < len(lengths)])
    nd = np.concatenate([[0.0], np.cumsum(L)])
    ne = np.concatenate([[e[0]], e[0] + np.cumsum(g * L)])
    return nd, ne
