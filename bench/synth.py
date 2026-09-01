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
              "steep_short", "variable", "descent", "switchback")


def _broadband(rng, n, length_m, beta, min_wavelength_m, grade_sd_pct):
    """Zero-mean roughness with a power-law spectrum, built by inverse FFT.

    Amplitude falls as k^-beta, the standard first-order model of natural
    topography, with energy generated down to min_wavelength_m rather than
    to a fixed harmonic count.

    That distinction is not cosmetic. An earlier version of this generator
    summed a fixed 24 harmonics over the route length, so its shortest
    wavelength was length/24: 250 m on a 6 km route. The benchmark
    therefore contained nothing a 120 m representation could
    under-resolve, and a resolution sweep run against it could not have
    detected under-resolution however badly the matcher suffered from it.

    The result is normalized on the standard deviation of its GRADE, not
    of its elevation. A power law otherwise couples two things that must
    be controlled separately: beta sets how energy is distributed across
    scales, while grade_sd_pct sets how rough the route is. Normalizing on
    elevation makes a rougher spectrum also a steeper route, so a sweep
    over beta would confound the two.
    """
    nf = n // 2 + 1
    k = np.arange(nf, dtype=float)
    kmax = max(2.0, length_m / max(min_wavelength_m, 1e-9))
    amp = np.zeros(nf)
    band = (k >= 1.0) & (k <= kmax)
    if not band.any():
        band = (k >= 1.0) & (k <= 2.0)
    amp[band] = k[band] ** (-beta)
    spec = amp * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, nf))
    y = np.fft.irfft(spec, n=n)
    dx = length_m / max(1, n - 1)
    g_sd = float(np.std(np.gradient(y, dx) * 100.0))
    return y / g_sd * grade_sd_pct if g_sd > 0 else y


def terrain(rng, length_m, kind="variable", spacing=10.0, beta=1.45,
            min_wavelength_m=25.0, grade_sd_pct=4.0):
    """Return (cum_dist, elev) for one synthetic route.

    beta sets how grade energy is spread across scales: 1.7 is smooth and
    long-wavelength dominated, 1.1 is rough with substantial structure
    below 100 m. Real running terrain spans that range, so the resolution
    sweep is run across it rather than at one value.

    min_wavelength_m sets the finest structure present. Keep it at or
    above 2.5*spacing so the profile is not aliased at generation time.
    """
    n = max(16, int(length_m / spacing) + 1)
    x = np.linspace(0.0, length_m, n)
    min_wl = max(min_wavelength_m, 2.5 * spacing)
    e = _broadband(rng, n, length_m, beta, min_wl, grade_sd_pct)

    if kind == "smooth_climb":
        e = 0.45 * e + 0.06 * x
    elif kind == "descent":
        e = 0.45 * e - 0.055 * x
    elif kind == "rolling":
        e = 1.25 * e + 0.005 * x
    elif kind == "alternating":
        e = 0.5 * e + 35 * np.sin(2 * np.pi * x / max(400.0, length_m / 6))
    elif kind == "flat_with_pinch":
        mid = length_m * 0.55
        pinch = 45 * np.exp(-0.5 * ((x - mid) / (length_m * 0.06)) ** 2)
        e = 0.25 * e + pinch + 0.004 * x
    elif kind == "steep_short":
        e = 0.3 * e + 0.11 * x
    elif kind == "switchback":
        # Short alternating pitches on a net climb, the shape a benchmark
        # made only of long wavelengths never produces.
        e = 0.5 * e + 0.05 * x + 6.0 * np.sin(2 * np.pi * x / 70.0)
    # "variable" keeps the raw broadband shape
    return x, e + 1500.0


def staircase(length_m, pitch_m, pitch_grade, spacing=2.0, base=1500.0):
    """Alternating steep pitch and flat recovery.

    Built so that a staircase and a uniform climb can be given identical
    length, gain and loss: over each pitch_m + flat_m cycle both rise by
    the same amount. Only their ordered shape differs, and only at scales
    below the cycle length, which makes this the sharpest available probe
    of whether the representation is resolved finely enough.
    """
    n = max(16, int(length_m / spacing) + 1)
    x = np.linspace(0.0, length_m, n)
    cycle = 2.0 * pitch_m
    rise = pitch_m * pitch_grade
    c = np.floor(x / cycle)
    off = x - cycle * c
    e = c * rise + np.where(off < pitch_m, pitch_grade * off, rise)
    return x, e + base


def uniform_climb(length_m, grade, spacing=2.0, base=1500.0):
    """Steady grade, for pairing with staircase()."""
    n = max(16, int(length_m / spacing) + 1)
    x = np.linspace(0.0, length_m, n)
    return x, base + grade * x


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


def quantize_elevation(elev, step_m=1.0):
    """Round elevation to a fixed step.

    Strava serves elevation derived from a DEM, and several sources round
    to whole metres. Quantization is not noise: it is a deterministic
    staircase whose steps are large compared with the elevation change
    across one grade sample, so it can manufacture alternating flat and
    steep samples on a perfectly uniform climb.
    """
    return np.round(np.asarray(elev, dtype=float) / step_m) * step_m


def smooth_elevation(cum_dist, elev, window_m=30.0):
    """Moving-average smoothing of the kind a provider applies before
    serving an elevation stream. Applied to one copy of a positive so the
    benchmark contains pairs that differ by provider processing alone."""
    d = np.asarray(cum_dist, dtype=float)
    e = np.asarray(elev, dtype=float)
    if d.size < 3:
        return e
    dx = float(np.median(np.diff(d)))
    w = max(1, int(round(window_m / max(dx, 1e-9))))
    if w < 2:
        return e
    k = np.ones(w) / w
    pad = w // 2
    padded = np.concatenate([np.full(pad, e[0]), e, np.full(pad, e[-1])])
    out = np.convolve(padded, k, mode="valid")
    return out[:e.size] if out.size >= e.size else np.resize(out, e.size)


def gain_matched_staircase(length_m, total_gain_m, pitch_m, spacing=2.0,
                            base=1500.0):
    """A staircase with exactly `total_gain_m` of ascent over `length_m`.

    Pairs with uniform_climb(length_m, total_gain_m / length_m) to give
    two routes of identical length, identical gain and identical loss
    whose ONLY difference is ordered shape below the cycle length. If the
    representation is too coarse to resolve `pitch_m`, the matcher cannot
    tell them apart, and no other scoring term can rescue it.
    """
    grade = 2.0 * total_gain_m / max(length_m, 1e-9)
    return staircase(length_m, pitch_m, grade, spacing=spacing, base=base)
