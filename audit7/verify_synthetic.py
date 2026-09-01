"""
Is the synthetic terrain what the experiment says it is?

The resolution conclusion rests entirely on the staircase-versus-ramp
probe, because the real streams cannot resolve the scale in question. So
the probe itself has to be verified before its result means anything. The
previous audit built a 500 percent ramp by passing a percent where a
fraction was expected and briefly read the resulting saturated score as a
resolution measurement.

Each check below is independent of bench/synth.py: the properties are
measured from the returned arrays, not asserted from the generator's
parameters.
"""

import numpy as np

from audit7.independent import grade_percent, resample_uniform, vertical_change_m


def measure(d, e):
    """Everything the experiment claims, measured from the arrays."""
    span = float(d[-1] - d[0])
    spacing = float(np.median(np.diff(d)))
    gain, loss = vertical_change_m(d, e, 5.0)
    grid, elev = resample_uniform(d, e, max(spacing, 1.0))
    g = grade_percent(grid, elev)
    return {"span_m": span, "spacing_m": spacing, "gain_m": gain,
            "loss_m": loss, "mean_grade_pct": float(np.mean(g)),
            "grade_sd_pct": float(np.std(g)),
            "net_rise_m": float(e[-1] - e[0])}


def dominant_wavelength_m(d, e):
    """Wavelength carrying the most grade variance, measured by FFT.

    The staircase is supposed to have a cycle of 2 * pitch. If the
    measured dominant wavelength disagrees with that, the experiment is
    not probing the scale it claims to probe.
    """
    spacing = float(np.median(np.diff(d)))
    grid, elev = resample_uniform(d, e, spacing)
    g = grade_percent(grid, elev)
    g = g - g.mean()
    n = len(g)
    if n < 16:
        return float("nan")
    spec = np.abs(np.fft.rfft(g * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=spacing)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    if freqs[k] <= 0:
        return float("nan")
    return float(1.0 / freqs[k])


def nyquist_limit_m(d):
    """Shortest wavelength the sampling can represent."""
    return 2.0 * float(np.median(np.diff(d)))


def grade_variance_below(d, e, wavelength_m):
    """Fraction of grade variance at wavelengths shorter than the given
    one. Refuses to answer below the sampling limit rather than reporting
    the interpolator."""
    spacing = float(np.median(np.diff(d)))
    if wavelength_m < 2.5 * spacing:
        return None
    grid, elev = resample_uniform(d, e, spacing)
    g = grade_percent(grid, elev)
    g = g - g.mean()
    n = len(g)
    spec = np.abs(np.fft.rfft(g * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=spacing)
    spec[0] = 0.0
    tot = spec.sum()
    if tot <= 0:
        return None
    return float(spec[freqs > 1.0 / wavelength_m].sum() / tot)
