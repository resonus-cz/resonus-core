"""Frequency-domain tools: band window, IIR band filters (forward / zero-phase), pre-equalization.

Band window semantics (decision 5a): the passband is kept exactly (weight 1 between the
edges); the tapers lie outside the band.

IIR filters (decision D-F1): zero-phase filtering (``sosfiltfilt``) applies the filter
twice, so the magnitude response is squared (a Butterworth's -3 dB edges become -6 dB)
and the effective order doubles; use it for spectra and IR display. Forward filtering
keeps the single-pass response and causality and is required for decay analysis.

References
----------
.. [1] A. Farina, "Advancements in impulse response measurements by sine sweeps",
   AES 122nd Convention, Vienna, 2007 (Kirkeby regularized inversion, Eqs. 3-5).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sp_signal
from numpy.typing import ArrayLike, NDArray

from resonus_core.signals import Excitation

# Default regularization of pre-equalization, relative to max|H|^2. Design choice: limits
# the boost at a deep notch to about 1/(2*sqrt(eps)) = 5 (+14 dB) re the peak response.
PREEQ_EPS = 1e-2


def band_window(
    frequencies: ArrayLike, band: tuple[float, float], taper_octaves: float = 1 / 3
) -> NDArray[np.float64]:
    """Weights that keep a band and taper outside it (half-Hann in log frequency).

    ``w = 1`` for ``f_low <= f <= f_high``; for ``f_high < f < f_high*2**t``:
    ``w = 0.5*(1 + cos(pi*(log2(f) - log2(f_high))/t))``; for ``f_low*2**-t < f < f_low``:
    ``w = 0.5*(1 + cos(pi*(log2(f_low) - log2(f))/t))``; 0 elsewhere (f = 0 included when
    ``f_low > 0``). With ``f_low == 0`` there is no low taper; ``t = 0`` gives hard edges.

    Parameters
    ----------
    frequencies
        Frequencies in Hz, >= 0.
    band
        ``(f_low, f_high)``, ``0 <= f_low < f_high``.
    taper_octaves
        Taper width ``t`` in octaves, >= 0.

    Returns
    -------
    numpy.ndarray
        Weights in [0, 1], same shape as ``frequencies``.

    Raises
    ------
    ValueError
        On an invalid band or negative taper.
    """
    f = np.asarray(frequencies, float)
    lo, hi = band
    t = taper_octaves
    if lo < 0 or lo >= hi or t < 0:
        raise ValueError("invalid band or negative taper")
    w = np.zeros_like(f)
    w[(f >= lo) & (f <= hi)] = 1.0
    if t > 0:
        with np.errstate(divide="ignore"):
            l2 = np.log2(f)
        up = (f > hi) & (f < hi * 2**t)
        w[up] = 0.5 * (1 + np.cos(np.pi * (l2[up] - np.log2(hi)) / t))
        if lo > 0:
            dn = (f < lo) & (f > lo * 2**-t)
            w[dn] = 0.5 * (1 + np.cos(np.pi * (np.log2(lo) - l2[dn]) / t))
    return w


def band_sos(
    f_low: float, f_high: float, sample_rate: float, order: int = 4
) -> NDArray[np.float64]:
    """Butterworth band filter as second-order sections (-3 dB at the edges, single pass).

    ``scipy.signal.butter(order, ..., fs=sample_rate, output="sos")``: band-pass for
    ``0 < f_low < f_high < sample_rate/2``; low-pass at ``f_high`` if ``f_low == 0``;
    high-pass at ``f_low`` if ``f_high >= sample_rate/2``.

    Raises
    ------
    ValueError
        If ``order < 1``, ``f_low < 0``, ``f_low >= f_high``, or both edges are trivial
        (``f_low == 0`` and ``f_high >= sample_rate/2``).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if f_low < 0 or f_low >= f_high:
        raise ValueError("invalid band")
    lowpass = f_low == 0
    highpass = f_high >= sample_rate / 2
    if lowpass and highpass:
        raise ValueError("both edges trivial")
    if lowpass:
        sos = sp_signal.butter(order, f_high, btype="lowpass", fs=sample_rate, output="sos")
    elif highpass:
        sos = sp_signal.butter(order, f_low, btype="highpass", fs=sample_rate, output="sos")
    else:
        sos = sp_signal.butter(
            order, [f_low, f_high], btype="bandpass", fs=sample_rate, output="sos"
        )
    return sos


def filter_forward(x: ArrayLike, sos: ArrayLike) -> NDArray[np.float64]:
    """Causal filtering along axis 0 (``scipy.signal.sosfilt``)."""
    return sp_signal.sosfilt(sos, np.asarray(x, float), axis=0)


def filter_zero_phase(x: ArrayLike, sos: ArrayLike) -> NDArray[np.float64]:
    """Zero-phase filtering along axis 0 (``scipy.signal.sosfiltfilt``); squared magnitude."""
    return sp_signal.sosfiltfilt(sos, np.asarray(x, float), axis=0)


def preequalize(
    excitation: Excitation,
    frequencies: ArrayLike,
    response: ArrayLike,
    band: tuple[float, float],
    taper_octaves: float = 1 / 3,
    eps: float = PREEQ_EPS,
) -> Excitation:
    """Pre-equalize an excitation by the inverse of a measured magnitude response.

    Generalizes ``filter_sweep`` of vyhyb/imptube (high cut at ``f_high``, regularized).
    With ``X = rfft(x)`` on ``f = rfftfreq(n, 1/fs)`` and ``M = |response|`` linearly
    interpolated onto ``f`` (``np.interp``, clamped at the ends): magnitude-only Kirkeby
    inverse ([1]_) ``C = M/(M**2 + eps*max(M)**2)``; ``Y = X*C*band_window(f, band,
    taper_octaves)`` (the excitation's phase is kept); ``y = irfft(Y, n)`` rescaled so
    that ``max|y| == max|x|``.

    Parameters
    ----------
    excitation
        Excitation to equalize.
    frequencies, response
        Measured response (complex or magnitude) at the given frequencies (Hz,
        increasing).
    band
        Band to keep, passed to :func:`band_window`.
    taper_octaves
        Taper width in octaves.
    eps
        Relative regularization, > 0.

    Returns
    -------
    Excitation
        ``kind="preequalized"``; params: the original params plus ``source_kind``,
        ``band``, ``taper_octaves``, ``eps``. Deconvolve it with the ``"regularized"``
        method and an explicit band.

    Raises
    ------
    ValueError
        On mismatched lengths, ``eps <= 0`` or an invalid band.
    """
    fm = np.asarray(frequencies, float)
    r = np.asarray(response)
    if fm.shape != r.shape:
        raise ValueError("frequencies and response must have same shape")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    # Validate band via band_window
    band_window(fm, band, taper_octaves)
    x = excitation.signal
    n = x.size
    f = np.fft.rfftfreq(n, 1 / excitation.sample_rate)
    m = np.interp(f, fm, np.abs(r))
    c = m / (m**2 + eps * np.max(m) ** 2)
    y = np.fft.irfft(np.fft.rfft(x) * c * band_window(f, band, taper_octaves), n)
    y *= np.max(np.abs(x)) / np.max(np.abs(y))
    params = dict(
        excitation.params,
        source_kind=excitation.kind,
        band=band,
        taper_octaves=taper_octaves,
        eps=eps,
    )
    return Excitation(y, excitation.sample_rate, "preequalized", params)
