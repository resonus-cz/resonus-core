"""Deconvolution of recorded responses into impulse responses.

Two methods share one result layout (:class:`Deconvolved`): the linear impulse response
starts at ``zero_index``; samples before it are negative times, where an exponential
sweep places its harmonic-distortion products.

- ``"inverse"``: aperiodic convolution with Farina's inverse filter (sweeps only) [1]_ [2]_.
- ``"regularized"``: spectral division with frequency-dependent regularization
  (Kirkeby's method as given in [2]_, Eqs. 3-5), for any excitation.
- ``"novak"``: analytic inverse of the synchronized swept sine ([3]_ Eq. 43).

References
----------
.. [1] A. Farina, "Simultaneous measurement of impulse response and distortion with a
   swept-sine technique", AES 108th Convention, preprint 5093, 2000.
.. [2] A. Farina, "Advancements in impulse response measurements by sine sweeps",
   AES 122nd Convention, Vienna, 2007.
.. [3] A. Novák, P. Lotton, L. Simon, "Synchronized swept-sine: Theory, application, and
   implementation", J. Audio Eng. Soc. 63(10), 786-798, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from resonus_core.signals import Excitation

# Default regularization levels relative to max|X|^2. Design choice (not from a source):
# [2] only requires "very small inside the swept range, much larger outside".
EPS_IN = 1e-6
EPS_OUT = 1.0


@dataclass(frozen=True)
class Deconvolved:
    """Result of a deconvolution.

    Parameters
    ----------
    signal
        Deconvolved signal, ``(m,)`` or ``(m, n_channels)``, float64.
    sample_rate
        Sample rate in Hz.
    zero_index
        Index of t = 0 (start of the linear impulse response).
    method
        ``"inverse"``, ``"regularized"`` or ``"novak"``.

    Notes
    -----
    Limiting the response to the excitation band makes it nearly zero-phase: its ringing
    is symmetric around t = 0, so part of the energy lies at negative times. An impulse
    response cut at t = 0 (:meth:`linear_ir`) loses that pre-ringing, most noticeably near
    the band edges (about 0.75 dB in band for a 20 Hz-20 kHz sweep at 48 kHz with a
    4096-sample cut). Pre-ringing is discussed in [2]_.
    """

    signal: NDArray[np.float64]
    sample_rate: float
    zero_index: int
    method: str

    @property
    def time(self) -> NDArray[np.float64]:
        """Time axis in s; negative before ``zero_index``."""
        return (np.arange(self.signal.shape[0]) - self.zero_index) / self.sample_rate

    def linear_ir(self, length: int) -> NDArray[np.float64]:
        """Return ``length`` samples starting at t = 0.

        Raises
        ------
        ValueError
            If ``length <= 0`` or the signal has fewer samples after ``zero_index``.
        """
        if length <= 0:
            raise ValueError("length must be > 0")
        if self.zero_index + length > self.signal.shape[0]:
            raise ValueError("requested length exceeds signal")
        return self.signal[self.zero_index : self.zero_index + length]


def inverse_filter(excitation: Excitation) -> NDArray[np.float64]:
    """Farina's inverse filter of an exponential or linear sweep [1]_ [2]_.

    The time-reversed sweep; for an exponential sweep multiplied by ``exp(-tau/L)``
    (``tau`` = time from the start of the filter, ``L = params["rate"]``), so the filter's
    spectrum rises +3 dB/octave and compensates the sweep's -3 dB/octave. The filter is
    scaled so that the median of ``|rfft(x)*rfft(f)|`` over the central half of the sweep
    range is 1 (central half in log frequency for exponential, in linear frequency for
    linear sweeps).

    Parameters
    ----------
    excitation
        An ``"exp_sweep"`` or ``"linear_sweep"`` excitation.

    Returns
    -------
    numpy.ndarray
        Inverse filter, same length as the sweep.

    Raises
    ------
    ValueError
        For other excitation kinds.
    """
    if excitation.kind not in ("exp_sweep", "linear_sweep"):
        raise ValueError("inverse_filter only supports sweeps")
    x = excitation.signal
    n = x.size
    tau = np.arange(n) / excitation.sample_rate
    f = x[::-1].copy()
    if excitation.kind == "exp_sweep":
        f *= np.exp(-tau / excitation.params["rate"])
    nfft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    g = np.abs(np.fft.rfft(x, nfft) * np.fft.rfft(f, nfft))
    fr = np.fft.rfftfreq(nfft, 1 / excitation.sample_rate)
    if excitation.kind == "exp_sweep":
        lo = (
            excitation.params["f_start"]
            * (excitation.params["f_stop"] / excitation.params["f_start"]) ** 0.25
        )
        hi = (
            excitation.params["f_start"]
            * (excitation.params["f_stop"] / excitation.params["f_start"]) ** 0.75
        )
    else:  # linear
        d = excitation.params["f_stop"] - excitation.params["f_start"]
        lo = excitation.params["f_start"] + d / 4
        hi = excitation.params["f_start"] + 3 * d / 4
    mask = (fr >= lo) & (fr <= hi)
    median = np.median(g[mask])
    return f / median


def regularized_inverse_spectrum(
    spectrum: ArrayLike,
    frequencies: ArrayLike,
    band: tuple[float, float],
    eps_in: float = EPS_IN,
    eps_out: float = EPS_OUT,
    transition: float = 1 / 3,
) -> NDArray[np.complex128]:
    """Kirkeby regularized inverse ``C = conj(X)/(conj(X)*X + eps(f))``, [2]_ Eqs. 3-5.

    ``eps(f) = level(f) * max|X|^2`` with ``level = eps_in`` inside ``band`` (edges
    included), ``eps_out`` at and beyond ``transition`` octaves outside each edge, and
    ``log10(level)`` linear in ``log2(f)`` in between. With ``band[0] == 0`` there is no
    low-side transition; f = 0 is outside the band whenever ``band[0] > 0``.

    Parameters
    ----------
    spectrum
        Complex spectrum ``X`` of the excitation (one-sided).
    frequencies
        Frequencies of the bins in Hz, same length.
    band
        ``(f_low, f_high)`` in Hz, ``0 <= f_low < f_high``.
    eps_in, eps_out
        Relative regularization inside / outside the band, ``0 < eps_in <= eps_out``.
    transition
        Width of each transition zone in octaves, > 0.

    Returns
    -------
    numpy.ndarray
        Complex inverse spectrum ``C``.

    Raises
    ------
    ValueError
        On invalid band, levels, transition or mismatched lengths.
    """
    X = np.asarray(spectrum, dtype=np.complex128)
    f = np.asarray(frequencies, dtype=np.float64)
    if X.shape != f.shape:
        raise ValueError("spectrum and frequencies must have same shape")
    f_low, f_high = band
    if not (0 <= f_low < f_high):
        raise ValueError("band must satisfy 0 <= f_low < f_high")
    if not (0 < eps_in <= eps_out):
        raise ValueError("eps_in must be > 0 and <= eps_out")
    if transition <= 0:
        raise ValueError("transition must be > 0")
    max_abs_sq = np.max(np.abs(X) ** 2)
    # base level
    a = np.log10(eps_in)
    b = np.log10(eps_out)
    level = np.full_like(f, b, dtype=np.float64)
    # inside band
    inside = (f >= f_low) & (f <= f_high)
    level[inside] = a
    # transition above band
    above = (f > f_high) & (f < f_high * 2**transition)
    if np.any(above):
        logf = np.log2(f[above])
        logfh = np.log2(f_high)
        level[above] = a + (b - a) * (logf - logfh) / transition
    # transition below band
    if f_low > 0:
        below = (f > f_low * 2**-transition) & (f < f_low)
        if np.any(below):
            logf = np.log2(f[below])
            logfl = np.log2(f_low)
            level[below] = a + (b - a) * (logfl - logf) / transition
    eps = 10**level * max_abs_sq
    return np.conj(X) / (np.abs(X) ** 2 + eps)


def deconvolve(
    recorded: ArrayLike,
    excitation: Excitation,
    method: str | None = None,
    band: tuple[float, float] | None = None,
    eps_in: float = EPS_IN,
    eps_out: float = EPS_OUT,
    transition: float = 1 / 3,
) -> Deconvolved:
    """Deconvolve a recording made with ``excitation`` into an impulse response.

    Parameters
    ----------
    recorded
        Recording, ``(n,)`` or ``(n, n_channels)``, ``n >= excitation.n_samples``
        (record a silent tail after the excitation to capture the decay).
    excitation
        The played excitation.
    method
        ``"inverse"`` (sweeps only), ``"regularized"`` or ``"novak"`` (synchronized
        sweeps only); None: ``"novak"`` for synchronized sweeps, ``"inverse"`` for other
        sweeps, ``"regularized"`` otherwise.
    band
        Regularization band in Hz; None: ``(f_start, f_stop)`` for sweeps,
        ``(0, sample_rate/2)`` otherwise.
    eps_in, eps_out, transition
        Passed to :func:`regularized_inverse_spectrum`.

    Returns
    -------
    Deconvolved
        ``zero_index = excitation.n_samples - 1`` for both methods; the output has the
        same number of dimensions as ``recorded``.

    Raises
    ------
    ValueError
        On a too short recording, unknown method, ``"inverse"`` with a non-sweep, or
        ``"novak"`` with a sweep that is not synchronized.
    """
    y = np.asarray(recorded, dtype=float)
    if y.shape[0] < excitation.n_samples:
        raise ValueError("recorded too short")
    sweep = excitation.kind in ("exp_sweep", "linear_sweep")
    sync = excitation.params.get("synchronized", False)
    if method is None:
        if sync:
            method = "novak"
        else:
            method = "inverse" if sweep else "regularized"
    if method not in ("inverse", "regularized", "novak"):
        raise ValueError("unknown method")
    if method == "inverse" and not sweep:
        raise ValueError("inverse requires a sweep")
    if method == "novak" and not sync:
        raise ValueError("novak requires a synchronized sweep")
    fs = excitation.sample_rate
    nx = excitation.n_samples
    if method == "inverse":
        f = inverse_filter(excitation)
        if y.ndim == 1:
            h = fftconvolve(y, f, axes=0)
        else:
            h = fftconvolve(y, f[:, None], axes=0)
    elif method == "regularized":
        if band is None:
            if sweep:
                band = (excitation.params["f_start"], excitation.params["f_stop"])
            else:
                band = (0.0, fs / 2)
        nfft = 2 ** int(np.ceil(np.log2(nx + y.shape[0] - 1)))
        X = np.fft.rfft(excitation.signal, nfft)
        C = regularized_inverse_spectrum(
            X, np.fft.rfftfreq(nfft, 1 / fs), band, eps_in, eps_out, transition
        )
        Y = np.fft.rfft(y, nfft, axis=0)
        h = np.fft.irfft(Y * (C if y.ndim == 1 else C[:, None]), nfft, axis=0)
        h = np.roll(h, nx - 1, axis=0)
    else:  # novak
        nfft = 2 ** int(np.ceil(np.log2(nx + y.shape[0] - 1)))
        si = novak_inverse_spectrum(excitation, np.fft.rfftfreq(nfft, 1 / fs))
        h = np.fft.irfft(
            np.fft.rfft(y, nfft, axis=0) / fs * (si if y.ndim == 1 else si[:, None]),
            nfft,
            axis=0,
        )
        h = np.roll(h, nx - 1, axis=0)
    return Deconvolved(h, fs, nx - 1, method)


def novak_inverse_spectrum(
    excitation: Excitation, frequencies: ArrayLike
) -> NDArray[np.complex128]:
    """Analytic inverse spectrum of a synchronized swept sine, [3]_ Eq. 43.

    ``SI(f) = 2*sqrt(f/L)*exp(-1j*2*pi*f*L*(1 - ln(f/f_start)) + 1j*pi/4) / A`` for f > 0,
    ``SI(0) = 0``, with ``L = params["rate"]`` and ``A = 10**(peak_dbfs/20)`` (the sweep
    amplitude). The impulse response is ``irfft(rfft(y)/sample_rate * SI)``.

    Parameters
    ----------
    excitation
        A synchronized exponential sweep (``params["synchronized"]`` is True).
    frequencies
        Frequencies in Hz (>= 0).

    Returns
    -------
    numpy.ndarray
        Complex inverse spectrum.

    Raises
    ------
    ValueError
        If the excitation is not a synchronized sweep.
    """
    if not excitation.params.get("synchronized", False):
        raise ValueError("novak_inverse_spectrum only supports synchronized sweeps")
    f = np.asarray(frequencies, float)
    L = excitation.params["rate"]
    f1 = excitation.params["f_start"]
    a = 10 ** (excitation.params["peak_dbfs"] / 20)
    out = np.zeros(f.shape, dtype=np.complex128)
    pos = f > 0
    out[pos] = (
        2
        * np.sqrt(f[pos] / L)
        * np.exp(-1j * 2 * np.pi * f[pos] * L * (1 - np.log(f[pos] / f1)) + 1j * np.pi / 4)
        / a
    )
    return out
