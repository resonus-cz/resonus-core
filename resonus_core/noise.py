"""Transfer-function estimation with noise excitation.

Two approaches:

- **Periodic noise** (``periodic=True`` noise or MLS) with period N: record several
  periods, discard the first (settling), average the rest synchronously
  (:func:`synchronous_average`, noise power falls by the number of averages K), and
  deconvolve one period circularly (:func:`periodic_deconvolve`; leakage-free, no
  window). The period sets the frequency resolution ``fs/N`` and the longest impulse
  response without time aliasing.
- **Random noise**: the H1 estimator from Welch-averaged spectra (:func:`h1_estimate`):
  ``H1 = G_xy/G_xx`` and coherence ``|G_xy|**2/(G_xx*G_yy)``; segment length sets the
  resolution, the number of segments the averaging.

References
----------
.. [1] ISO 10534-2, Determination of sound absorption coefficient and impedance in
   impedance tubes - Transfer-function method (transfer function from averaged cross-
   and auto-spectra).
.. [2] P. D. Welch, "The use of fast Fourier transform for the estimation of power
   spectra", IEEE Trans. Audio Electroacoust. 15(2), 70-73, 1967.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike, NDArray

from resonus_core.deconv import Deconvolved
from resonus_core.signals import Excitation


def synchronous_average(
    recorded: ArrayLike, period: int, n_discard: int = 1
) -> NDArray[np.float64]:
    """Average whole periods of a recording of periodic excitation.

    Parameters
    ----------
    recorded
        Recording ``(n,)`` or ``(n, n_channels)``.
    period
        Period length in samples, >= 1.
    n_discard
        Leading periods to drop (system settling), >= 0.

    Returns
    -------
    numpy.ndarray
        Mean of the ``K = n//period - n_discard`` following whole periods, shape
        ``(period,)`` or ``(period, n_channels)``. Incomplete trailing samples are ignored.

    Raises
    ------
    ValueError
        If ``period < 1``, ``n_discard < 0`` or ``K < 1``.
    """
    y = np.asarray(recorded, float)
    if period < 1:
        raise ValueError("period must be >= 1")
    if n_discard < 0:
        raise ValueError("n_discard must be >= 0")
    k = y.shape[0] // period - n_discard
    if k < 1:
        raise ValueError("not enough periods after discarding")
    seg = y[n_discard * period : (n_discard + k) * period]
    seg = seg.reshape((k, period) + y.shape[1:])
    return seg.mean(axis=0)


def periodic_deconvolve(averaged: ArrayLike, excitation: Excitation) -> Deconvolved:
    """Circular deconvolution of one averaged period: ``h = irfft(rfft(y)/rfft(x))``.

    Bins where ``|X| <= 1e-9 * max|X|`` (not excited) get ``H = 0``: the periodic noise
    generators do not excite DC, so the impulse response then has its mean removed.

    Parameters
    ----------
    averaged
        One period of the response, ``(N,)`` or ``(N, n_channels)``, N = excitation length.
    excitation
        Periodic excitation: noise with ``params["periodic"]`` True, or ``kind == "mls"``.

    Returns
    -------
    Deconvolved
        ``zero_index = 0``, ``method = "periodic"``; circular: content at negative times
        wraps to the end.

    Raises
    ------
    ValueError
        If the excitation is not periodic or the lengths differ.
    """
    if excitation.kind != "mls" and not excitation.params.get("periodic", False):
        raise ValueError("excitation must be periodic")
    y = np.asarray(averaged, float)
    n = excitation.n_samples
    if y.shape[0] != n:
        raise ValueError("averaged length must match excitation length")
    y2 = y[:, None] if y.ndim == 1 else y
    X = np.fft.rfft(excitation.signal)
    ok = np.abs(X) > 1e-9 * np.max(np.abs(X))
    inv = np.zeros_like(X)
    inv[ok] = 1 / X[ok]
    h = np.fft.irfft(np.fft.rfft(y2, axis=0) * inv[:, None], n, axis=0)
    return Deconvolved(h[:, 0] if y.ndim == 1 else h, excitation.sample_rate, 0, "periodic")


def h1_estimate(
    reference: ArrayLike,
    response: ArrayLike,
    sample_rate: float,
    nperseg: int,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.float64]]:
    """H1 transfer function and coherence from Welch-averaged spectra ([1]_, [2]_).

    ``H1 = Pxy/Pxx`` and ``coherence = |Pxy|**2/(Pxx*Pyy)`` with
    ``scipy.signal.csd(x, y, ...)`` (which is ``E[conj(X)*Y]``) and ``scipy.signal.welch``,
    all with the same ``fs``, ``window``, ``nperseg`` and ``noverlap`` (None: scipy's
    default, ``nperseg//2``).

    Parameters
    ----------
    reference
        Excitation or reference channel x, 1-D.
    response
        Response y, ``(n,)`` or ``(n, n_channels)``, same length as ``reference``.
    sample_rate
        Sample rate in Hz.
    nperseg
        Segment length (frequency resolution ``sample_rate/nperseg``).
    noverlap
        Overlap in samples.
    window
        Window name for scipy.

    Returns
    -------
    frequencies, h1, coherence
        ``h1`` and ``coherence`` have shape ``(bins,)`` or ``(bins, n_channels)``.

    Raises
    ------
    ValueError
        If ``reference`` is not 1-D or the lengths differ.
    """
    x = np.asarray(reference, float)
    y = np.asarray(response, float)
    if x.ndim != 1:
        raise ValueError("reference must be 1-D")
    if y.shape[0] != x.size:
        raise ValueError("reference and response must have same length")
    y2 = y[:, None] if y.ndim == 1 else y
    kw = dict(fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)
    f, pxx = scipy.signal.welch(x, **kw)
    _, pxy = scipy.signal.csd(x[:, None], y2, axis=0, **kw)
    _, pyy = scipy.signal.welch(y2, axis=0, **kw)
    h1 = pxy / pxx[:, None]
    coh = np.abs(pxy) ** 2 / (pxx[:, None] * pyy)
    if y.ndim == 1:
        return f, h1[:, 0], coh[:, 0]
    return f, h1, coh
