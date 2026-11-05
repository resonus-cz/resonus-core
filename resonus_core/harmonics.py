"""Separation of harmonic-distortion impulse responses from exponential-sweep measurements.

After deconvolution of an exponential sweep, the response to the n-th harmonic arrives
``L*ln(n)`` before the linear impulse response ([1]_; [2]_), ``L`` being the sweep rate
(``params["rate"]``). Each one is cut out, aligned to its arrival (including the
fractional-sample part of the delay, as in [2]_ and its reference code), and can be
transformed to a harmonic frequency response ``H_n``.

Harmonic phases are meaningful only for synchronized sweeps ([2]_); with a plain Farina
sweep the magnitudes are correct but the phases carry an offset.

References
----------
.. [1] A. Farina, "Simultaneous measurement of impulse response and distortion with a
   swept-sine technique", AES 108th Convention, preprint 5093, 2000.
.. [2] A. Novák, P. Lotton, L. Simon, "Synchronized swept-sine: Theory, application, and
   implementation", J. Audio Eng. Soc. 63(10), 786-798, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from resonus_core.deconv import Deconvolved
from resonus_core.signals import Excitation


@dataclass(frozen=True)
class Harmonics:
    """Separated harmonic impulse responses.

    Parameters
    ----------
    irs
        ``(n_max, length)`` or ``(n_max, length, n_channels)``; row ``n-1`` is harmonic
        ``n`` (row 0 is the linear impulse response).
    sample_rate
        Sample rate in Hz.
    pre
        Samples kept before each arrival (each segment starts ``pre`` samples early).
    """

    irs: NDArray[np.float64]
    sample_rate: float
    pre: int


def harmonic_delays(excitation: Excitation, n_max: int) -> NDArray[np.float64]:
    """Arrival of harmonics 1..n_max before the linear IR, ``L*ln(n)`` in s ([1]_, [2]_).

    Raises
    ------
    ValueError
        If the excitation is not an exponential sweep or ``n_max < 1``.
    """
    if excitation.kind != "exp_sweep":
        raise ValueError("excitation must be an exponential sweep")
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    return excitation.params["rate"] * np.log(np.arange(1, n_max + 1))


def split_harmonics(
    deconvolved: Deconvolved,
    excitation: Excitation,
    n_max: int,
    length: int,
    pre: int = 0,
) -> Harmonics:
    """Cut out harmonics 1..n_max, each aligned to its arrival.

    For harmonic n with delay ``dt = L*ln(n)*sample_rate`` samples: ``i = round(dt)``,
    ``rem = dt - i``; segment ``signal[zero_index - i - pre : zero_index - i - pre + length]``;
    the fractional delay is removed by multiplying the segment's ``rfft`` by
    ``exp(-1j*rem*w)``, ``w = linspace(0, pi, length//2 + 1)`` (rad/sample), then ``irfft``
    to ``length`` samples ([2]_ reference code).

    Parameters
    ----------
    deconvolved
        Result of :func:`resonus_core.deconv.deconvolve` for ``excitation``.
    excitation
        The exponential sweep used (synchronized or not).
    n_max
        Highest harmonic, >= 1.
    length
        Segment length in samples, > ``pre``.
    pre
        Samples kept before each arrival, >= 0.

    Returns
    -------
    Harmonics

    Raises
    ------
    ValueError
        If the excitation is not an exponential sweep, ``n_max < 1``, ``pre < 0``,
        ``length <= pre``, a segment lies outside the signal, or segments of adjacent
        harmonics would overlap: for every n in 1..n_max, ``pre`` must not exceed the
        spacing to harmonic n+1, and for n >= 2 ``length - pre`` must not exceed the
        spacing to harmonic n-1 (spacings ``(L*ln(n+1) - L*ln(n))*sample_rate`` etc.).
    """
    if pre < 0:
        raise ValueError("pre must be >= 0")
    if length <= pre:
        raise ValueError("length must be > pre")
    fs = deconvolved.sample_rate
    dts = harmonic_delays(excitation, n_max + 1) * fs
    # Overlap checks
    for n in range(1, n_max + 1):
        if pre > dts[n] - dts[n - 1]:
            raise ValueError("overlap")
        if n >= 2 and (length - pre) > dts[n - 1] - dts[n - 2]:
            raise ValueError("overlap")
    w = np.linspace(0, np.pi, length // 2 + 1)
    signal = deconvolved.signal
    segments = []
    for n in range(1, n_max + 1):
        dt = dts[n - 1]
        i = int(round(dt))
        rem = dt - i
        start = deconvolved.zero_index - i - pre
        if start < 0 or start + length > signal.shape[0]:
            raise ValueError("segment out of bounds")
        seg = signal[start : start + length]
        phase = np.exp(-1j * rem * w)
        H = np.fft.rfft(seg, axis=0) * (phase if seg.ndim == 1 else phase[:, None])
        ir = np.fft.irfft(H, length, axis=0)
        segments.append(ir)
    return Harmonics(np.array(segments), fs, pre)


def harmonic_responses(
    harmonics: Harmonics, n_fft: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Frequency responses ``H_n(f)`` of the separated harmonics, referenced to arrival.

    ``H = rfft(irs, n_fft, axis=1) * exp(1j*2*pi*f*pre/sample_rate)`` (broadcast over
    harmonics and channels) so the segment offset ``pre`` adds no linear phase.

    Parameters
    ----------
    harmonics
        Separated harmonics.
    n_fft
        FFT length, >= segment length; None: the segment length.

    Returns
    -------
    frequencies, responses
        ``rfftfreq(n_fft, 1/sample_rate)`` and ``H`` of shape ``(n_max, n_fft//2 + 1)``
        (plus a channel axis for multichannel input).
    """
    irs = harmonics.irs
    n = irs.shape[1] if n_fft is None else n_fft
    f = np.fft.rfftfreq(n, 1 / harmonics.sample_rate)
    H = np.fft.rfft(irs, n, axis=1)
    shift = np.exp(1j * 2 * np.pi * f * harmonics.pre / harmonics.sample_rate)
    if irs.ndim == 3:
        H = H * shift[None, :, None]
    else:
        H = H * shift[None, :]
    return f, H


def thd(
    harmonics: Harmonics,
    frequencies: ArrayLike,
    n_fft: int = 2**14,
    channel: int | None = None,
) -> NDArray[np.float64]:
    """Total harmonic distortion relative to the fundamental, vs fundamental frequency.

    ``THD(f) = sqrt(sum_{n=2..n_max} |H_n(n*f)|**2) / |H_1(f)|`` as a ratio (not %).
    An input component at f produces its n-th harmonic at ``n*f``, which passes through
    ``h_n``; hence ``H_n`` is evaluated at ``n*f`` (derived from the model of [1]_; not a
    quoted definition). Magnitudes are linearly interpolated on the FFT grid; harmonics
    above ``sample_rate/2`` are omitted. This is THD re fundamental (some standards use
    the total RMS instead).

    ``channel`` selects one channel of multichannel harmonics (as returned by
    :func:`resonus_core.measure.measure`, which is always multichannel).

    Raises
    ------
    ValueError
        For multichannel harmonics without ``channel``, an invalid ``channel``, or
        ``n_max < 2``.
    """
    if channel is not None:
        if harmonics.irs.ndim != 3 or not 0 <= channel < harmonics.irs.shape[2]:
            raise ValueError("channel requires multichannel harmonics and a valid index")
        harmonics = Harmonics(harmonics.irs[..., channel], harmonics.sample_rate, harmonics.pre)
    if harmonics.irs.ndim != 2:
        raise ValueError("multichannel harmonics: pass channel=...")
    if harmonics.irs.shape[0] < 2:
        raise ValueError("n_max must be >= 2")
    fr, H = harmonic_responses(harmonics, n_fft)
    mag = np.abs(H)
    f = np.asarray(frequencies, float)
    total = np.zeros_like(f)
    n_max = harmonics.irs.shape[0]
    for n in range(2, n_max + 1):
        fn = n * f
        ok = fn <= harmonics.sample_rate / 2
        total[ok] += np.interp(fn[ok], fr, mag[n - 1]) ** 2
    denom = np.interp(f, fr, mag[0])
    return np.sqrt(total) / denom
