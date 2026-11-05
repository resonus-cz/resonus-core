"""Excitation signals for impulse-response and transfer-function measurements.

Levels are stated explicitly, never as bare "dBFS":

- ``peak_dbfs``: peak amplitude re full-scale amplitude 1, ``20*log10(max|x|)``
  (sweeps, MLS);
- ``rms_db``: ``20*log10(rms(x))`` re amplitude 1 (noise). This differs from AES17
  "dBFS", which references a full-scale sine (rms 1/sqrt(2)), by 3.01 dB.

References
----------
.. [1] A. Farina, "Simultaneous measurement of impulse response and distortion with a
   swept-sine technique", AES 108th Convention, preprint 5093, 2000.
.. [2] S. Müller, P. Massarani, "Transfer-function measurement with sweeps",
   J. Audio Eng. Soc. 49(6), 2001.
.. [3] A. Novák, P. Lotton, L. Simon, "Synchronized swept-sine: Theory, application, and
   implementation", J. Audio Eng. Soc. 63(10), 786-798, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.signal as sp_signal
from numpy.typing import NDArray


@dataclass(frozen=True)
class Excitation:
    """A generated excitation signal and the parameters that define it.

    Parameters
    ----------
    signal
        Samples, 1-D float64.
    sample_rate
        Sample rate in Hz.
    kind
        One of ``"exp_sweep"``, ``"linear_sweep"``, ``"white_noise"``, ``"pink_noise"``,
        ``"mls"``.
    params
        Generator parameters (needed later for deconvolution).
    """

    signal: NDArray[np.float64]
    sample_rate: float
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.signal.size)

    @property
    def duration(self) -> float:
        """Duration in s."""
        return self.signal.size / self.sample_rate


def exp_sweep(
    f_start: float,
    f_stop: float,
    duration: float,
    sample_rate: float,
    peak_dbfs: float = -6.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
    synchronized: bool = False,
) -> Excitation:
    """Exponential (logarithmic) sine sweep, Farina [1]_, optionally synchronized [3]_.

    Parameters
    ----------
    f_start, f_stop
        Start and stop frequency in Hz, ``0 < f_start < f_stop <= sample_rate/2``.
    duration
        Duration in s, > 0; ``n = round(duration*sample_rate)`` samples.
    sample_rate
        Sample rate in Hz, > 0.
    peak_dbfs
        Peak amplitude in dB re 1, <= 0.
    fade_in, fade_out
        Raised-cosine fade lengths in s, >= 0, together <= duration.
    synchronized
        False (default): ``scipy.signal.chirp`` sweep (Farina). True: synchronized swept
        sine [3]_ Eqs. 31-33: ``k = round(f_start*duration/ln(f_stop/f_start))`` (>= 1),
        ``L = k/f_start``, exact duration ``T = k*ln(f_stop/f_start)/f_start``,
        ``x(t) = sin(2*pi*f_start*L*exp(t/L))`` (phase zero at t = 0, harmonics
        phase-aligned); ``duration`` is then only the requested approximate length.

    Returns
    -------
    Excitation
        ``kind="exp_sweep"``; params: f_start, f_stop, duration, peak_dbfs, fade_in,
        fade_out and ``rate = duration/ln(f_stop/f_start)`` (Farina's L, in s).
        Synchronized: ``duration = T`` (actual), ``rate = L``, plus
        ``requested_duration`` and ``synchronized=True``.

    Raises
    ------
    ValueError
        On invalid parameters.
    """
    if synchronized:
        return _sync_sweep(f_start, f_stop, duration, sample_rate, peak_dbfs, fade_in, fade_out)
    return _sweep(
        "logarithmic",
        "exp_sweep",
        f_start,
        f_stop,
        duration,
        sample_rate,
        peak_dbfs,
        fade_in,
        fade_out,
    )


def linear_sweep(
    f_start: float,
    f_stop: float,
    duration: float,
    sample_rate: float,
    peak_dbfs: float = -6.0,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> Excitation:
    """Linear sine sweep [2]_; parameters and validation as :func:`exp_sweep`.

    ``f_start`` may be 0 here. Returns ``kind="linear_sweep"`` with params f_start,
    f_stop, duration, peak_dbfs, fade_in, fade_out.
    """
    return _sweep(
        "linear",
        "linear_sweep",
        f_start,
        f_stop,
        duration,
        sample_rate,
        peak_dbfs,
        fade_in,
        fade_out,
    )


def white_noise(
    n_samples: int,
    sample_rate: float,
    rms_db: float = -20.0,
    periodic: bool = False,
    seed: int = 0,
) -> Excitation:
    """White noise with exact RMS level.

    Parameters
    ----------
    n_samples
        Length (for ``periodic=True`` this is the period), > 1.
    sample_rate
        Sample rate in Hz, > 0.
    rms_db
        RMS level in dB re 1. Default -20 dB leaves headroom for the Gaussian crest
        factor (peaks of ~4.6 sigma over 2**18 samples).
    periodic
        False: Gaussian random noise. True: flat-magnitude spectrum with random phases
        over exactly ``n_samples`` (periodic pseudo-random noise, leakage-free).
    seed
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    Excitation
        ``kind="white_noise"``; params: rms_db, periodic, seed.

    Raises
    ------
    ValueError
        On invalid parameters, or if the scaled signal would clip (any ``|x| > 1``).
    """
    return _noise("white_noise", n_samples, sample_rate, rms_db, periodic, seed)


def pink_noise(
    n_samples: int,
    sample_rate: float,
    rms_db: float = -20.0,
    periodic: bool = False,
    seed: int = 0,
) -> Excitation:
    """Pink noise (power spectral density proportional to 1/f), exact RMS level.

    Shaped in the frequency domain over the whole length (decision D-N1): spectrum
    magnitude proportional to ``1/sqrt(f)`` for f > 0, zero at DC. Parameters, returns
    (``kind="pink_noise"``) and errors as :func:`white_noise`.
    """
    return _noise("pink_noise", n_samples, sample_rate, rms_db, periodic, seed)


def mls(n_bits: int, sample_rate: float, peak_dbfs: float = -6.0) -> Excitation:
    """Maximum-length sequence of ``2**n_bits - 1`` samples, values ``±10**(peak_dbfs/20)``.

    Generated with ``scipy.signal.max_len_seq`` (default state and taps). ``n_bits`` in
    2..32. Returns ``kind="mls"`` with params n_bits, peak_dbfs.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if not 2 <= n_bits <= 32:
        raise ValueError("n_bits must be in 2..32")
    seq = sp_signal.max_len_seq(n_bits)[0]
    x = 10 ** (peak_dbfs / 20) * (2.0 * seq - 1.0)
    return Excitation(x, sample_rate, "mls", {"n_bits": n_bits, "peak_dbfs": peak_dbfs})


def _sweep(
    method: str,
    kind: str,
    f_start: float,
    f_stop: float,
    duration: float,
    sample_rate: float,
    peak_dbfs: float,
    fade_in: float,
    fade_out: float,
) -> Excitation:
    """Generate an exponential or linear sweep with ``scipy.signal.chirp`` (shared code)."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if method == "logarithmic" and f_start <= 0:
        raise ValueError("f_start must be > 0 for an exponential sweep")
    if f_start < 0:
        raise ValueError("f_start must be >= 0")
    if not f_start < f_stop <= sample_rate / 2:
        raise ValueError("need f_start < f_stop <= sample_rate/2")
    if peak_dbfs > 0:
        raise ValueError("peak_dbfs must be <= 0")
    if fade_in < 0 or fade_out < 0 or fade_in + fade_out > duration:
        raise ValueError("fades must be >= 0 and fade_in + fade_out <= duration")
    n = round(duration * sample_rate)
    t = np.arange(n) / sample_rate
    x = sp_signal.chirp(t, f0=f_start, t1=duration, f1=f_stop, method=method, phi=-90)
    x *= 10 ** (peak_dbfs / 20)
    _apply_fades(x, round(fade_in * sample_rate), round(fade_out * sample_rate))
    params: dict[str, Any] = {
        "f_start": f_start,
        "f_stop": f_stop,
        "duration": duration,
        "peak_dbfs": peak_dbfs,
        "fade_in": fade_in,
        "fade_out": fade_out,
    }
    if method == "logarithmic":
        params["rate"] = duration / np.log(f_stop / f_start)
    return Excitation(x, sample_rate, kind, params)


def _sync_sweep(
    f_start: float,
    f_stop: float,
    duration: float,
    sample_rate: float,
    peak_dbfs: float,
    fade_in: float,
    fade_out: float,
) -> Excitation:
    """Generate a synchronized exponential sweep (Novák)."""
    # Validate parameters using _sweep with zero fades
    _sweep(
        "logarithmic",
        "exp_sweep",
        f_start,
        f_stop,
        duration,
        sample_rate,
        peak_dbfs,
        0.0,
        0.0,
    )
    k = round(f_start * duration / np.log(f_stop / f_start))
    if k < 1:
        raise ValueError(
            "duration too short to synchronize: round(f_start*duration/ln(f_stop/f_start)) < 1"
        )
    L = k / f_start
    T = k * np.log(f_stop / f_start) / f_start
    if fade_in < 0 or fade_out < 0 or fade_in + fade_out > T:
        raise ValueError("fades must be >= 0 and fade_in + fade_out <= duration")
    n = round(T * sample_rate)
    t = np.arange(n) / sample_rate
    x = 10 ** (peak_dbfs / 20) * np.sin(2 * np.pi * f_start * L * np.exp(t / L))
    _apply_fades(x, round(fade_in * sample_rate), round(fade_out * sample_rate))
    params: dict[str, Any] = {
        "f_start": f_start,
        "f_stop": f_stop,
        "duration": T,
        "requested_duration": duration,
        "peak_dbfs": peak_dbfs,
        "fade_in": fade_in,
        "fade_out": fade_out,
        "rate": L,
        "synchronized": True,
    }
    return Excitation(x, sample_rate, "exp_sweep", params)


def _apply_fades(x: NDArray[np.float64], m: int, k: int) -> None:
    """Raised-cosine fade-in over ``m`` and fade-out over ``k`` samples, in place."""
    if m > 0:
        x[:m] *= (1 - np.cos(np.pi * np.arange(m) / m)) / 2
    if k > 0:
        x[-k:] *= (1 + np.cos(np.pi * np.arange(k) / k)) / 2


def _noise(
    kind: str, n_samples: int, sample_rate: float, rms_db: float, periodic: bool, seed: int
) -> Excitation:
    """Generate white or pink noise with exact RMS (shared code)."""
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    rng = np.random.default_rng(seed)
    if periodic:
        k = np.arange(1, n_samples // 2 + 1)  # all non-DC bins, incl. Nyquist for even n
        spectrum = np.zeros(n_samples // 2 + 1, dtype=np.complex128)
        spectrum[1:] = np.exp(1j * rng.uniform(0, 2 * np.pi, k.size))
        if n_samples % 2 == 0:  # the Nyquist bin of a real signal is real: +1 or -1
            spectrum[-1] = np.sign(spectrum[-1].real) or 1.0
        if kind == "pink_noise":
            spectrum[1:] /= np.sqrt(k)
        x = np.fft.irfft(spectrum, n_samples)
    else:
        x = rng.standard_normal(n_samples)
        if kind == "pink_noise":
            w = np.fft.rfft(x)
            f = np.fft.rfftfreq(n_samples, 1 / sample_rate)
            w[0] = 0
            w[1:] /= np.sqrt(f[1:])
            x = np.fft.irfft(w, n_samples)
    x *= 10 ** (rms_db / 20) / np.sqrt(np.mean(x**2))
    if np.max(np.abs(x)) > 1:
        raise ValueError("the noise would clip at this rms_db; lower the level")
    return Excitation(x, sample_rate, kind, {"rms_db": rms_db, "periodic": periodic, "seed": seed})
