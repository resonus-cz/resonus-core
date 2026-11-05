"""Time-domain tools for impulse responses: envelope, onset, alignment, windows, normalization.

References
----------
.. [1] ISO 3382-1:2009, Acoustics - Measurement of room acoustic parameters - Part 1:
   Performance spaces, Annex A.3.4 (start of the impulse response).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.signal
from numpy.typing import ArrayLike, NDArray


class OnsetWarning(UserWarning):
    """The impulse response has too little dynamic range for the ISO 3382-1 onset rule."""


def envelope(ir: ArrayLike) -> NDArray[np.float64]:
    """Amplitude envelope ``|scipy.signal.hilbert(ir, axis=0)|``, same shape as ``ir``."""
    x = np.asarray(ir, float)
    return np.abs(scipy.signal.hilbert(x, axis=0))


def onset(
    ir: ArrayLike,
    background_db: float | None = None,
    margin_db: float = 10.0,
    tail_fraction: float = 0.1,
) -> int:
    """Start of an impulse response per ISO 3382-1:2009 Annex A.3.4 [1]_.

    [1]_: the start is "where the signal first rises significantly above the background
    but is more than 20 dB below the maximum". Implementation on the envelope level
    ``L = 20*log10(envelope(ir))`` (``-inf`` where the envelope is 0), max ``Lmax`` at
    index ``p``:

    1. Background ``B``: ``background_db`` if given, else ``10*log10(mean(env**2))`` over
       the last ``tail_fraction`` of the samples (design choice; [1]_ does not specify).
    2. ``T20 = Lmax - 20``; ``TB = B + margin_db`` ("significantly": the margin is a design
       choice; [1]_ does not quantify it).
    3. If ``TB >= T20`` (too little dynamic range) an :class:`OnsetWarning` is issued and a
       compromise is used: ``TB = B + (T20 - B)/2`` if ``B < T20``, else ``TB = T20``.
    4. Walk back from ``p`` to the first index of the contiguous run with ``L >= T20``
       ending at ``p`` (``i20``); then walk back from ``i20`` to the first index of the
       contiguous run with ``L > TB`` ending at ``i20``. That index is returned (0 if a run
       reaches the beginning). Searching backwards from the peak ignores earlier spikes.

    Parameters
    ----------
    ir
        Broadband impulse response, 1-D.
    background_db
        Background level in dB re 1 (envelope level); None: estimate from the tail.
    margin_db
        Required rise above the background in dB, >= 0.
    tail_fraction
        Fraction of samples at the end used for the background estimate, in (0, 1].

    Returns
    -------
    int
        Index of the start of the impulse response.

    Raises
    ------
    ValueError
        If ``ir`` is not 1-D, is empty or all zero, or a parameter is out of range.
    """
    x = np.asarray(ir, float)
    if x.ndim != 1:
        raise ValueError("ir must be 1-D")
    if x.size == 0 or not np.any(x):
        raise ValueError("ir must not be empty or all zero")
    if margin_db < 0:
        raise ValueError("margin_db must be >= 0")
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")

    env = envelope(x)
    with np.errstate(divide="ignore"):
        L = 20 * np.log10(env)
    p = int(np.argmax(env))
    lmax = L[p]

    if background_db is None:
        k = max(1, int(round(tail_fraction * x.size)))
        B = 10 * np.log10(np.mean(env[-k:] ** 2))
    else:
        B = background_db

    t20 = lmax - 20
    tb = B + margin_db
    if tb >= t20:
        warnings.warn(
            f"onset: dynamic range {lmax - B:.1f} dB is too small for ISO 3382-1 A.3.4 "
            f"(background {B:.1f} dB + margin {margin_db:.1f} dB >= Lmax - 20 dB); "
            "using a reduced margin (half the available range) or the -20 dB point",
            OnsetWarning,
            stacklevel=2,
        )
        tb = B + (t20 - B) / 2 if B < t20 else t20

    i = p
    while i > 0 and L[i - 1] >= t20:
        i -= 1
    while i > 0 and L[i - 1] > tb:
        i -= 1
    return i


def align(ir: ArrayLike, start: int, pre: int = 0) -> NDArray[np.float64]:
    """Shift ``ir`` so that sample ``start`` lands at index ``pre``; keeps the length.

    Returns ``ir[start - pre:]`` followed by zeros (along axis 0, 1-D or 2-D input).

    Raises
    ------
    ValueError
        If ``pre < 0``, ``start - pre < 0`` or ``start >= len(ir)``.
    """
    x = np.asarray(ir, float)
    if pre < 0:
        raise ValueError("pre must be >= 0")
    if start - pre < 0:
        raise ValueError("start - pre must be >= 0")
    if start >= x.shape[0]:
        raise ValueError("start must be < len(ir)")
    out = np.zeros_like(x)
    seg = x[start - pre :]
    out[: seg.shape[0]] = seg
    return out


def time_window(
    n_samples: int,
    reference: int,
    left: int,
    right: int,
    left_shape: str = "hann",
    right_shape: str = "hann",
) -> NDArray[np.float64]:
    """Asymmetric window around a reference sample (left and right windows, as in REW).

    The window is 0 before ``reference - left``, rises over ``left`` samples to 1 at
    ``reference``, falls over ``right`` samples and is 0 from ``reference + right`` on.
    Half-window shapes: a symmetric ``w = scipy.signal.windows.get_window(shape,
    2*width + 1, fftbins=False)``, rising half ``w[:width]`` (placed at
    ``reference - width .. reference - 1``), falling half ``w[width + 1:]`` (placed at
    ``reference + 1 .. reference + width``); for ``"rect"`` the halves are
    ``[0, 1, ..., 1]`` and ``[1, ..., 1, 0]``, so for every shape the samples at
    ``reference - left`` and ``reference + right`` are 0.

    Parameters
    ----------
    n_samples
        Window length, > 0.
    reference
        Index where the window is 1 (e.g. the onset or the peak), ``0 <= reference < n``.
    left, right
        Widths in samples, >= 0 (0: hard edge at ``reference``).
    left_shape, right_shape
        ``"rect"`` or a scipy window name (``"hann"``, ``"blackman"``, ...).

    Returns
    -------
    numpy.ndarray
        Window of length ``n_samples``; parts outside the array are clipped.

    Raises
    ------
    ValueError
        On invalid sizes or reference.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if not (0 <= reference < n_samples):
        raise ValueError("reference out of bounds")
    if left < 0 or right < 0:
        raise ValueError("left and right must be >= 0")

    def _halves(shape: str, width: int) -> tuple[np.ndarray, np.ndarray]:
        if shape == "rect":
            w = np.ones(2 * width + 1)
            w[0] = w[-1] = 0
            return w[:width], w[width + 1 :]
        else:
            w = scipy.signal.windows.get_window(shape, 2 * width + 1, fftbins=False)
            return w[:width], w[width + 1 :]

    full = np.zeros(n_samples + left + right)
    r = reference + left
    full[r] = 1.0
    if left:
        rise, _ = _halves(left_shape, left)
        full[r - left : r] = rise
    if right:
        _, fall = _halves(right_shape, right)
        full[r + 1 : r + 1 + right] = fall
    return full[left : left + n_samples]


def normalize(ir: ArrayLike, mode: str = "peak", target: float = 1.0) -> NDArray[np.float64]:
    """Scale an impulse response (all channels jointly, preserving their ratios).

    ``"peak"``: ``max|ir| = target``; ``"rms"``: ``sqrt(mean(ir**2)) = target``;
    ``"energy"``: ``sum(ir**2) = target``.

    Raises
    ------
    ValueError
        On an unknown mode, ``target <= 0`` or an all-zero ``ir``.
    """
    if target <= 0:
        raise ValueError("target must be > 0")
    x = np.asarray(ir, float)
    if not np.any(x):
        raise ValueError("ir must not be all zero")
    if mode == "peak":
        v = np.max(np.abs(x))
    elif mode == "rms":
        v = np.sqrt(np.mean(x**2))
    elif mode == "energy":
        return x * np.sqrt(target / np.sum(x**2))  # sum(out**2) = target
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return x * (target / v)
