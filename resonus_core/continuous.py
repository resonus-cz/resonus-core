"""Continuous impulse-response measurement with periodic noise.

The periodic excitation (periodic white/pink noise or MLS) is played in a loop
(:func:`cyclic_block` provides playback blocks). Recorded blocks of any size are fed to
:class:`ContinuousAnalyzer`, which cuts them into whole periods, discards the settling
periods, updates the average after every period and deconvolves it on request
(:func:`resonus_core.noise.periodic_deconvolve`).

The period sets the frequency resolution and the longest IR without time aliasing; the
averaging trades measurement time for signal-to-noise ratio. The exponential mode
tracks changes; its weighting (cumulative mean for the first ``n_exponential`` periods,
then a fixed weight ``1/n_exponential``) is a design choice of this library, not taken
from a published method.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from resonus_core.deconv import Deconvolved
from resonus_core.noise import periodic_deconvolve
from resonus_core.signals import Excitation


def cyclic_block(signal: ArrayLike, start: int, n_samples: int) -> NDArray[np.float64]:
    """Return ``n_samples`` of a periodic signal starting at sample ``start`` (wrapping).

    ``signal[(start + arange(n_samples)) % len(signal)]``.

    Raises
    ------
    ValueError
        If the signal is empty or ``n_samples < 0``.
    """
    s = np.asarray(signal, float)
    if s.size == 0:
        raise ValueError("signal must not be empty")
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    return s[(start + np.arange(n_samples)) % s.size]


class ContinuousAnalyzer:
    """Running synchronous average and impulse response of a looped periodic excitation.

    Parameters
    ----------
    excitation
        Periodic excitation (noise with ``params["periodic"]`` True, or MLS); its length
        is the period N.
    n_channels
        Number of recorded channels, >= 1.
    averaging
        ``"linear"``: cumulative mean of all periods. ``"exponential"``: weight
        ``1/min(k, n_exponential)`` for the k-th averaged period.
    n_exponential
        Effective number of averages of the exponential mode, >= 1.
    n_discard
        Leading periods ignored (system settling), >= 0.

    Raises
    ------
    ValueError
        On a non-periodic excitation or invalid parameters.
    """

    def __init__(
        self,
        excitation: Excitation,
        n_channels: int = 1,
        averaging: str = "linear",
        n_exponential: int = 8,
        n_discard: int = 1,
    ) -> None:
        if excitation.kind != "mls" and not excitation.params.get("periodic", False):
            raise ValueError("excitation must be periodic")
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if averaging not in ("linear", "exponential"):
            raise ValueError("averaging must be 'linear' or 'exponential'")
        if n_exponential < 1:
            raise ValueError("n_exponential must be >= 1")
        if n_discard < 0:
            raise ValueError("n_discard must be >= 0")
        self.excitation = excitation
        self.n_channels = n_channels
        self.averaging = averaging
        self.n_exponential = n_exponential
        self.n_discard = n_discard
        self._buffer: NDArray[np.float64] = np.zeros((0, n_channels))
        self._average: NDArray[np.float64] | None = None
        self._count = 0
        self._discarded = 0

    @property
    def n_averages(self) -> int:
        """Number of periods averaged so far."""
        return self._count

    def feed(self, block: ArrayLike) -> int:
        """Add recorded samples ``(n,)`` or ``(n, n_channels)``.

        Returns the number of periods completed by this block (discarded ones included).

        Raises
        ------
        ValueError
            If the channel count does not match.
        """
        b = np.asarray(block, float)
        b2 = b[:, None] if b.ndim == 1 else b
        if b2.shape[1] != self.n_channels:
            raise ValueError("channel count does not match")
        self._buffer = np.concatenate((self._buffer, b2), axis=0)
        n = self.excitation.n_samples
        completed = 0
        while self._buffer.shape[0] >= n:
            p, self._buffer = self._buffer[:n], self._buffer[n:]
            completed += 1
            if self._discarded < self.n_discard:
                self._discarded += 1
                continue
            self._count += 1
            if self._average is None:
                self._average = p.copy()
            else:
                k = (
                    self._count
                    if self.averaging == "linear"
                    else min(self._count, self.n_exponential)
                )
                self._average = self._average + (p - self._average) / k
        return completed

    def average(self) -> NDArray[np.float64]:
        """Return the current average, shape ``(N, n_channels)``.

        Raises
        ------
        ValueError
            If no period has been averaged yet.
        """
        if self._average is None:
            raise ValueError("no period has been averaged yet")
        return self._average.copy()

    def result(self) -> Deconvolved:
        """Return the current impulse response (``periodic_deconvolve`` of the average).

        Raises
        ------
        ValueError
            If no period has been averaged yet.
        """
        return periodic_deconvolve(self.average(), self.excitation)

    def reset(self) -> None:
        """Clear buffered samples and the average (settling periods are discarded again)."""
        self._buffer = np.zeros((0, self.n_channels))
        self._average = None
        self._count = 0
        self._discarded = 0
