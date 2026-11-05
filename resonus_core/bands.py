"""Fractional-octave frequency bands (base-10 system) and Butterworth band filters.

PROVISIONAL (docs/DECISIONS.md D-B1): IEC 61260-1:2014 was not
available. Band formulas are ported from python-acoustics
(``acoustics.standards.iec_61260_1_2014``, BSD-3-Clause, see THIRD_PARTY.md) and agree
with ISO 9613-1:1993 Eq. (6) for one-third-octave bands. **Compliance of the filters
with IEC 61260-1 (class 1 or 2) has not been verified.**

Band index ``x`` counts bands from the reference frequency 1000 Hz (``x = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from resonus_core.spectral import band_sos, filter_forward

# Octave frequency ratio G of the base-10 system: python-acoustics iec_61260_1_2014
# (Eq. 1 of IEC 61260-1:2014, not verified); consistent with ISO 9613-1:1993 Eq. (6)
# (one-third octaves: f_m = 1000 * 10**(k/10)).
OCTAVE_RATIO = 10.0 ** (3.0 / 10.0)
# Reference frequency f_r: ISO 9613-1:1993 6.4 NOTE 5 ("1 000 Hz is the exact reference
# frequency"); python-acoustics REFERENCE_FREQUENCY.
REFERENCE_FREQUENCY = 1000.0
# Nominal mid-band frequencies (labels), python-acoustics NOMINAL_*_CENTER_FREQUENCIES;
# the one-third-octave values 50 Hz-10 kHz agree with ISO 9613-1:1993 Table 1 and
# 100 Hz-5 kHz with ISO 354:2003 clause 5.
NOMINAL_OCTAVE = (31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)
NOMINAL_THIRD_OCTAVE = (
    25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
)  # fmt: skip
# Band index of the first entry of each nominal table (31.5 Hz octave, 25 Hz third octave).
_FIRST_INDEX = {1: -5, 3: -16}
_NOMINAL = {1: NOMINAL_OCTAVE, 3: NOMINAL_THIRD_OCTAVE}
# PROVISIONAL (D-B1, IEC 61260-1 missing): Butterworth design order passed to
# scipy.signal.butter; a band-pass of order 3 has 6 poles.
DEFAULT_FILTER_ORDER = 3


def exact_center(index: ArrayLike, fraction: int) -> NDArray[np.float64]:
    """Exact mid-band frequencies of the base-10 system.

    ``f_m = f_r * G**(x/b)`` for odd ``b``, ``f_m = f_r * G**((2x+1)/(2b))`` for even ``b``
    (python-acoustics ``exact_center_frequency``; IEC 61260-1:2014 Eqs. (2)-(3), not
    verified), with ``G = OCTAVE_RATIO``, ``f_r = REFERENCE_FREQUENCY``.

    Parameters
    ----------
    index
        Band index ``x`` (integer, scalar or array).
    fraction
        Bandwidth designator ``b`` (1 = octave, 3 = one-third octave), >= 1.

    Returns
    -------
    numpy.ndarray
        Exact mid-band frequencies in Hz, same shape as ``index``.

    Raises
    ------
    ValueError
        If ``fraction < 1``.

    References
    ----------
    ISO 9613-1:1993, 6.4, Eq. (6) (one-third octaves: ``f_m = 1000 * 10**(k/10)``).
    """
    if fraction < 1:
        raise ValueError("fraction must be >= 1")
    x = np.asarray(index, dtype=float)
    if fraction % 2:
        return REFERENCE_FREQUENCY * OCTAVE_RATIO ** (x / fraction)
    else:
        return REFERENCE_FREQUENCY * OCTAVE_RATIO ** ((2 * x + 1) / (2 * fraction))


def band_edges(center: ArrayLike, fraction: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lower and upper band-edge frequencies ``f_m * G**(-1/(2b))``, ``f_m * G**(1/(2b))``.

    Parameters
    ----------
    center
        Exact mid-band frequencies in Hz, > 0.
    fraction
        Bandwidth designator ``b``, >= 1.

    Returns
    -------
    lower, upper
        Edge frequencies in Hz, same shape as ``center``.

    Raises
    ------
    ValueError
        If ``fraction < 1`` or any centre frequency is <= 0.
    """
    if fraction < 1:
        raise ValueError("fraction must be >= 1")
    fm = np.asarray(center, dtype=float)
    if np.any(fm <= 0):
        raise ValueError("center frequencies must be > 0")
    k = OCTAVE_RATIO ** (1.0 / (2 * fraction))
    return fm / k, fm * k


def band_index(frequency: ArrayLike, fraction: int) -> NDArray[np.int64]:
    """Index of the band whose exact mid-band frequency is nearest to ``frequency``.

    Odd ``b``: ``round(b * ln(f/f_r) / ln(G))``; even ``b``:
    ``round((2b * ln(f/f_r)/ln(G) - 1) / 2)`` (inverse of :func:`exact_center`).

    Raises
    ------
    ValueError
        If ``fraction < 1`` or any frequency is <= 0.
    """
    if fraction < 1:
        raise ValueError("fraction must be >= 1")
    f = np.asarray(frequency, dtype=float)
    if np.any(f <= 0):
        raise ValueError("frequencies must be > 0")
    r = np.log(f / REFERENCE_FREQUENCY) / np.log(OCTAVE_RATIO)
    if fraction % 2:
        return np.round(fraction * r).astype(np.int64)
    else:
        return np.round((2 * fraction * r - 1) / 2).astype(np.int64)


def nominal_frequency(index: ArrayLike, fraction: int) -> NDArray[np.float64]:
    """Nominal mid-band frequency (label) of octave and one-third-octave bands.

    Looked up in ``NOMINAL_OCTAVE`` (``fraction == 1``, indices -5...4) or
    ``NOMINAL_THIRD_OCTAVE`` (``fraction == 3``, indices -16...13).

    Raises
    ------
    ValueError
        If ``fraction`` is not 1 or 3, or an index is outside the table.
    """
    if fraction not in _NOMINAL:
        raise ValueError("fraction must be 1 or 3")
    table = np.asarray(_NOMINAL[fraction])
    pos = np.asarray(index, dtype=np.int64) - _FIRST_INDEX[fraction]
    if np.any(pos < 0) or np.any(pos >= table.size):
        raise ValueError("index out of nominal table range")
    return table[pos]


def frequency_label(nominal: float) -> str:
    """Short label of a nominal frequency: ``31.5``, ``500``, ``1k``, ``1.25k``, ``12.5k``."""
    if nominal >= 1000:
        return f"{nominal / 1000:g}k"
    else:
        return f"{nominal:g}"


@dataclass(frozen=True)
class BandSet:
    """Consecutive fractional-octave bands.

    Parameters
    ----------
    fraction
        Bandwidth designator ``b``: 1 (octave) or 3 (one-third octave).
    indices
        Band indices ``x``, consecutive and increasing.
    """

    fraction: int
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate fraction and indices (via :func:`nominal_frequency`)."""
        if self.fraction not in (1, 3):
            raise ValueError("fraction must be 1 or 3")
        if len(self.indices) == 0:
            raise ValueError("indices cannot be empty")
        if np.any(np.diff(self.indices) != 1):
            raise ValueError("indices must be consecutive")
        # Validate indices are within nominal tables
        nominal_frequency(self.indices, self.fraction)

    def __len__(self) -> int:
        """Return the number of bands."""
        return len(self.indices)

    @property
    def center(self) -> NDArray[np.float64]:
        """Exact mid-band frequencies in Hz."""
        return exact_center(self.indices, self.fraction)

    @property
    def lower(self) -> NDArray[np.float64]:
        """Lower band-edge frequencies in Hz."""
        return band_edges(self.center, self.fraction)[0]

    @property
    def upper(self) -> NDArray[np.float64]:
        """Upper band-edge frequencies in Hz."""
        return band_edges(self.center, self.fraction)[1]

    @property
    def nominal(self) -> NDArray[np.float64]:
        """Nominal mid-band frequencies in Hz (labels)."""
        return nominal_frequency(self.indices, self.fraction)

    @property
    def labels(self) -> list[str]:
        """Labels from :func:`frequency_label`."""
        return [frequency_label(float(f)) for f in self.nominal]


def band_set(fraction: int, f_min: float, f_max: float) -> BandSet:
    """Bands from the one nearest ``f_min`` to the one nearest ``f_max`` (inclusive).

    Example: ``band_set(3, 100, 5000)`` gives the 18 ISO 354 one-third-octave bands.

    Raises
    ------
    ValueError
        If ``fraction`` is not 1 or 3, ``f_min <= 0``, ``f_min > f_max``, or a band lies
        outside the nominal tables.
    """
    if fraction not in (1, 3):
        raise ValueError("fraction must be 1 or 3")
    if f_min <= 0 or f_min > f_max:
        raise ValueError("f_min must be > 0 and <= f_max")
    lo = int(band_index(f_min, fraction))
    hi = int(band_index(f_max, fraction))
    return BandSet(fraction, tuple(range(lo, hi + 1)))


def band_filter(
    center: float, fraction: int, sample_rate: float, order: int = DEFAULT_FILTER_ORDER
) -> NDArray[np.float64]:
    """Butterworth band-pass for one band as second-order sections.

    ``resonus_core.spectral.band_sos(lower, upper, sample_rate, order)`` with the edges of
    :func:`band_edges`: -3 dB at the band edges, single pass. PROVISIONAL (D-B1):
    compliance with IEC 61260-1 has not been verified.

    Raises
    ------
    ValueError
        If the upper band edge is not below ``sample_rate / 2`` or ``order < 1``.
    """
    lower, upper = band_edges(center, fraction)
    if upper >= sample_rate / 2:
        raise ValueError("upper band edge must be below Nyquist")
    if order < 1:
        raise ValueError("order must be >= 1")
    return band_sos(float(lower), float(upper), sample_rate, order)


def filter_bands(
    x: ArrayLike,
    bands: BandSet,
    sample_rate: float,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
) -> NDArray[np.float64]:
    """Filter a 1-D signal into all bands of ``bands``.

    Forward (causal) filtering with :func:`band_filter` per band. ``reverse=True``
    filters the time-reversed signal and reverses the result (time-reversal technique,
    ISO 3382-2:2008 7.3 NOTE: lower limit ``BT > 4`` instead of ``BT > 16``).

    Parameters
    ----------
    x
        Signal, 1-D, length ``n``.
    bands
        Band set.
    sample_rate
        Sample rate in Hz.
    order
        Butterworth order (see :func:`band_filter`).
    reverse
        Time-reversed filtering.

    Returns
    -------
    numpy.ndarray
        ``(n, len(bands))`` band signals.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D or a band is not below half the sample rate.
    """
    s = np.asarray(x, dtype=float)
    if s.ndim != 1:
        raise ValueError("x must be 1-D")
    if reverse:
        s = s[::-1]
    out = np.empty((s.size, len(bands)), dtype=float)
    for k, fm in enumerate(bands.center):
        out[:, k] = filter_forward(
            s,
            band_filter(float(fm), bands.fraction, sample_rate, order),
        )
    return out[::-1] if reverse else out
