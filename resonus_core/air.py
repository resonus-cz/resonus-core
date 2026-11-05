"""Sound attenuation by atmospheric absorption (ISO 9613-1:1993) and speed of sound.

Implemented from ISO 9613-1:1993 clause 6 and Annex B; interface after
sapyne ``attenuation.py`` (https://github.com/vyhyb/sapyne, GPL-3.0, not copied; see
docs/DECISIONS.md D-A1). Temperatures are given in degrees Celsius at the interface and
converted to kelvin with ``T = t + 273.15``.

References
----------
ISO 9613-1:1993, Acoustics - Attenuation of sound during propagation outdoors - Part 1:
Calculation of the absorption of sound by the atmosphere.
ISO 354:2003, Acoustics - Measurement of sound absorption in a reverberation room, 8.1.2.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Reference ambient atmospheric pressure p_r in kPa: ISO 9613-1:1993, 4.2.
REFERENCE_PRESSURE = 101.325
# Reference air temperature T_0 in K (20 degC): ISO 9613-1:1993, 4.2.
REFERENCE_TEMPERATURE = 293.15
# Triple-point isotherm temperature T_01 in K: ISO 9613-1:1993, Annex B, Eq. (B.3).
TRIPLE_POINT_TEMPERATURE = 273.16
# Offset between degrees Celsius and kelvin (T_0 = 293.15 K at 20 degC, ISO 9613-1 4.2).
KELVIN_OFFSET = 273.15


def saturation_pressure_ratio(temperature: ArrayLike) -> NDArray[np.float64]:
    """Ratio ``p_sat / p_r`` of the saturation vapour pressure over liquid water.

    ``p_sat/p_r = 10**C`` with ``C = -6.8346 * (T_01/T)**1.261 + 4.6151``.

    Parameters
    ----------
    temperature
        Air temperature in degrees Celsius.

    Returns
    -------
    numpy.ndarray
        ``p_sat / p_r`` (dimensionless).

    Raises
    ------
    ValueError
        If any temperature is at or below absolute zero.

    References
    ----------
    ISO 9613-1:1993, Annex B, Eqs. (B.2) and (B.3).
    """
    t_k = np.asarray(temperature, dtype=float) + KELVIN_OFFSET
    if np.any(t_k <= 0):
        raise ValueError("Temperature must be above absolute zero.")
    c = -6.8346 * (TRIPLE_POINT_TEMPERATURE / t_k) ** 1.261 + 4.6151
    return 10.0**c


def molar_concentration(
    relative_humidity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike = REFERENCE_PRESSURE
) -> NDArray[np.float64]:
    """Molar concentration of water vapour ``h`` in percent.

    ``h = h_r * (p_sat/p_r) / (p_a/p_r)``.

    Parameters
    ----------
    relative_humidity
        Relative humidity ``h_r`` in percent, 0...100.
    temperature
        Air temperature in degrees Celsius.
    pressure
        Ambient atmospheric pressure ``p_a`` in kPa, > 0.

    Returns
    -------
    numpy.ndarray
        ``h`` in percent.

    Raises
    ------
    ValueError
        If the relative humidity is outside 0...100 or the pressure is <= 0.

    References
    ----------
    ISO 9613-1:1993, Annex B, Eq. (B.1).
    """
    hr = np.asarray(relative_humidity, dtype=float)
    pa = np.asarray(pressure, dtype=float)
    if np.any(hr < 0) or np.any(hr > 100):
        raise ValueError("Relative humidity must be in [0, 100] percent.")
    if np.any(pa <= 0):
        raise ValueError("Pressure must be positive.")
    return hr * saturation_pressure_ratio(temperature) / (pa / REFERENCE_PRESSURE)


def attenuation_coefficient(
    frequency: ArrayLike,
    temperature: float,
    relative_humidity: float,
    pressure: float = REFERENCE_PRESSURE,
) -> NDArray[np.float64]:
    """Pure-tone attenuation coefficient ``alpha`` for atmospheric absorption, in dB/m.

    With ``T`` in K, ``h`` from :func:`molar_concentration`, ``p_a`` in kPa:

    - ``f_rO = (p_a/p_r) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))``  (Eq. 3)
    - ``f_rN = (p_a/p_r) * (T/T_0)**(-1/2)
      * (9 + 280 * h * exp(-4.170 * ((T/T_0)**(-1/3) - 1)))``  (Eq. 4)
    - ``alpha = 8.686 * f**2 * (1.84e-11 * (p_a/p_r)**-1 * (T/T_0)**(1/2)
      + (T/T_0)**(-5/2) * (0.01275 * exp(-2239.1/T) / (f_rO + f**2/f_rO)
      + 0.1068 * exp(-3352.0/T) / (f_rN + f**2/f_rN)))``  (Eq. 5)

    Accuracy (clause 7): +-10 % for h 0.05...5 %, -20...+50 degC, p < 200 kPa and
    f/p 4e-4...10 Hz/Pa; outside these ranges the error may be larger (not checked here).

    Parameters
    ----------
    frequency
        Frequency in Hz (scalar or array), >= 0.
    temperature
        Air temperature in degrees Celsius.
    relative_humidity
        Relative humidity in percent.
    pressure
        Ambient atmospheric pressure in kPa.

    Returns
    -------
    numpy.ndarray
        ``alpha`` in dB/m, same shape as ``frequency``.

    Raises
    ------
    ValueError
        On negative frequencies or invalid atmospheric conditions.

    References
    ----------
    ISO 9613-1:1993, 6.2, Eqs. (3)-(5); Annex B.
    """
    f = np.asarray(frequency, dtype=float)
    if np.any(f < 0):
        raise ValueError("Frequency must be non-negative.")
    h = molar_concentration(relative_humidity, temperature, pressure)
    t_k = float(temperature) + KELVIN_OFFSET
    p = float(pressure) / REFERENCE_PRESSURE
    tr = t_k / REFERENCE_TEMPERATURE
    fr_o = p * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = p * tr ** (-0.5) * (9 + 280 * h * np.exp(-4.170 * (tr ** (-1 / 3) - 1)))
    return (
        8.686
        * f**2
        * (
            1.84e-11 / p * tr**0.5
            + tr ** (-2.5)
            * (
                0.01275 * np.exp(-2239.1 / t_k) / (fr_o + f**2 / fr_o)
                + 0.1068 * np.exp(-3352.0 / t_k) / (fr_n + f**2 / fr_n)
            )
        )
    )


def power_attenuation_coefficient(
    frequency: ArrayLike,
    temperature: float,
    relative_humidity: float,
    pressure: float = REFERENCE_PRESSURE,
) -> NDArray[np.float64]:
    """Power attenuation coefficient ``m = alpha / (10 lg e)`` in 1/m.

    ``alpha`` from :func:`attenuation_coefficient` (dB/m).

    References
    ----------
    ISO 354:2003, 8.1.2.1 (definition of ``m``); ISO 9613-1:1993, Eq. (5).
    """
    return attenuation_coefficient(frequency, temperature, relative_humidity, pressure) / (
        10 * np.log10(np.e)
    )


def speed_of_sound(temperature: ArrayLike) -> NDArray[np.float64]:
    """Speed of sound ``c = 331 + 0.6 t`` in m/s, ``t`` in degrees Celsius.

    The formula is given for 15...30 degC; outside that range a ValueError is raised.

    Raises
    ------
    ValueError
        If any temperature is outside 15...30 degC.

    References
    ----------
    ISO 354:2003, 8.1.2.1, NOTE, Eq. (6).
    """
    t = np.asarray(temperature, dtype=float)
    if np.any(t < 15) or np.any(t > 30):
        raise ValueError("Temperature must be in [15, 30] °C.")
    return 331.0 + 0.6 * t
