import numpy as np
import pytest

from resonus_core.air import (
    attenuation_coefficient,
    molar_concentration,
    power_attenuation_coefficient,
    saturation_pressure_ratio,
    speed_of_sound,
)

RH = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# ISO 9613-1:1993 Table 1 (i) and (j), dB/km at 101.325 kPa, for the exact one-third-octave
# mid-band frequencies f = 1000 * 10**(k/10) (6.4 NOTE 5). Key: (temperature, k).
TABLE1 = {
    (20, -13): [0.270, 0.214, 0.174, 0.125, 0.0965, 0.0784, 0.0660, 0.0570, 0.0501, 0.0447, 0.0403],
    (20, -10): [0.622, 0.579, 0.529, 0.429, 0.351, 0.294, 0.252, 0.220, 0.194, 0.174, 0.158],
    (20, 0): [14.1, 8.72, 6.53, 5.01, 4.65, 4.66, 4.80, 4.98, 5.15, 5.30, 5.42],
    (20, 6): [109, 95.1, 74.1, 48.5, 36.1, 29.4, 25.4, 22.9, 21.3, 20.2, 19.4],
    (20, 10): [193, 267, 284, 242, 194, 159, 135, 118, 105, 95.3, 87.9],
    (25, -13): [0.262, 0.197, 0.156, 0.109, 0.0830, 0.0671, 0.0563, 0.0485, 0.0426, 0.0379, 0.0342],
    (25, -10): [0.681, 0.604, 0.523, 0.397, 0.314, 0.258, 0.219, 0.190, 0.167, 0.149, 0.135],
    (25, 0): [10.7, 7.13, 5.87, 5.27, 5.39, 5.68, 5.96, 6.19, 6.35, 6.44, 6.47],
    (25, 6): [110, 79.4, 58.8, 38.8, 30.1, 25.7, 23.2, 21.9, 21.1, 20.8, 20.6],
    (25, 10): [274, 308, 279, 207, 160, 131, 112, 98.9, 89.6, 82.8, 77.6],
}  # fmt: skip


@pytest.mark.parametrize(("temp", "k"), sorted(TABLE1))
def test_attenuation_matches_iso9613_table1(temp, k):
    f = 1000 * 10 ** (k / 10)
    got = [1000 * float(attenuation_coefficient(f, temp, rh)) for rh in RH]
    # Table values have 3 significant digits: half a unit in the last digit is <= 0.5 %.
    np.testing.assert_allclose(got, TABLE1[(temp, k)], rtol=0.0051)


def test_vectorized_over_frequency():
    f = 1000 * 10 ** (np.array([-13, -10, 0, 6, 10]) / 10)
    got = 1000 * attenuation_coefficient(f, 20.0, 50.0)
    assert got.shape == (5,)
    np.testing.assert_allclose(got, [0.0784, 0.294, 4.66, 29.4, 159], rtol=0.0051)
    assert attenuation_coefficient(0.0, 20.0, 50.0) == 0.0


def test_power_attenuation_coefficient():
    f = np.array([125.0, 1000.0, 4000.0])
    alpha = attenuation_coefficient(f, 22.0, 45.0, 98.0)
    m = power_attenuation_coefficient(f, 22.0, 45.0, 98.0)
    # ISO 354:2003 8.1.2.1: m = alpha / (10 lg e).
    np.testing.assert_allclose(m, alpha / (10 * np.log10(np.e)), rtol=1e-12)


def test_humidity_conversion_annex_b():
    # Eq. (B.2)/(B.3) at the triple point: C = -6.8346 + 4.6151.
    np.testing.assert_allclose(saturation_pressure_ratio(0.01), 10 ** (-6.8346 + 4.6151))
    r20 = saturation_pressure_ratio(20.0)
    np.testing.assert_allclose(molar_concentration(50.0, 20.0), 50 * r20)
    np.testing.assert_allclose(molar_concentration(50.0, 20.0, 101.325 / 2), 100 * r20)
    np.testing.assert_allclose(molar_concentration([0, 100], 20.0), [0, 100 * r20])


def test_pressure_scaling_identity():
    # From Eqs. (3)-(5): at a fixed molar concentration h, f_rO and f_rN scale with
    # q = p_a/p_r, hence alpha(f, q*p_r) = q * alpha(f/q, p_r). h fixed: h_r scales with q (B.1).
    q = 0.5
    for f in (125.0, 2000.0, 8000.0):
        a = attenuation_coefficient(f, 20.0, 50.0 * q, 101.325 * q)
        np.testing.assert_allclose(a, q * attenuation_coefficient(f / q, 20.0, 50.0), rtol=1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        attenuation_coefficient(-1.0, 20.0, 50.0)
    with pytest.raises(ValueError):
        attenuation_coefficient(1000.0, 20.0, 101.0)
    with pytest.raises(ValueError):
        attenuation_coefficient(1000.0, 20.0, -1.0)
    with pytest.raises(ValueError):
        attenuation_coefficient(1000.0, 20.0, 50.0, 0.0)
    with pytest.raises(ValueError):
        saturation_pressure_ratio(-273.15)


def test_speed_of_sound_iso354_eq6():
    np.testing.assert_allclose(speed_of_sound([15.0, 20.0, 30.0]), [340.0, 343.0, 349.0])
    with pytest.raises(ValueError):
        speed_of_sound(14.9)
    with pytest.raises(ValueError):
        speed_of_sound(30.1)
