import numpy as np
import pytest
from scipy.signal import sosfilt, sosfreqz

from resonus_core.bands import (
    BandSet,
    band_edges,
    band_filter,
    band_index,
    band_set,
    exact_center,
    filter_bands,
    frequency_label,
    nominal_frequency,
)

# ISO 354:2003 clause 5: one-third-octave centre frequencies of the measurement.
ISO354_THIRDS = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                 2000, 2500, 3150, 4000, 5000]  # fmt: skip
# ISO 9613-1:1993 Table 1: preferred frequencies (one-third octaves 50 Hz-10 kHz).
ISO9613_THIRDS = [50, 63, 80] + ISO354_THIRDS + [6300, 8000, 10000]


def test_exact_center_third_octave_iso9613_eq6():
    # ISO 9613-1:1993 Eq. (6) with b = 1/3: f_m = 1000 * 10**(k/10).
    k = np.arange(-13, 11)
    np.testing.assert_allclose(exact_center(k, 3), 1000 * 10 ** (k / 10), rtol=1e-12)


def test_exact_center_octave_and_even_fraction():
    np.testing.assert_allclose(exact_center([-3, 0, 3], 1), [10**2.1, 1000, 10**3.9], rtol=1e-12)
    # Even b: f_m = f_r * G**((2x+1)/(2b)), G = 10**0.3.
    np.testing.assert_allclose(exact_center(0, 2), 1000 * 10 ** (0.3 * 1 / 4), rtol=1e-12)
    np.testing.assert_allclose(exact_center(-1, 2), 1000 * 10 ** (-0.3 * 1 / 4), rtol=1e-12)
    assert exact_center(0, 3).shape == ()
    with pytest.raises(ValueError):
        exact_center(0, 0)


def test_band_edges():
    lo, hi = band_edges([1000.0, 125.0], 1)
    np.testing.assert_allclose(lo, [1000 * 10**-0.15, 125 * 10**-0.15], rtol=1e-12)
    np.testing.assert_allclose(hi, [1000 * 10**0.15, 125 * 10**0.15], rtol=1e-12)
    lo, hi = band_edges(1000.0, 3)
    np.testing.assert_allclose([lo, hi], [1000 * 10**-0.05, 1000 * 10**0.05], rtol=1e-12)
    # Adjacent bands share their edges.
    c = exact_center(np.arange(-5, 5), 3)
    lo, hi = band_edges(c, 3)
    np.testing.assert_allclose(hi[:-1], lo[1:], rtol=1e-12)
    with pytest.raises(ValueError):
        band_edges(0.0, 3)
    with pytest.raises(ValueError):
        band_edges(1000.0, 0)


def test_band_index_inverts_exact_center():
    for b in (1, 2, 3, 6):
        x = np.arange(-12, 10)
        np.testing.assert_array_equal(band_index(exact_center(x, b), b), x)
    assert band_index(100.0, 3) == -10 and band_index(125.0, 1) == -3
    assert band_index(90.0, 3) == -10  # nearest band
    with pytest.raises(ValueError):
        band_index(0.0, 3)


def test_nominal_frequencies_match_iso_lists():
    x = band_index(np.array(ISO9613_THIRDS, float), 3)
    np.testing.assert_array_equal(nominal_frequency(x, 3), ISO9613_THIRDS)
    np.testing.assert_array_equal(
        nominal_frequency(np.arange(-5, 5), 1),
        [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    )
    np.testing.assert_array_equal(nominal_frequency([-16, 13], 3), [25, 20000])
    with pytest.raises(ValueError):
        nominal_frequency(14, 3)
    with pytest.raises(ValueError):
        nominal_frequency(-6, 1)
    with pytest.raises(ValueError):
        nominal_frequency(0, 2)


def test_frequency_label():
    assert [frequency_label(f) for f in (31.5, 500.0, 1000.0, 1250.0, 12500.0)] == [
        "31.5", "500", "1k", "1.25k", "12.5k"]  # fmt: skip


def test_band_set_iso354():
    bs = band_set(3, 100, 5000)
    assert isinstance(bs, BandSet) and len(bs) == 18
    np.testing.assert_array_equal(bs.nominal, ISO354_THIRDS)
    assert bs.indices == tuple(range(-10, 8))
    np.testing.assert_allclose(bs.center, 1000 * 10 ** (np.arange(-10, 8) / 10), rtol=1e-12)
    np.testing.assert_allclose(bs.upper / bs.lower, 10**0.1, rtol=1e-12)
    assert bs.labels[0] == "100" and bs.labels[-1] == "5k"


def test_band_set_octaves_and_errors():
    bs = band_set(1, 125, 4000)
    np.testing.assert_array_equal(bs.nominal, [125, 250, 500, 1000, 2000, 4000])
    with pytest.raises(ValueError):
        band_set(2, 100, 1000)
    with pytest.raises(ValueError):
        band_set(3, 1000, 100)
    with pytest.raises(ValueError):
        band_set(3, 0, 100)
    with pytest.raises(ValueError):
        band_set(3, 10, 100)  # 10 Hz is outside the nominal table
    with pytest.raises(ValueError):
        BandSet(3, (0, 2))
    with pytest.raises(ValueError):
        BandSet(3, ())
    with pytest.raises(ValueError):
        BandSet(2, (0,))


@pytest.mark.parametrize(("fc", "b"), [(125.0, 1), (1000.0, 1), (100.0, 3), (4000.0, 3)])
def test_band_filter_edges_minus_3db(fc, b):
    fs = 48000.0
    fm = float(exact_center(band_index(fc, b), b))
    lo, hi = band_edges(fm, b)
    sos = band_filter(fm, b, fs)
    _, h = sosfreqz(sos, worN=[float(lo), fm, float(hi)], fs=fs)
    db = 20 * np.log10(np.abs(h))
    np.testing.assert_allclose(db[[0, 2]], -10 * np.log10(2), atol=1e-6)
    assert abs(db[1]) < 0.05


def test_band_filter_order_and_nyquist():
    sos = band_filter(1000.0, 1, 48000.0, order=4)
    assert sos.shape == (4, 6)  # 8 poles
    assert band_filter(1000.0, 1, 48000.0).shape == (3, 6)  # default order 3
    with pytest.raises(ValueError):
        band_filter(20000.0, 3, 44100.0)  # upper edge 22.4 kHz > 22.05 kHz


def test_filter_bands_forward_and_reverse():
    fs = 48000.0
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4800)
    bs = band_set(1, 250, 1000)
    y = filter_bands(x, bs, fs)
    assert y.shape == (4800, 3)
    for k, fm in enumerate(bs.center):
        np.testing.assert_allclose(y[:, k], sosfilt(band_filter(float(fm), 1, fs), x))
    yr = filter_bands(x, bs, fs, order=2, reverse=True)
    for k, fm in enumerate(bs.center):
        ref = sosfilt(band_filter(float(fm), 1, fs, order=2), x[::-1])[::-1]
        np.testing.assert_allclose(yr[:, k], ref)
    with pytest.raises(ValueError):
        filter_bands(np.zeros((10, 2)), bs, fs)
