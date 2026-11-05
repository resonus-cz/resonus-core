import numpy as np
import pytest
from scipy.signal import sosfreqz

from resonus_core.signals import exp_sweep
from resonus_core.spectral import (
    band_sos,
    band_window,
    filter_forward,
    filter_zero_phase,
    preequalize,
)

FS = 48000


# ------------------------------------------------------------------ band window
def test_band_window_values():
    t = 1.0
    f = np.array(
        [0.0, 50.0, 100.0 * 2**-0.5, 100.0, 1000.0, 2000.0, 2000.0 * 2**0.5, 4000.0, 8000.0]
    )
    w = band_window(f, (100.0, 2000.0), taper_octaves=t)
    np.testing.assert_allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-12)


def test_band_window_monotonic_tapers_and_edges():
    f = np.linspace(0, 24000, 4801)
    w = band_window(f, (500.0, 5000.0), taper_octaves=1 / 3)
    assert np.all((w >= 0) & (w <= 1))
    inside = (f >= 500) & (f <= 5000)
    assert np.all(w[inside] == 1.0)
    low = (f > 500 * 2 ** (-1 / 3)) & (f < 500)
    assert np.all(np.diff(w[low]) >= 0)
    high = (f > 5000) & (f < 5000 * 2 ** (1 / 3))
    assert np.all(np.diff(w[high]) <= 0)
    assert np.all(w[f >= 5000 * 2 ** (1 / 3)] == 0) and np.all(w[f <= 500 * 2 ** (-1 / 3)] == 0)


def test_band_window_zero_low_edge_and_hard_edges():
    f = np.array([0.0, 10.0, 999.0, 1000.0, 1001.0])
    np.testing.assert_array_equal(band_window(f, (0.0, 1000.0), 0.0), [1, 1, 1, 1, 0])
    with pytest.raises(ValueError):
        band_window(f, (100.0, 50.0))
    with pytest.raises(ValueError):
        band_window(f, (10.0, 50.0), -1.0)


# ------------------------------------------------------------------ IIR band filters
def test_band_sos_edges_minus_3_db():
    sos = band_sos(500.0, 2000.0, FS, order=4)
    _, h = sosfreqz(sos, worN=[500.0, 1000.0, 2000.0], fs=FS)
    np.testing.assert_allclose(20 * np.log10(np.abs(h[[0, 2]])), -3.01, atol=0.01)
    assert 20 * np.log10(abs(h[1])) == pytest.approx(0.0, abs=0.1)


def test_band_sos_lowpass_and_highpass():
    _, h = sosfreqz(band_sos(0.0, 1000.0, FS), worN=[10.0, 1000.0], fs=FS)
    assert abs(h[0]) == pytest.approx(1.0, abs=1e-3) and 20 * np.log10(abs(h[1])) == pytest.approx(
        -3.01, abs=0.01
    )
    _, h = sosfreqz(band_sos(1000.0, FS / 2, FS), worN=[1000.0, 20000.0], fs=FS)
    assert 20 * np.log10(abs(h[0])) == pytest.approx(-3.01, abs=0.01) and abs(
        h[1]
    ) == pytest.approx(1.0, abs=1e-3)
    for args in ((0.0, FS / 2), (100.0, 50.0), (-1.0, 100.0)):
        with pytest.raises(ValueError):
            band_sos(*args, FS)
    with pytest.raises(ValueError):
        band_sos(100.0, 1000.0, FS, order=0)


def test_zero_phase_squares_magnitude_and_is_symmetric():
    sos = band_sos(500.0, 2000.0, FS)
    x = np.zeros(2**14)
    x[2**13] = 1.0
    y = filter_zero_phase(x, sos)
    np.testing.assert_allclose(
        y[2**13 - 500 : 2**13], y[2**13 + 500 : 2**13 : -1], atol=1e-9
    )  # symmetric
    spec = np.abs(np.fft.rfft(np.roll(y, -(2**13))))
    f = np.fft.rfftfreq(2**14, 1 / FS)
    _, h = sosfreqz(sos, worN=f, fs=FS)
    np.testing.assert_allclose(spec[f > 100], np.abs(h[f > 100]) ** 2, atol=1e-3)


def test_forward_filter_matches_sosfilt_and_multichannel():
    sos = band_sos(500.0, 2000.0, FS)
    x = np.random.default_rng(0).standard_normal((1000, 2))
    y = filter_forward(x, sos)
    assert y.shape == (1000, 2)
    np.testing.assert_allclose(y[:, 0], filter_forward(x[:, 0], sos))
    assert filter_zero_phase(x, sos).shape == (1000, 2)


# ------------------------------------------------------------------ pre-equalization
def tilt_response(f):
    # smooth measured response: first-order low-pass at 1 kHz
    return 1 / (1 + 1j * f / 1000.0)


def test_preequalized_excitation_flattens_the_system():
    e = exp_sweep(50.0, 10000.0, 1.0, FS, peak_dbfs=-6.0, fade_in=0.05, fade_out=0.005)
    fm = np.linspace(0, FS / 2, 2049)
    pe = preequalize(e, fm, tilt_response(fm), band=(100.0, 8000.0), eps=1e-6)
    assert pe.kind == "preequalized" and pe.params["source_kind"] == "exp_sweep"
    assert pe.n_samples == e.n_samples
    assert np.max(np.abs(pe.signal)) == pytest.approx(np.max(np.abs(e.signal)))
    f = np.fft.rfftfreq(e.n_samples, 1 / FS)
    through = np.abs(np.fft.rfft(pe.signal)) * np.abs(tilt_response(f))
    original = np.abs(np.fft.rfft(e.signal))
    band = (f >= 200) & (f <= 5000)
    ratio_db = 20 * np.log10(through[band] / original[band])
    assert np.ptp(ratio_db) < 0.5  # flat in band: the system's tilt is compensated


def test_preequalization_is_regularized_and_band_limited():
    e = exp_sweep(50.0, 10000.0, 1.0, FS)
    fm = np.linspace(0, FS / 2, 2049)
    notch = np.ones(fm.size)
    notch[(fm > 990) & (fm < 1010)] = 1e-6  # deep notch
    pe = preequalize(e, fm, notch, band=(100.0, 8000.0), eps=1e-2)
    f = np.fft.rfftfreq(e.n_samples, 1 / FS)
    gain = np.abs(np.fft.rfft(pe.signal)) / np.abs(np.fft.rfft(e.signal))
    band = (f >= 200) & (f <= 5000)
    # boost at the notch is limited (no 120 dB spike): within ~+20 dB of the in-band median
    assert 20 * np.log10(np.max(gain[band]) / np.median(gain[band])) < 20.0
    assert np.max(gain[f > 8000 * 2 ** (1 / 3)]) < 1e-9  # removed above the band (+ taper)


def test_preequalize_validation():
    e = exp_sweep(50.0, 10000.0, 0.2, FS)
    fm = np.linspace(0, FS / 2, 10)
    with pytest.raises(ValueError):
        preequalize(e, fm, np.ones(9), (100.0, 8000.0))
    with pytest.raises(ValueError):
        preequalize(e, fm, np.ones(10), (100.0, 8000.0), eps=0.0)
    with pytest.raises(ValueError):
        preequalize(e, fm, np.ones(10), (8000.0, 100.0))
