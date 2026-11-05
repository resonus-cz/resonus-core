import numpy as np
import pytest
from scipy.signal import chirp, max_len_seq

from resonus_core.signals import Excitation, exp_sweep, linear_sweep, mls, pink_noise, white_noise

FS = 48000


def band_power(x, fs, f_lo, f_hi):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1 / fs)
    return spec[(f >= f_lo) & (f < f_hi)].sum()


# ------------------------------------------------------------------ sweeps
def test_exp_sweep_matches_farina_formula():
    e = exp_sweep(20.0, 20000.0, 1.0, FS, peak_dbfs=0.0)
    assert isinstance(e, Excitation) and e.kind == "exp_sweep"
    assert e.n_samples == FS and e.duration == pytest.approx(1.0)
    t = np.arange(FS) / FS
    L = 1.0 / np.log(1000.0)
    expected = np.sin(2 * np.pi * 20.0 * L * (np.exp(t / L) - 1))
    np.testing.assert_allclose(e.signal, expected, atol=1e-6)
    assert e.params["rate"] == pytest.approx(L)
    assert e.params["f_start"] == 20.0 and e.params["f_stop"] == 20000.0


def test_exp_sweep_equals_scipy_chirp():
    e = exp_sweep(50.0, 10000.0, 0.5, FS, peak_dbfs=0.0)
    t = np.arange(e.n_samples) / FS
    np.testing.assert_allclose(
        e.signal, chirp(t, 50.0, 0.5, 10000.0, method="logarithmic", phi=-90), atol=1e-9
    )


def test_peak_level_and_fades():
    e = exp_sweep(20.0, 20000.0, 1.0, FS, peak_dbfs=-6.0, fade_in=0.1, fade_out=0.01)
    a = 10 ** (-6 / 20)
    assert np.max(np.abs(e.signal)) == pytest.approx(a, rel=1e-3)
    m, k = 4800, 480
    plain = exp_sweep(20.0, 20000.0, 1.0, FS, peak_dbfs=-6.0)
    np.testing.assert_allclose(
        e.signal[:m], plain.signal[:m] * (1 - np.cos(np.pi * np.arange(m) / m)) / 2
    )
    np.testing.assert_allclose(
        e.signal[-k:], plain.signal[-k:] * (1 + np.cos(np.pi * np.arange(k) / k)) / 2
    )
    np.testing.assert_allclose(e.signal[m:-k], plain.signal[m:-k])
    assert e.signal[0] == 0.0


def test_linear_sweep():
    e = linear_sweep(0.0, 24000.0, 0.25, FS, peak_dbfs=0.0)
    assert e.kind == "linear_sweep" and e.n_samples == 12000
    t = np.arange(12000) / FS
    np.testing.assert_allclose(
        e.signal, chirp(t, 0.0, 0.25, 24000.0, method="linear", phi=-90), atol=1e-9
    )
    # energy is spread evenly over frequency (white): compare two equal bands
    p1 = band_power(e.signal, FS, 2000, 6000)
    p2 = band_power(e.signal, FS, 12000, 16000)
    assert 10 * np.log10(p1 / p2) == pytest.approx(0.0, abs=0.5)


def test_exp_sweep_is_pink():
    # exponential sweep: equal energy per octave
    e = exp_sweep(20.0, 20000.0, 2.0, FS)
    p1 = band_power(e.signal, FS, 250, 500)
    p2 = band_power(e.signal, FS, 2000, 4000)
    assert 10 * np.log10(p1 / p2) == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"f_start": 0.0},
        {"f_start": 100.0, "f_stop": 50.0},
        {"f_stop": 30000.0},
        {"duration": 0.0},
        {"peak_dbfs": 1.0},
        {"fade_in": -0.1},
        {"fade_in": 0.6, "fade_out": 0.6},
    ],
)
def test_sweep_validation(kwargs):
    args = {"f_start": 20.0, "f_stop": 20000.0, "duration": 1.0, "sample_rate": FS} | kwargs
    with pytest.raises(ValueError):
        exp_sweep(**args)


def test_linear_sweep_validation():
    with pytest.raises(ValueError):
        linear_sweep(-1.0, 1000.0, 1.0, FS)
    with pytest.raises(ValueError):
        linear_sweep(0.0, 1000.0, 1.0, 0)


# ------------------------------------------------------------------ noise
@pytest.mark.parametrize("gen", [white_noise, pink_noise])
@pytest.mark.parametrize("periodic", [False, True])
def test_noise_level_and_reproducibility(gen, periodic):
    e = gen(2**15, FS, rms_db=-20.0, periodic=periodic, seed=3)
    assert e.n_samples == 2**15 and e.signal.dtype == np.float64
    assert 20 * np.log10(np.sqrt(np.mean(e.signal**2))) == pytest.approx(-20.0, abs=1e-9)
    np.testing.assert_array_equal(
        e.signal, gen(2**15, FS, rms_db=-20.0, periodic=periodic, seed=3).signal
    )
    assert not np.array_equal(
        e.signal, gen(2**15, FS, rms_db=-20.0, periodic=periodic, seed=4).signal
    )
    assert e.params["periodic"] is periodic and e.params["seed"] == 3


def test_periodic_white_has_exactly_flat_spectrum():
    x = white_noise(4096, FS, periodic=True).signal
    mag = np.abs(np.fft.rfft(x))
    assert mag[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(mag[1:], mag[1], rtol=1e-9)


def test_periodic_pink_has_exact_slope():
    x = pink_noise(4096, FS, periodic=True).signal
    mag = np.abs(np.fft.rfft(x))
    k = np.arange(1, mag.size)
    np.testing.assert_allclose(mag[1:] * np.sqrt(k), mag[1] * 1.0, rtol=1e-9)
    assert mag[0] == pytest.approx(0.0, abs=1e-9)


def test_random_pink_equal_power_per_octave():
    x = pink_noise(2**18, FS, seed=1).signal
    p1 = band_power(x, FS, 250, 500)
    p2 = band_power(x, FS, 4000, 8000)
    assert 10 * np.log10(p1 / p2) == pytest.approx(0.0, abs=0.5)


def test_random_white_is_flat_per_hz():
    x = white_noise(2**18, FS, seed=1).signal
    p1 = band_power(x, FS, 1000, 3000)
    p2 = band_power(x, FS, 15000, 17000)
    assert 10 * np.log10(p1 / p2) == pytest.approx(0.0, abs=0.3)


def test_noise_clipping_rejected():
    with pytest.raises(ValueError, match="clip"):
        white_noise(2**15, FS, rms_db=0.0)


def test_noise_validation():
    with pytest.raises(ValueError):
        white_noise(1, FS)
    with pytest.raises(ValueError):
        pink_noise(1024, 0)


# ------------------------------------------------------------------ MLS
def test_mls():
    e = mls(10, FS, peak_dbfs=-6.0)
    a = 10 ** (-6 / 20)
    assert e.kind == "mls" and e.n_samples == 1023
    seq = max_len_seq(10)[0]
    np.testing.assert_allclose(e.signal, a * (2.0 * seq - 1.0))
    # MLS property: circular autocorrelation is a delta (N at lag 0, -1 elsewhere, unit amplitude)
    s = e.signal / a
    ac = np.fft.irfft(np.abs(np.fft.rfft(s)) ** 2, s.size)
    assert ac[0] == pytest.approx(1023.0)
    np.testing.assert_allclose(ac[1:], -1.0, atol=1e-9)
    with pytest.raises(ValueError):
        mls(1, FS)


@pytest.mark.parametrize("n", [4095, 4096])
def test_periodic_noise_fills_every_bin(n):
    # odd lengths have no Nyquist bin: the top bin must be filled like the others
    white = np.abs(np.fft.rfft(white_noise(n, FS, periodic=True).signal))
    np.testing.assert_allclose(white[1:], white[1], rtol=1e-9)
    pink = np.abs(np.fft.rfft(pink_noise(n, FS, periodic=True).signal))
    k = np.arange(1, pink.size)
    np.testing.assert_allclose(pink[1:] * np.sqrt(k), pink[1], rtol=1e-9)


def test_mls_sample_rate_validated():
    with pytest.raises(ValueError):
        mls(10, 0)
