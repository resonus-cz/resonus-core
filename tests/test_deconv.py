import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.deconv import (
    Deconvolved,
    deconvolve,
    inverse_filter,
    regularized_inverse_spectrum,
)
from resonus_core.signals import exp_sweep, linear_sweep, mls, white_noise

FS = 48000
TAIL = FS // 4


def measure(excitation, h=None, latency=0, nonlinear=None):
    """Play excitation + silent tail through a simulated system; returns 1-D recording."""
    x = np.concatenate([excitation.signal, np.zeros(TAIL)])
    if nonlinear is not None:
        x = nonlinear(x)
    return SimulatedIO(FS, impulse_responses=h, latency_samples=latency).playrec(x)[:, 0]


def mag_db(ir, n_fft=2**14):
    return 20 * np.log10(np.abs(np.fft.rfft(ir, n_fft))), np.fft.rfftfreq(n_fft, 1 / FS)


def centred(d, half=2048):
    # Band-limited deconvolution is near zero-phase: its ringing is symmetric around t = 0.
    # Magnitude checks therefore use a window centred on t = 0 (a time shift does not
    # change the magnitude); cutting at t = 0 would lose ~0.75 dB for a 20 Hz-20 kHz band.
    return d.signal[d.zero_index - half : d.zero_index + half]


TRUE_H = np.zeros(64)
TRUE_H[[3, 10, 25]] = [1.0, -0.5, 0.25]


# ------------------------------------------------------------------ inverse filter
def test_inverse_filter_is_modulated_reversed_sweep():
    e = exp_sweep(20.0, 20000.0, 1.0, FS)
    f = inverse_filter(e)
    assert f.shape == e.signal.shape and f.dtype == np.float64
    tau = np.arange(e.n_samples) / FS
    ratio = f / (e.signal[::-1] * np.exp(-tau / e.params["rate"]))
    finite = np.abs(e.signal[::-1]) > 1e-3
    np.testing.assert_allclose(ratio[finite], ratio[finite][0], rtol=1e-9)


def test_inverse_filter_linear_is_reversed_sweep():
    e = linear_sweep(0.0, 24000.0, 0.5, FS)
    f = inverse_filter(e)
    finite = np.abs(e.signal[::-1]) > 1e-3
    ratio = f[finite] / e.signal[::-1][finite]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


@pytest.mark.parametrize(
    "sweep", [exp_sweep(20.0, 20000.0, 1.0, FS), linear_sweep(0.0, 24000.0, 1.0, FS)]
)
def test_sweep_times_inverse_is_unit_gain(sweep):
    g = np.abs(np.fft.rfft(sweep.signal, 2**17) * np.fft.rfft(inverse_filter(sweep), 2**17))
    f = np.fft.rfftfreq(2**17, 1 / FS)
    central = (f >= 200) & (f <= 2000) if sweep.kind == "exp_sweep" else (f >= 6000) & (f <= 18000)
    assert np.median(g[central]) == pytest.approx(1.0, rel=1e-3)


def test_inverse_filter_rejects_noise():
    with pytest.raises(ValueError):
        inverse_filter(white_noise(1024, FS))


# ------------------------------------------------------------------ regularized inverse
def test_regularization_profile():
    f = np.array(
        [0.0, 50.0, 100.0, 1000.0, 2000.0, 2000.0 * 2 ** (1 / 6), 2000.0 * 2 ** (1 / 3), 8000.0]
    )
    c = regularized_inverse_spectrum(np.ones(f.size), f, (100.0, 2000.0), eps_in=1e-6, eps_out=1.0)
    assert c.dtype == np.complex128
    np.testing.assert_allclose(c[[2, 3, 4]], 1 / (1 + 1e-6))  # in band, edges included
    np.testing.assert_allclose(c[[0, 1, 6, 7]], 0.5)  # f=0, >=1/3 octave outside
    np.testing.assert_allclose(c[5], 1 / (1 + 1e-3))  # log-midpoint of the transition


def test_regularization_scales_with_peak_power_and_conjugates():
    x = np.array([2.0 + 0j, 1j, 4.0])
    f = np.array([100.0, 200.0, 300.0])
    c = regularized_inverse_spectrum(x, f, (50.0, 1000.0), eps_in=1e-2)
    np.testing.assert_allclose(c, np.conj(x) / (np.abs(x) ** 2 + 1e-2 * 16.0))


def test_zero_low_edge_has_no_low_transition():
    f = np.array([0.0, 10.0, 100.0])
    c = regularized_inverse_spectrum(np.ones(3), f, (0.0, 1000.0))
    np.testing.assert_allclose(c, 1 / (1 + 1e-6))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"band": (100.0, 50.0)},
        {"band": (-1.0, 50.0)},
        {"eps_in": 0.0},
        {"eps_in": 2.0, "eps_out": 1.0},
        {"transition": 0.0},
    ],
)
def test_regularization_validation(kwargs):
    args = {
        "spectrum": np.ones(3),
        "frequencies": np.array([1.0, 2.0, 3.0]),
        "band": (1.0, 2.0),
    } | kwargs
    with pytest.raises(ValueError):
        regularized_inverse_spectrum(**args)
    with pytest.raises(ValueError):
        regularized_inverse_spectrum(np.ones(3), np.ones(2), (1.0, 2.0))


# ------------------------------------------------------------------ deconvolve
@pytest.mark.parametrize("method", ["inverse", "regularized"])
def test_wire_gives_flat_unit_response(method):
    e = exp_sweep(20.0, 20000.0, 1.0, FS, fade_in=0.05, fade_out=0.005)
    d = deconvolve(measure(e), e, method=method)
    assert isinstance(d, Deconvolved) and d.method == method
    assert d.zero_index == e.n_samples - 1
    assert np.argmax(np.abs(d.linear_ir(4096))) == 0
    db, f = mag_db(centred(d))
    central = (f >= 100) & (f <= 10000)
    np.testing.assert_allclose(db[central], 0.0, atol=0.2)


@pytest.mark.parametrize("method", ["inverse", "regularized"])
def test_known_system_recovered(method):
    e = exp_sweep(20.0, 20000.0, 1.0, FS, fade_in=0.05, fade_out=0.005)
    d = deconvolve(measure(e, TRUE_H), e, method=method)
    db, f = mag_db(centred(d))
    true_db, _ = mag_db(TRUE_H)
    central = (f >= 100) & (f <= 10000)
    np.testing.assert_allclose(db[central], true_db[central], atol=0.2)
    assert np.argmax(np.abs(d.linear_ir(4096))) == 3


def test_latency_shifts_ir():
    e = exp_sweep(20.0, 20000.0, 1.0, FS)
    ir = deconvolve(measure(e, latency=100), e).linear_ir(4096)
    assert np.argmax(np.abs(ir)) == 100


@pytest.mark.parametrize("exc", [white_noise(2**15, FS, periodic=True), mls(15, FS)])
def test_regularized_with_noise_and_mls(exc):
    d = deconvolve(measure(exc, TRUE_H), exc)
    assert d.method == "regularized" and d.zero_index == exc.n_samples - 1
    ir = d.linear_ir(64)
    np.testing.assert_allclose(ir, TRUE_H, atol=1e-3)


def test_harmonic_products_appear_before_zero():
    # memoryless 2nd-order distortion: its IR packs at -L*ln(2) (Farina 2000)
    e = exp_sweep(20.0, 20000.0, 2.0, FS, peak_dbfs=0.0, fade_in=0.05, fade_out=0.005)
    d = deconvolve(measure(e, nonlinear=lambda x: x + 0.1 * x**2), e)
    lag = int(round(e.params["rate"] * np.log(2) * FS))
    around = np.abs(d.signal[d.zero_index - lag - 200 : d.zero_index - lag + 200])
    quiet = np.abs(d.signal[d.zero_index - lag // 2 - 200 : d.zero_index - lag // 2 + 200])
    assert around.max() > 20 * quiet.max()
    assert d.time[d.zero_index] == 0.0 and d.time[0] < 0


@pytest.mark.parametrize("method", ["inverse", "regularized"])
def test_multichannel_matches_single_channel(method):
    e = exp_sweep(50.0, 10000.0, 0.5, FS)
    r = measure(e, TRUE_H)
    two = deconvolve(np.column_stack([r, 2 * r]), e, method=method)
    one = deconvolve(r, e, method=method)
    assert two.signal.shape == (one.signal.size, 2)
    np.testing.assert_allclose(two.signal[:, 0], one.signal, atol=1e-12)
    np.testing.assert_allclose(two.signal[:, 1], 2 * one.signal, atol=1e-12)


def test_linear_ir_and_time():
    d = Deconvolved(np.arange(10.0), FS, 4, "inverse")
    np.testing.assert_array_equal(d.linear_ir(3), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(d.time, (np.arange(10) - 4) / FS)
    with pytest.raises(ValueError):
        d.linear_ir(0)
    with pytest.raises(ValueError):
        d.linear_ir(7)


def test_deconvolve_validation():
    e = exp_sweep(20.0, 20000.0, 0.5, FS)
    with pytest.raises(ValueError, match="short"):
        deconvolve(np.zeros(100), e)
    with pytest.raises(ValueError):
        deconvolve(np.zeros(e.n_samples), e, method="magic")
    noise = white_noise(1024, FS)
    with pytest.raises(ValueError):
        deconvolve(np.zeros(2048), noise, method="inverse")
