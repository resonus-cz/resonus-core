import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.deconv import Deconvolved
from resonus_core.noise import h1_estimate, periodic_deconvolve, synchronous_average
from resonus_core.signals import exp_sweep, mls, pink_noise, white_noise

FS = 48000
TRUE_H = np.zeros(64)
TRUE_H[[3, 10, 25]] = [1.0, -0.5, 0.25]


def true_response(f):
    return np.array([np.sum(TRUE_H * np.exp(-2j * np.pi * fi * np.arange(64) / FS)) for fi in f])


# ------------------------------------------------------------------ synchronous averaging
def test_average_shapes_and_discard():
    x = np.arange(10.0)
    np.testing.assert_allclose(
        synchronous_average(x, 3, n_discard=1), [4.5, 5.5, 6.5]
    )  # periods 2, 3
    np.testing.assert_allclose(synchronous_average(x, 3, n_discard=0), [3.0, 4.0, 5.0])
    y = synchronous_average(np.column_stack([x, -x]), 3)
    assert y.shape == (3, 2)
    np.testing.assert_allclose(y[:, 1], -y[:, 0])


def test_average_noise_power_falls_by_k():
    rng = np.random.default_rng(0)
    clean = np.tile(rng.standard_normal(1024), 33)
    noisy = clean + rng.standard_normal(clean.size)
    avg = synchronous_average(noisy, 1024, n_discard=1)  # K = 32
    residual = avg - clean[:1024]
    assert 10 * np.log10(np.mean(residual**2)) == pytest.approx(-10 * np.log10(32), abs=0.5)


def test_average_validation():
    with pytest.raises(ValueError):
        synchronous_average(np.ones(10), 0)
    with pytest.raises(ValueError):
        synchronous_average(np.ones(10), 3, n_discard=-1)
    with pytest.raises(ValueError):
        synchronous_average(np.ones(5), 3, n_discard=1)  # K = 0


# ------------------------------------------------------------------ periodic deconvolution
@pytest.mark.parametrize(
    "exc",
    [white_noise(4096, FS, periodic=True), pink_noise(4096, FS, periodic=True), mls(12, FS)],
    ids=["white", "pink", "mls"],
)
def test_periodic_noise_recovers_system_exactly(exc):
    played = np.tile(exc.signal, 4)
    recorded = SimulatedIO(FS, impulse_responses=TRUE_H).playrec(played)
    d = periodic_deconvolve(synchronous_average(recorded, exc.n_samples), exc)
    assert isinstance(d, Deconvolved) and d.zero_index == 0 and d.method == "periodic"
    assert d.signal.shape == (exc.n_samples, 1)
    expected = np.zeros(exc.n_samples)
    expected[:64] = TRUE_H
    if exc.kind != "mls":  # DC not excited: the mean of the IR is removed
        expected -= TRUE_H.sum() / exc.n_samples
    np.testing.assert_allclose(d.signal[:, 0], expected, atol=1e-9)


def test_periodic_deconvolve_1d_and_validation():
    exc = white_noise(1024, FS, periodic=True)
    d = periodic_deconvolve(exc.signal, exc)
    assert d.signal.shape == (1024,)
    assert np.argmax(np.abs(d.signal)) == 0
    with pytest.raises(ValueError):
        periodic_deconvolve(np.ones(1000), exc)
    with pytest.raises(ValueError):
        rnd = white_noise(1024, FS)
        periodic_deconvolve(rnd.signal, rnd)
    with pytest.raises(ValueError):
        sw = exp_sweep(20.0, 20000.0, 0.1, FS)
        periodic_deconvolve(sw.signal, sw)


# ------------------------------------------------------------------ H1 and coherence
def test_h1_recovers_system_and_coherence_is_one_without_noise():
    x = white_noise(2**18, FS, seed=1).signal
    y = SimulatedIO(FS, impulse_responses=TRUE_H).playrec(x)[:, 0]
    f, h1, coh = h1_estimate(x, y, FS, nperseg=4096)
    assert f.size == 2049 and h1.shape == (2049,) and coh.shape == (2049,)
    band = (f >= 200) & (f <= 15000)
    ref = true_response(f[band])
    strong = np.abs(ref) > 0.3
    np.testing.assert_allclose(
        20 * np.log10(np.abs(h1[band][strong])), 20 * np.log10(np.abs(ref[strong])), atol=0.2
    )
    assert np.min(coh[band][strong]) > 0.99


def test_coherence_matches_theory_with_measurement_noise():
    # white x (rms sx) and white uncorrelated noise (rms sn): coh = |H|^2 sx^2 / (|H|^2 sx^2 + sn^2)
    x = white_noise(2**19, FS, rms_db=-20.0, seed=2).signal
    sx, sn = 0.1, 0.05
    y = SimulatedIO(FS, impulse_responses=TRUE_H, noise_rms=sn, seed=3).playrec(x)[:, 0]
    f, h1, coh = h1_estimate(x, y, FS, nperseg=2048)
    band = (f >= 200) & (f <= 15000)
    h2 = np.abs(true_response(f[band])) ** 2
    theory = h2 * sx**2 / (h2 * sx**2 + sn**2)
    assert np.median(np.abs(coh[band] - theory)) < 0.02


def test_h1_multichannel():
    x = white_noise(2**16, FS, seed=4).signal
    y = SimulatedIO(FS, impulse_responses=TRUE_H).playrec(x)[:, 0]
    f, h1, coh = h1_estimate(x, np.column_stack([y, 2 * y]), FS, nperseg=1024)
    assert h1.shape == (513, 2) and coh.shape == (513, 2)
    np.testing.assert_allclose(h1[:, 1], 2 * h1[:, 0], rtol=1e-9)


def test_h1_validation():
    with pytest.raises(ValueError):
        h1_estimate(np.ones((10, 2)), np.ones(10), FS, 4)
    with pytest.raises(ValueError):
        h1_estimate(np.ones(10), np.ones(12), FS, 4)
