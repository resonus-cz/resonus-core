import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.continuous import ContinuousAnalyzer, cyclic_block
from resonus_core.noise import periodic_deconvolve, synchronous_average
from resonus_core.signals import exp_sweep, mls, white_noise

FS = 48000
N = 1024
TRUE_H = np.zeros(64)
TRUE_H[[3, 10, 25]] = [1.0, -0.5, 0.25]


def exc():
    return white_noise(N, FS, periodic=True, seed=5)


def feed_in_chunks(analyzer, data, sizes):
    pos, done = 0, 0
    for s in sizes:
        done += analyzer.feed(data[pos : pos + s])
        pos += s
    done += analyzer.feed(data[pos:])
    return done


def test_cyclic_block_wraps():
    s = np.arange(5.0)
    np.testing.assert_array_equal(cyclic_block(s, 3, 6), [3, 4, 0, 1, 2, 3])
    np.testing.assert_array_equal(cyclic_block(s, 12, 2), [2, 3])
    assert cyclic_block(s, 0, 0).size == 0
    with pytest.raises(ValueError):
        cyclic_block([], 0, 3)


def test_chunking_does_not_matter():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(N * 6)
    a, b = ContinuousAnalyzer(exc()), ContinuousAnalyzer(exc())
    assert a.feed(data) == 6
    sizes = rng.integers(1, 700, size=20)
    assert feed_in_chunks(b, data, sizes) == 6
    assert a.n_averages == b.n_averages == 5
    np.testing.assert_allclose(a.average(), b.average(), atol=1e-12)


def test_linear_average_equals_synchronous_average():
    data = np.random.default_rng(1).standard_normal((N * 5 + 300, 2))
    a = ContinuousAnalyzer(exc(), n_channels=2, n_discard=1)
    a.feed(data)
    assert a.average().shape == (N, 2) and a.n_averages == 4
    np.testing.assert_allclose(a.average(), synchronous_average(data, N, n_discard=1), atol=1e-12)


def test_settling_periods_are_ignored():
    data = np.ones(N * 3)
    data[:N] = 100.0
    a = ContinuousAnalyzer(exc(), n_discard=1)
    a.feed(data)
    np.testing.assert_allclose(a.average()[:, 0], 1.0)


def test_exponential_tracks_a_change():
    a = ContinuousAnalyzer(exc(), averaging="exponential", n_exponential=4, n_discard=0)
    a.feed(np.zeros(N * 8))  # warm up on A = 0
    a.feed(np.ones(N * 3))  # then B = 1 for 3 periods
    # avg = B + (A - B)*(3/4)**3
    np.testing.assert_allclose(a.average()[:, 0], 1 - 0.75**3, atol=1e-12)
    lin = ContinuousAnalyzer(exc(), n_discard=0)
    lin.feed(np.zeros(N * 8))
    lin.feed(np.ones(N * 3))
    np.testing.assert_allclose(lin.average()[:, 0], 3 / 11, atol=1e-12)


def test_exponential_warm_start_is_cumulative_mean():
    a = ContinuousAnalyzer(exc(), averaging="exponential", n_exponential=4, n_discard=0)
    a.feed(np.concatenate([np.full(N, 1.0), np.full(N, 3.0)]))
    np.testing.assert_allclose(a.average()[:, 0], 2.0)


def test_continuous_measurement_recovers_system():
    e = exc()
    played = cyclic_block(e.signal, 0, N * 9)
    recorded = SimulatedIO(FS, impulse_responses=TRUE_H, noise_rms=1e-3, seed=2).playrec(played)
    a = ContinuousAnalyzer(e)
    feed_in_chunks(a, recorded, [1000] * 8)
    d = a.result()
    assert a.n_averages == 8 and d.method == "periodic" and d.signal.shape == (N, 1)
    expected = np.zeros(N)
    expected[:64] = TRUE_H
    expected -= TRUE_H.sum() / N  # DC not excited (see periodic_deconvolve)
    np.testing.assert_allclose(d.signal[:, 0], expected, atol=5e-3)
    np.testing.assert_allclose(d.signal, periodic_deconvolve(a.average(), e).signal, atol=1e-12)


def test_reset_and_errors():
    a = ContinuousAnalyzer(exc())
    with pytest.raises(ValueError):
        a.average()
    with pytest.raises(ValueError):
        a.result()
    a.feed(np.ones(N * 3))
    a.reset()
    assert a.n_averages == 0
    with pytest.raises(ValueError):
        a.average()
    with pytest.raises(ValueError):
        a.feed(np.ones((10, 2)))  # 1 channel expected
    for kwargs in (
        {"n_channels": 0},
        {"averaging": "median"},
        {"n_exponential": 0},
        {"n_discard": -1},
    ):
        with pytest.raises(ValueError):
            ContinuousAnalyzer(exc(), **kwargs)
    with pytest.raises(ValueError):
        ContinuousAnalyzer(exp_sweep(20.0, 20000.0, 0.1, FS))
    ContinuousAnalyzer(mls(10, FS))  # MLS is periodic
