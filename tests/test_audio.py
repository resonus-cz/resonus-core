import sys
import types

import numpy as np
import pytest

from resonus_core.audio import SimulatedIO, SounddeviceIO

FS = 48000


def impulse(n, at=0, value=1.0):
    h = np.zeros(n)
    h[at] = value
    return h


# ------------------------------------------------------------------ SimulatedIO
def test_default_is_mono_wire():
    io = SimulatedIO(FS)
    assert (io.n_inputs, io.n_outputs) == (1, 1)
    x = np.random.default_rng(1).standard_normal(1000)
    y = io.playrec(x)
    assert y.shape == (1000, 1) and y.dtype == np.float64
    np.testing.assert_allclose(y[:, 0], x, atol=1e-12)


def test_1d_impulse_response_with_latency():
    io = SimulatedIO(FS, impulse_responses=[0.0, 0.5], latency_samples=3)
    y = io.playrec(impulse(10))
    np.testing.assert_allclose(y[:, 0], impulse(10, at=4, value=0.5), atol=1e-12)


def test_matrix_paths_and_mixing():
    # 2 inputs x 2 outputs: in1 = out1 delayed 1 + 0.5*out2; in2 = 2*out2
    h = np.zeros((2, 2, 4))
    h[0, 0] = impulse(4, at=1)
    h[0, 1] = impulse(4, value=0.5)
    h[1, 1] = impulse(4, value=2.0)
    io = SimulatedIO(FS, impulse_responses=h)
    assert (io.n_inputs, io.n_outputs) == (2, 2)
    x = np.zeros((8, 2))
    x[0, 0] = 1.0
    x[2, 1] = 1.0
    y = io.playrec(x)
    assert y.shape == (8, 2)
    np.testing.assert_allclose(y[:, 0], impulse(8, at=1) + impulse(8, at=2, value=0.5), atol=1e-12)
    np.testing.assert_allclose(y[:, 1], impulse(8, at=2, value=2.0), atol=1e-12)


def test_1d_signal_goes_to_every_output():
    h = np.zeros((1, 2, 1))
    h[0, 0, 0] = 1.0
    h[0, 1, 0] = 1.0
    y = SimulatedIO(FS, impulse_responses=h).playrec(np.ones(5))
    np.testing.assert_allclose(y[:, 0], 2 * np.ones(5))


def test_output_truncated_to_signal_length():
    io = SimulatedIO(FS, impulse_responses=np.ones(100), latency_samples=50)
    assert io.playrec(np.ones(20)).shape == (20, 1)


def test_noise_independent_reproducible_scaled():
    h = np.zeros((2, 1, 1))
    h[:, 0, 0] = 1.0
    a = SimulatedIO(FS, impulse_responses=h, noise_rms=0.1, seed=7).rec(FS)
    b = SimulatedIO(FS, impulse_responses=h, noise_rms=0.1, seed=7).rec(FS)
    assert a.shape == (FS, 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.sqrt(np.mean(a**2, axis=0)), [0.1, 0.1], rtol=0.02)
    assert abs(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) < 0.02


def test_no_noise_rec_is_silent():
    assert not np.any(SimulatedIO(FS).rec(100))


def test_simulated_validation():
    for kwargs in (
        {"sample_rate": 0},
        {"sample_rate": FS, "latency_samples": -1},
        {"sample_rate": FS, "noise_rms": -0.1},
        {"sample_rate": FS, "impulse_responses": np.ones((2, 3))},
        {"sample_rate": FS, "impulse_responses": []},
    ):
        with pytest.raises(ValueError):
            SimulatedIO(**kwargs)
    io = SimulatedIO(FS, impulse_responses=np.ones((1, 2, 3)))
    with pytest.raises(ValueError):
        io.playrec(np.ones((10, 3)))  # 3 columns, 2 outputs
    with pytest.raises(ValueError):
        io.playrec(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        io.playrec([])
    with pytest.raises(ValueError):
        io.rec(0)


def test_simulated_describe():
    d = SimulatedIO(FS, impulse_responses=np.ones((2, 1, 3))).describe()
    assert "simulated" in d and "2 in" in d and "1 out" in d


# ------------------------------------------------------------------ SounddeviceIO (mocked)
@pytest.fixture
def fake_sd(monkeypatch):
    """Only query_devices: playrec and rec run on streams since 0.11.0 (test_audio_stop.py)."""

    def query_devices(device=None):
        if device is None:
            return [{"name": "dev0"}, {"name": "dev1"}]
        return {"name": f"card {device}"}

    mod = types.SimpleNamespace(query_devices=query_devices)
    monkeypatch.setitem(sys.modules, "sounddevice", mod)


def test_sounddevice_describe_and_devices(fake_sd):
    assert "card 3" in SounddeviceIO(FS, device=3).describe()
    assert "default" in SounddeviceIO(FS).describe()
    assert [d["name"] for d in SounddeviceIO.devices()] == ["dev0", "dev1"]


def test_sounddevice_validation():
    for kwargs in (
        {"sample_rate": 0},
        {"sample_rate": FS, "input_channels": ()},
        {"sample_rate": FS, "input_channels": (0,)},
        {"sample_rate": FS, "output_channels": (1, 1)},
    ):
        with pytest.raises(ValueError):
            SounddeviceIO(**kwargs)
