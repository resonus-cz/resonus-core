import warnings

import numpy as np
import pytest
from scipy.signal import firwin

from resonus_core.window import OnsetWarning, align, envelope, normalize, onset, time_window

FS = 48000
ARRIVAL = 1000


def synthetic_ir(background_db=-80.0, spike=None, seed=0):
    """Band-limited direct sound at ARRIVAL, decaying tail, white background noise."""
    rng = np.random.default_rng(seed)
    n = 8000
    ir = 10 ** (background_db / 20) * rng.standard_normal(n)
    pulse = firwin(63, 15000, fs=FS)  # linear-phase low-pass, centre at tap 31
    pulse /= np.max(np.abs(pulse))
    ir[ARRIVAL - 31 : ARRIVAL + 32] += pulse
    t = np.arange(n - ARRIVAL - 100) / FS
    ir[ARRIVAL + 100 :] += 0.05 * np.exp(-t / 0.05) * rng.standard_normal(t.size)  # peaks < direct
    if spike is not None:
        ir[200] += spike
    return ir


def level(ir, i):
    return 20 * np.log10(envelope(ir)[i])


def test_envelope_of_tone_burst():
    t = np.arange(4800) / FS
    x = 0.5 * np.cos(2 * np.pi * 1000 * t)
    env = envelope(x)
    assert env.shape == x.shape
    np.testing.assert_allclose(env[500:-500], 0.5, atol=1e-3)
    two = envelope(np.column_stack([x, 2 * x]))
    np.testing.assert_allclose(two[500:-500, 1], 1.0, atol=2e-3)


def test_onset_meets_iso_conditions():
    ir = synthetic_ir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        i = onset(ir)
    lmax = 20 * np.log10(envelope(ir).max())
    assert ARRIVAL - 40 <= i <= ARRIVAL
    assert level(ir, i) <= lmax - 20  # more than 20 dB below the maximum
    assert level(ir, i) > -80 + 10  # significantly above the background (default margin)


def test_onset_ignores_early_spike():
    # a spike only 12 dB below the maximum would fool a first-crossing search
    ir = synthetic_ir(spike=10 ** (-12 / 20))
    assert ARRIVAL - 40 <= onset(ir) <= ARRIVAL


def test_explicit_background_and_margin():
    ir = synthetic_ir()
    loose = onset(ir, background_db=-80.0, margin_db=0.0)
    strict = onset(ir, background_db=-80.0, margin_db=40.0)
    assert loose <= strict <= ARRIVAL


def test_low_dynamic_range_warns_and_compromises():
    ir = synthetic_ir(background_db=-25.0)  # only ~5 dB between background+margin and -20 dB
    with pytest.warns(OnsetWarning):
        i = onset(ir)
    assert ARRIVAL - 40 <= i <= ARRIVAL


def test_background_above_minus_20_db_warns_and_uses_t20():
    ir = synthetic_ir(background_db=-15.0, seed=3)
    with pytest.warns(OnsetWarning):
        i = onset(ir)
    assert ARRIVAL - 40 <= i <= ARRIVAL


def test_onset_validation():
    with pytest.raises(ValueError):
        onset(np.zeros(100))
    with pytest.raises(ValueError):
        onset(np.ones((10, 2)))
    with pytest.raises(ValueError):
        onset(synthetic_ir(), margin_db=-1.0)
    with pytest.raises(ValueError):
        onset(synthetic_ir(), tail_fraction=0.0)


def test_align():
    x = np.arange(10.0)
    np.testing.assert_array_equal(align(x, 4, pre=1), [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])
    two = align(np.column_stack([x, -x]), 4)
    assert two.shape == (10, 2) and two[0, 1] == -4
    with pytest.raises(ValueError):
        align(x, 2, pre=3)
    with pytest.raises(ValueError):
        align(x, 10)


def test_time_window_shapes():
    w = time_window(100, 50, 10, 20)
    assert w.shape == (100,) and w[50] == 1.0
    assert np.all(w[:40] == 0) and np.all(w[70:] == 0)
    full = np.hanning(21)  # scipy hann(21, sym) == numpy hanning(21)
    np.testing.assert_allclose(w[40:50], full[:10])
    full_r = np.hanning(41)
    np.testing.assert_allclose(w[51:71][:19], full_r[21:40])
    assert np.all(np.diff(w[40:51]) >= 0) and np.all(np.diff(w[50:71]) <= 0)
    rect = time_window(100, 50, 10, 20, "rect", "rect")
    np.testing.assert_array_equal(rect, np.r_[np.zeros(41), np.ones(29), np.zeros(30)])
    hard = time_window(100, 50, 0, 0)
    assert hard[50] == 1.0 and hard.sum() == 1.0


def test_time_window_clipping_and_validation():
    w = time_window(30, 2, 10, 50)
    assert w.shape == (30,) and w[2] == 1.0
    for args in ((0, 0, 1, 1), (10, 10, 1, 1), (10, 5, -1, 1)):
        with pytest.raises(ValueError):
            time_window(*args)


def test_normalize():
    x = np.array([0.0, -2.0, 1.0, 1.0])
    assert np.max(np.abs(normalize(x))) == pytest.approx(1.0)
    assert np.sqrt(np.mean(normalize(x, "rms", 0.5) ** 2)) == pytest.approx(0.5)
    assert np.sum(normalize(x, "energy") ** 2) == pytest.approx(1.0)
    assert np.sum(normalize(x, "energy", 4.0) ** 2) == pytest.approx(4.0)
    two = normalize(np.column_stack([x, 2 * x]))
    np.testing.assert_allclose(two[:, 1], 2 * two[:, 0])  # joint scaling keeps ratios
    for kwargs in ({"mode": "max"}, {"target": 0.0}):
        with pytest.raises(ValueError):
            normalize(x, **kwargs)
    with pytest.raises(ValueError):
        normalize(np.zeros(4))
