import numpy as np
import pytest
from scipy.signal import chirp

from resonus_core.signals import exp_sweep

FS = 48000


def test_synchronization_condition_novak_eq_31_33():
    e = exp_sweep(20.0, 20000.0, 1.0, FS, synchronized=True)
    p = e.params
    k = p["f_start"] * p["rate"]
    assert k == pytest.approx(round(k), abs=1e-9) and round(k) == 3  # round(20*1.0/ln(1000))
    assert p["duration"] == pytest.approx(3 * np.log(1000.0) / 20.0)
    assert p["requested_duration"] == 1.0 and p["synchronized"] is True
    assert e.n_samples == round(p["duration"] * FS) and e.kind == "exp_sweep"


def test_synchronized_signal_formula_and_zero_phase():
    e = exp_sweep(50.0, 10000.0, 2.0, FS, peak_dbfs=-6.0, synchronized=True)
    L, a = e.params["rate"], 10 ** (-6 / 20)
    t = np.arange(e.n_samples) / FS
    np.testing.assert_allclose(
        e.signal, a * np.sin(2 * np.pi * 50.0 * L * np.exp(t / L)), atol=1e-9
    )
    assert abs(e.signal[0]) < 1e-9 and e.signal[1] > 0


def test_synchronized_fades():
    e = exp_sweep(20.0, 20000.0, 1.0, FS, synchronized=True, fade_in=0.1, fade_out=0.01)
    plain = exp_sweep(20.0, 20000.0, 1.0, FS, synchronized=True)
    m, k = 4800, 480
    np.testing.assert_allclose(
        e.signal[:m], plain.signal[:m] * (1 - np.cos(np.pi * np.arange(m) / m)) / 2
    )
    np.testing.assert_allclose(
        e.signal[-k:], plain.signal[-k:] * (1 + np.cos(np.pi * np.arange(k) / k)) / 2
    )


def test_default_sweep_unchanged():
    e = exp_sweep(50.0, 10000.0, 0.5, FS, peak_dbfs=0.0)
    t = np.arange(e.n_samples) / FS
    np.testing.assert_allclose(
        e.signal, chirp(t, 50.0, 0.5, 10000.0, method="logarithmic", phi=-90), atol=1e-9
    )
    assert not e.params.get("synchronized", False)


def test_synchronized_validation():
    with pytest.raises(ValueError):
        exp_sweep(20.0, 20000.0, 0.01, FS, synchronized=True)  # k = 0
    with pytest.raises(ValueError):
        exp_sweep(20.0, 30000.0, 1.0, FS, synchronized=True)  # above Nyquist
