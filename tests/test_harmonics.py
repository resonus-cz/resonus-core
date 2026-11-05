import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.deconv import deconvolve
from resonus_core.harmonics import (
    Harmonics,
    harmonic_delays,
    harmonic_responses,
    split_harmonics,
    thd,
)
from resonus_core.signals import exp_sweep, linear_sweep

FS = 48000


def run(nonlinear, h=None, synchronized=True):
    e = exp_sweep(
        20.0,
        20000.0,
        2.0,
        FS,
        peak_dbfs=0.0,
        synchronized=synchronized,
        fade_in=0.05,
        fade_out=0.005,
    )
    x = nonlinear(np.concatenate([e.signal, np.zeros(FS // 4)]))
    return e, deconvolve(SimulatedIO(FS, impulse_responses=h).playrec(x)[:, 0], e)


def cubic(x):
    # x^2 = (1 - cos 2p)/2, x^3 = (3 sin p - sin 3p)/4 for x = sin p:
    # H1 = 1 + 0.75*0.05 = 1.0375, |H2| = 0.05, |H3| = 0.0125
    return x + 0.1 * x**2 + 0.05 * x**3


def band(f):
    return (f >= 200) & (f <= 5000)


def test_delays():
    e = exp_sweep(20.0, 20000.0, 2.0, FS, synchronized=True)
    np.testing.assert_allclose(harmonic_delays(e, 4), e.params["rate"] * np.log([1, 2, 3, 4]))
    with pytest.raises(ValueError):
        harmonic_delays(linear_sweep(0.0, 20000.0, 1.0, FS), 3)
    with pytest.raises(ValueError):
        harmonic_delays(e, 0)


def test_linear_row_is_plain_cut():
    e, d = run(cubic)
    h = split_harmonics(d, e, 3, 2048, pre=256)
    assert isinstance(h, Harmonics) and h.irs.shape == (3, 2048) and h.pre == 256
    np.testing.assert_allclose(
        h.irs[0], d.signal[d.zero_index - 256 : d.zero_index - 256 + 2048], atol=1e-12
    )


@pytest.mark.parametrize("synchronized", [True, False])
def test_harmonic_magnitudes(synchronized):
    e, d = run(cubic, synchronized=synchronized)
    f, H = harmonic_responses(split_harmonics(d, e, 3, 2048, pre=256), n_fft=2**14)
    assert H.shape == (3, 2**13 + 1) and f.size == 2**13 + 1
    for k, expected in enumerate([1.0375, 0.05, 0.0125]):
        assert np.median(np.abs(H[k][band(f)])) == pytest.approx(expected, rel=5e-3)


def test_synchronized_sweep_gives_correct_harmonic_phase():
    # -0.05 cos(2p) = 0.05 sin(2p - pi/2): H2 has constant phase -pi/2 (Novák 2015)
    e, d = run(lambda x: x + 0.1 * x**2)
    f, H = harmonic_responses(split_harmonics(d, e, 2, 2048, pre=256), n_fft=2**14)
    phase = np.angle(H[1][band(f)])
    mean = np.angle(np.mean(np.exp(1j * phase)))
    assert mean == pytest.approx(-np.pi / 2, abs=0.01)
    assert np.max(np.abs(np.angle(np.exp(1j * (phase - mean))))) < 0.05
    lin = np.angle(H[0][band(f)])
    np.testing.assert_allclose(lin, 0.0, atol=0.05)  # linear part: zero phase at arrival


def test_plain_sweep_harmonic_phase_is_offset():
    e, d = run(lambda x: x + 0.1 * x**2, synchronized=False)
    f, H = harmonic_responses(split_harmonics(d, e, 2, 2048, pre=256), n_fft=2**14)
    mean = np.angle(np.mean(np.exp(1j * np.angle(H[1][band(f)]))))
    assert abs(np.angle(np.exp(1j * (mean + np.pi / 2)))) > 0.1


def test_thd_memoryless():
    e, d = run(cubic)
    h = split_harmonics(d, e, 3, 2048, pre=256)
    t = thd(h, np.array([500.0, 1000.0, 2000.0]))
    np.testing.assert_allclose(t, np.sqrt(0.05**2 + 0.0125**2) / 1.0375, rtol=1e-2)


def test_thd_evaluates_harmonic_at_n_times_f():
    # (x + 0.1 x^2) through g = [0.5, 0.5]: |G(f)| = cos(pi f/fs); THD(f) = 0.05 |G(2f)|/|G(f)|
    e, d = run(lambda x: x + 0.1 * x**2, h=[0.5, 0.5])
    t = thd(split_harmonics(d, e, 2, 2048, pre=256), np.array([6000.0]))
    assert t[0] == pytest.approx(0.05 * np.cos(np.pi / 4) / np.cos(np.pi / 8), rel=1e-2)


def test_thd_omits_harmonics_above_nyquist():
    e, d = run(cubic)
    assert thd(split_harmonics(d, e, 3, 2048, pre=256), np.array([15000.0]))[0] == 0.0


def test_overlap_and_bounds_validation():
    e, d = run(cubic)
    spacing_2_1 = e.params["rate"] * np.log(2) * FS  # ~9981 samples
    with pytest.raises(ValueError, match="overlap"):
        split_harmonics(d, e, 3, 12000, pre=256)  # harmonic 2 segment runs into harmonic 1
    with pytest.raises(ValueError, match="overlap"):
        split_harmonics(d, e, 2, 20000, pre=int(spacing_2_1))  # pre reaches harmonic 3 region
    with pytest.raises(ValueError):
        split_harmonics(d, e, 3, 256, pre=256)
    with pytest.raises(ValueError):
        split_harmonics(d, e, 3, 2048, pre=-1)
    with pytest.raises(ValueError):
        split_harmonics(d, e, 3, d.signal.size, pre=0)  # linear segment beyond the end
    lin = linear_sweep(0.0, 20000.0, 1.0, FS)
    with pytest.raises(ValueError):
        split_harmonics(d, lin, 2, 2048)


def test_multichannel_split_and_thd_guard():
    e, d = run(cubic)
    x = cubic(np.concatenate([e.signal, np.zeros(FS // 4)]))
    r = SimulatedIO(FS).playrec(x)[:, 0]
    two = deconvolve(np.column_stack([r, 2 * r]), e)
    h = split_harmonics(two, e, 3, 2048, pre=256)
    assert h.irs.shape == (3, 2048, 2)
    np.testing.assert_allclose(h.irs[..., 1], 2 * h.irs[..., 0], atol=1e-12)
    f, H = harmonic_responses(h)
    assert H.shape == (3, 1025, 2)
    with pytest.raises(ValueError):
        thd(h, np.array([1000.0]))
    with pytest.raises(ValueError):
        thd(split_harmonics(d, e, 1, 2048, pre=256), np.array([1000.0]))


def test_thd_channel_selection():
    e, d = run(cubic)
    x = cubic(np.concatenate([e.signal, np.zeros(FS // 4)]))
    r = SimulatedIO(FS).playrec(x)[:, 0]
    h = split_harmonics(deconvolve(np.column_stack([r, r]), e), e, 3, 2048, pre=256)
    single = split_harmonics(d, e, 3, 2048, pre=256)
    np.testing.assert_allclose(
        thd(h, np.array([1000.0]), channel=1), thd(single, np.array([1000.0])), rtol=1e-9
    )
    with pytest.raises(ValueError):
        thd(h, np.array([1000.0]), channel=2)
    with pytest.raises(ValueError):
        thd(single, np.array([1000.0]), channel=0)
