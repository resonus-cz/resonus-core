import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.deconv import deconvolve, novak_inverse_spectrum
from resonus_core.signals import exp_sweep

FS = 48000
TAIL = FS // 4
TRUE_H = np.zeros(64)
TRUE_H[[3, 10, 25]] = [1.0, -0.5, 0.25]


def measure(excitation, h=None, nonlinear=None):
    x = np.concatenate([excitation.signal, np.zeros(TAIL)])
    if nonlinear is not None:
        x = nonlinear(x)
    return SimulatedIO(FS, impulse_responses=h).playrec(x)[:, 0]


def mag_db(x, n_fft=2**14):
    return 20 * np.log10(np.abs(np.fft.rfft(x, n_fft))), np.fft.rfftfreq(n_fft, 1 / FS)


def centred(signal, index, half=2048):
    # band-limited responses ring symmetrically around their arrival (near zero-phase)
    return signal[index - half : index + half]


def test_novak_inverse_spectrum_formula():
    e = exp_sweep(20.0, 20000.0, 1.0, FS, peak_dbfs=-6.0, synchronized=True)
    L = e.params["rate"]
    f = np.array([0.0, 100.0, 1000.0, 5000.0])
    si = novak_inverse_spectrum(e, f)
    assert si.dtype == np.complex128 and si[0] == 0
    fp = f[1:]
    expected = (
        2
        * np.sqrt(fp / L)
        * np.exp(-1j * 2 * np.pi * fp * L * (1 - np.log(fp / 20.0)) + 1j * np.pi / 4)
    )
    np.testing.assert_allclose(si[1:], expected / 10 ** (-6 / 20), rtol=1e-12)


def test_novak_inverse_requires_synchronized():
    with pytest.raises(ValueError):
        novak_inverse_spectrum(exp_sweep(20.0, 20000.0, 1.0, FS), np.array([100.0]))


@pytest.mark.parametrize("fades", [(0.0, 0.0), (0.05, 0.005)])
def test_novak_deconvolution_recovers_system(fades):
    e = exp_sweep(20.0, 20000.0, 1.0, FS, synchronized=True, fade_in=fades[0], fade_out=fades[1])
    d = deconvolve(measure(e, TRUE_H), e)
    assert d.method == "novak" and d.zero_index == e.n_samples - 1
    db, f = mag_db(centred(d.signal, d.zero_index))
    true_db, _ = mag_db(TRUE_H)
    central = (f >= 100) & (f <= 10000)
    np.testing.assert_allclose(db[central], true_db[central], atol=0.2)
    assert np.argmax(np.abs(d.linear_ir(4096))) == 3


@pytest.mark.parametrize("method", ["inverse", "regularized"])
def test_other_methods_work_on_synchronized_sweeps(method):
    e = exp_sweep(20.0, 20000.0, 1.0, FS, synchronized=True)
    d = deconvolve(measure(e, TRUE_H), e, method=method)
    db, f = mag_db(centred(d.signal, d.zero_index))
    true_db, _ = mag_db(TRUE_H)
    central = (f >= 100) & (f <= 10000)
    np.testing.assert_allclose(db[central], true_db[central], atol=0.25)


def test_second_harmonic_level():
    # y = x + 0.1 x^2 with peak 1: x^2 = (1 - cos 2phi)/2, so |H2| = 0.05 (-26.02 dB);
    # its IR arrives L*ln(2) before the linear IR (Novák 2015).
    e = exp_sweep(
        20.0, 20000.0, 2.0, FS, peak_dbfs=0.0, synchronized=True, fade_in=0.05, fade_out=0.005
    )
    d = deconvolve(measure(e, nonlinear=lambda x: x + 0.1 * x**2), e)
    i = int(round(d.zero_index - e.params["rate"] * np.log(2) * FS))
    db, f = mag_db(centred(d.signal, i, 1024))
    band = (f >= 200) & (f <= 5000)
    assert np.median(db[band]) == pytest.approx(20 * np.log10(0.05), abs=0.1)


def test_novak_method_rejects_plain_sweep():
    e = exp_sweep(20.0, 20000.0, 0.5, FS)
    with pytest.raises(ValueError):
        deconvolve(measure(e), e, method="novak")


def test_novak_multichannel():
    e = exp_sweep(50.0, 10000.0, 0.5, FS, synchronized=True)
    r = measure(e, TRUE_H)
    two = deconvolve(np.column_stack([r, -r]), e)
    one = deconvolve(r, e)
    np.testing.assert_allclose(two.signal[:, 0], one.signal, atol=1e-12)
    np.testing.assert_allclose(two.signal[:, 1], -one.signal, atol=1e-12)
