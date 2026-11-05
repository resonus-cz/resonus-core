import numpy as np
import pytest

from resonus_core.measure import MeasurementSettings, make_excitation

FS = 48000


def test_default_is_exp_sweep():
    e = make_excitation(MeasurementSettings())
    assert e.kind == "exp_sweep" and e.n_samples == 3 * FS
    assert np.max(np.abs(e.signal)) == pytest.approx(10 ** (-12 / 20), rel=1e-3)
    assert not e.params.get("synchronized", False)


def test_sync_and_linear_sweeps():
    e = make_excitation(MeasurementSettings(excitation="sync_sweep", duration=1.0))
    assert e.kind == "exp_sweep" and e.params["synchronized"] is True
    e = make_excitation(
        MeasurementSettings(excitation="linear_sweep", f_start=0.0, duration=0.5, fade_out=0.01)
    )
    assert e.kind == "linear_sweep" and e.params["fade_out"] == 0.01


@pytest.mark.parametrize("kind", ["white_noise", "pink_noise"])
def test_noise_is_periodic_with_rms_level(kind):
    s = MeasurementSettings(excitation=kind, duration=0.1, level=-20.0, ir_length=0.1)
    e = make_excitation(s)
    assert e.kind == kind and e.params["periodic"] is True and e.n_samples == 4800
    assert 20 * np.log10(np.sqrt(np.mean(e.signal**2))) == pytest.approx(-20.0, abs=1e-9)


def test_mls_length_from_duration():
    s = MeasurementSettings(excitation="mls", duration=0.1, level=-6.0, ir_length=0.1)
    e = make_excitation(s)
    assert e.kind == "mls" and e.n_samples == 2**13 - 1  # ceil(log2(4801)) = 13
    assert np.max(np.abs(e.signal)) == pytest.approx(10 ** (-6 / 20))


def test_invalid_combinations_raise():
    with pytest.raises(ValueError):
        make_excitation(MeasurementSettings(f_stop=30000.0))  # above Nyquist at 48 kHz
    with pytest.raises(ValueError):
        MeasurementSettings(excitation="chirp")


def test_settings_roundtrip(tmp_path):
    s = MeasurementSettings(excitation="pink_noise", repetitions=8, reference_channel=2)
    s.save(tmp_path / "m.toml")
    assert MeasurementSettings.load(tmp_path / "m.toml") == s
