import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.data import FrequencyResponse, ImpulseResponse
from resonus_core.measure import MeasurementResult, MeasurementSettings, measure

FS = 48000
TRUE_H = np.zeros(64)
TRUE_H[[3, 10, 25]] = [1.0, -0.5, 0.25]


def true_db(f):
    return 20 * np.log10(
        np.abs([np.sum(TRUE_H * np.exp(-2j * np.pi * fi * np.arange(64) / FS)) for fi in f])
    )


def check_response(fr, atol=0.2, scale=1.0, f_low=100.0):
    band = (fr.frequency >= f_low) & (fr.frequency <= 10000)
    np.testing.assert_allclose(
        fr.magnitude_db()[band], true_db(fr.frequency[band]) + 20 * np.log10(scale), atol=atol
    )


def test_sweep_measurement_recovers_system():
    s = MeasurementSettings(duration=1.0, tail=0.25, ir_length=0.2, fade_in=0.05, fade_out=0.005)
    r = measure(
        s, SimulatedIO(FS, impulse_responses=TRUE_H), software="test-app", software_version="9.9"
    )
    assert isinstance(r, MeasurementResult)
    assert len(r.impulse_responses) == 1 and len(r.frequency_responses) == 1
    ir, fr = r.impulse_responses[0], r.frequency_responses[0]
    assert isinstance(ir, ImpulseResponse) and isinstance(fr, FrequencyResponse)
    assert ir.data.size == int(0.2 * FS) and np.argmax(np.abs(ir.data)) == 3
    check_response(fr)
    assert ir.metadata.software == "test-app" and ir.metadata.software_version == "9.9"
    assert "simulated" in ir.metadata.device and ir.metadata.extra["settings"]["duration"] == 1.0
    assert r.recording.shape == (int(1.25 * FS), 1)


def test_sync_sweep():
    s = MeasurementSettings(
        excitation="sync_sweep",
        duration=1.0,
        tail=0.25,
        ir_length=0.2,
        fade_in=0.05,
        fade_out=0.005,
    )
    check_response(measure(s, SimulatedIO(FS, impulse_responses=TRUE_H)).frequency_responses[0])


def test_linear_sweep():
    # short fades; linear sweeps are weak at low frequencies (Müller & Massarani 2001)
    s = MeasurementSettings(
        excitation="linear_sweep",
        duration=1.0,
        tail=0.25,
        ir_length=0.2,
        fade_in=0.001,
        fade_out=0.001,
    )
    check_response(
        measure(s, SimulatedIO(FS, impulse_responses=TRUE_H)).frequency_responses[0], f_low=500.0
    )


@pytest.mark.parametrize("kind", ["white_noise", "pink_noise", "mls"])
def test_periodic_excitations(kind):
    s = MeasurementSettings(
        excitation=kind,
        duration=0.1,
        level=-20.0 if "noise" in kind else -6.0,
        repetitions=3,
        ir_length=0.05,
    )
    r = measure(s, SimulatedIO(FS, impulse_responses=TRUE_H))
    n = r.excitation.n_samples
    assert r.recording.shape == (4 * n, 1)  # repetitions + 1 settling period
    assert r.deconvolved.method == "periodic"
    assert np.argmax(np.abs(r.impulse_responses[0].data)) == 3
    check_response(r.frequency_responses[0], atol=0.05)


def test_repetitions_reduce_noise():
    def floor(reps):
        s = MeasurementSettings(duration=0.5, tail=0.25, ir_length=0.2, repetitions=reps)
        ir = (
            measure(s, SimulatedIO(FS, impulse_responses=TRUE_H, noise_rms=1e-2, seed=1))
            .impulse_responses[0]
            .data
        )
        return 10 * np.log10(np.mean(ir[1000:] ** 2))

    assert floor(1) - floor(4) == pytest.approx(10 * np.log10(4), abs=1.5)


def test_reference_channel_removes_latency_and_loopback_gain():
    # io inputs: input_channels, then the reference (D-17): io input 1 = system (device
    # input 2), io input 2 = loopback (device input 1; gain 0.5, latency 37)
    h = np.zeros((2, 1, 128))
    h[0, 0, 37 : 37 + 64] = TRUE_H
    h[1, 0, 37] = 0.5
    s = MeasurementSettings(
        duration=1.0,
        tail=0.25,
        ir_length=0.2,
        reference_channel=1,
        input_channels=(2,),
        fade_in=0.05,
        fade_out=0.005,
    )
    r = measure(s, SimulatedIO(FS, impulse_responses=h))
    assert len(r.impulse_responses) == 1  # reference channel excluded
    assert np.argmax(np.abs(r.impulse_responses[0].data)) == 3  # latency removed
    check_response(r.frequency_responses[0], scale=2.0)  # divided by the 0.5 loopback gain


def test_two_measurement_channels():
    h = np.zeros((2, 1, 64))
    h[0, 0] = TRUE_H
    h[1, 0] = 2 * TRUE_H
    r = measure(
        MeasurementSettings(
            duration=0.5, tail=0.25, ir_length=0.1, input_channels=(1, 2), input_labels=("m1", "m2")
        ),
        SimulatedIO(FS, impulse_responses=h),
    )
    a, b = (x.data for x in r.impulse_responses)
    np.testing.assert_allclose(b, 2 * a, atol=1e-9)


def test_measure_validation():
    with pytest.raises(ValueError, match="sample rate"):
        measure(MeasurementSettings(), SimulatedIO(44100))
    with pytest.raises(ValueError):
        measure(MeasurementSettings(duration=0.2, tail=0.1, reference_channel=2), SimulatedIO(FS))
    with pytest.raises(ValueError):
        measure(
            MeasurementSettings(duration=0.2, tail=0.1, reference_channel=1), SimulatedIO(FS)
        )  # nothing left
    with pytest.raises(ValueError):
        measure(
            MeasurementSettings(
                excitation="white_noise", duration=0.05, ir_length=0.2, level=-20.0
            ),
            SimulatedIO(FS),
        )
