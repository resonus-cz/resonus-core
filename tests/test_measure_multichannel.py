"""Multichannel measure(): io channel order, output modes, channel metadata.

resonus-room D-17 (user decisions): io inputs are the
``input_channels`` followed by the reference; ``"simultaneous"`` plays all outputs at
once (one IR per input), ``"sequential"`` one output after the other (one IR per input
and output, output-major). Reference values come from linearity: each path measured
alone through a 1x1 simulated device.
"""

import numpy as np
import pytest

from resonus_core.audio import SimulatedIO
from resonus_core.measure import MeasurementSettings, measure

FS = 48000
BASE = {"duration": 0.5, "tail": 0.25, "ir_length": 0.1, "fade_in": 0.02, "fade_out": 0.005}


def paths():
    """Four distinct paths h[input, output] of a 2-input, 2-output system."""
    h = np.zeros((2, 2, 64))
    h[0, 0, 3], h[0, 1, 7], h[1, 0, 11], h[1, 1, 19] = 1.0, -0.5, 0.25, 0.75
    h[0, 0, 30], h[1, 1, 40] = 0.1, -0.2
    return h


def single(h, **kw):
    """IR of one path measured alone (the reference via linearity)."""
    s = MeasurementSettings(**{**BASE, **kw})
    return measure(s, SimulatedIO(FS, impulse_responses=h)).impulse_responses[0].data


def two_by_two(**kw):
    return MeasurementSettings(
        **BASE,
        input_channels=(1, 2),
        input_labels=("m1", "m2"),
        output_channels=(1, 2),
        output_labels=("s1", "s2"),
        **kw,
    )


def test_sequential_one_ir_per_input_and_output():
    h = paths()
    r = measure(two_by_two(output_mode="sequential"), SimulatedIO(FS, impulse_responses=h))
    assert len(r.impulse_responses) == len(r.frequency_responses) == 4
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]  # output-major: (input, output)
    for ir, (i, j) in zip(r.impulse_responses, order, strict=True):
        np.testing.assert_allclose(ir.data, single(h[i, j]), atol=1e-9)
    labels = [ir.metadata.extra["channel"]["label"] for ir in r.impulse_responses]
    assert labels == ["s1m1", "s1m2", "s2m1", "s2m2"]  # naming s#m# (D-17)
    assert len(r.recordings) == len(r.deconvolutions) == 2  # one run per output
    assert r.recording is r.recordings[0] and r.deconvolved is r.deconvolutions[0]
    n = int(0.75 * FS)
    assert all(rec.shape == (n, 2) for rec in r.recordings)


def test_sequential_plays_one_output_at_a_time():
    played = []

    class Spy(SimulatedIO):
        def playrec(self, signal):
            played.append(np.asarray(signal))
            return super().playrec(signal)

    measure(two_by_two(output_mode="sequential"), Spy(FS, impulse_responses=paths()))
    assert len(played) == 2
    for j, x in enumerate(played):
        assert x.ndim == 2 and x.shape[1] == 2
        assert np.any(x[:, j]) and not np.any(x[:, 1 - j])


def test_simultaneous_sums_the_outputs():
    h = paths()
    r = measure(two_by_two(), SimulatedIO(FS, impulse_responses=h))
    assert len(r.impulse_responses) == 2 and len(r.recordings) == 1
    for i, ir in enumerate(r.impulse_responses):
        np.testing.assert_allclose(ir.data, single(h[i, 0] + h[i, 1]), atol=1e-9)
    ch = r.impulse_responses[1].metadata.extra["channel"]
    assert ch == {
        "input": 2,
        "input_label": "m2",
        "outputs": [1, 2],
        "output_labels": ["s1", "s2"],
        "label": "s1+s2m2",
    }


def test_sequential_periodic_excitation():
    h = paths()
    s = two_by_two(output_mode="sequential", excitation="mls", repetitions=2)
    r = measure(s, SimulatedIO(FS, impulse_responses=h))
    ref = single(h[1, 0], excitation="mls", repetitions=2)
    np.testing.assert_allclose(r.impulse_responses[1].data, ref, atol=1e-9)
    assert len(r.recordings) == 2


def test_single_channel_metadata_and_modes_agree():
    h = np.zeros(64)
    h[5] = 1.0
    io = SimulatedIO(FS, impulse_responses=h)
    a = measure(MeasurementSettings(**BASE), io)
    b = measure(MeasurementSettings(**BASE, output_mode="sequential"), io)
    np.testing.assert_allclose(a.impulse_responses[0].data, b.impulse_responses[0].data)
    extra = a.impulse_responses[0].metadata.extra
    assert extra["channel"] == {
        "input": 1,
        "input_label": "m1",
        "outputs": [1],
        "output_labels": ["s1"],
        "label": "s1m1",
    }
    assert extra["excitation"] == "exp_sweep" and "settings" in extra
    assert a.frequency_responses[0].metadata.extra["channel"]["label"] == "s1m1"


def test_reference_is_the_last_io_input():
    # io inputs: m1, m2, then the loopback; latency 29 removed, loopback gain 0.5 divided out
    h = np.zeros((3, 1, 128))
    h[0, 0, 29 + 3] = 1.0
    h[1, 0, 29 + 8] = -2.0
    h[2, 0, 29] = 0.5
    s = MeasurementSettings(
        **BASE, input_channels=(1, 2), input_labels=("m1", "m2"), reference_channel=3
    )
    r = measure(s, SimulatedIO(FS, impulse_responses=h))
    assert len(r.impulse_responses) == 2
    a, b = (ir.data for ir in r.impulse_responses)
    assert np.argmax(np.abs(a)) == 3 and np.argmax(np.abs(b)) == 8
    # each input equals that path measured alone with the same loopback
    for k, ir in enumerate((a, b)):
        alone = MeasurementSettings(**BASE, reference_channel=2)
        r1 = measure(alone, SimulatedIO(FS, impulse_responses=h[[k, 2]]))
        np.testing.assert_allclose(ir, r1.impulse_responses[0].data, atol=1e-9)


@pytest.mark.parametrize(
    ("settings", "io_shape", "match"),
    [
        ({}, (2, 1), "inputs"),  # two io inputs, one input channel
        ({"reference_channel": 2}, (1, 1), "inputs"),  # reference not recorded
        ({}, (1, 2), "outputs"),
        ({"output_channels": (1, 2), "output_labels": ("s1", "s2")}, (1, 1), "outputs"),
    ],
)
def test_io_must_match_the_channel_settings(settings, io_shape, match):
    n_in, n_out = io_shape
    h = np.zeros((n_in, n_out, 8))
    h[:, :, 0] = 1.0
    with pytest.raises(ValueError, match=match):
        measure(MeasurementSettings(**BASE, **settings), SimulatedIO(FS, impulse_responses=h))
