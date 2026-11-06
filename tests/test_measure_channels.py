import sys
import types
from dataclasses import fields

import numpy as np
import pytest

from resonus_core.devices import COMMON_SAMPLE_RATES
from resonus_core.measure import MeasurementSettings, measurement_warnings, open_io

FS = 48000


def groups():
    return {f.name: f.metadata["group"] for f in fields(MeasurementSettings)}


def test_channel_defaults_and_groups():
    s = MeasurementSettings()
    assert (s.input_device, s.output_device) == ("", "")
    assert s.input_channels == (1,) and s.input_labels == ("m1",)
    assert s.output_channels == (1,) and s.output_labels == ("s1",)
    assert s.output_mode == "simultaneous" and s.reference_channel == 0
    g = groups()
    assert g["ir_length"] == g["method"] == "Processing"  # D-17: measurement processing
    assert g["input_device"] == g["output_device"] == g["sample_rate"] == "Device"
    for name in ("input_channels", "input_labels", "output_channels", "output_labels"):
        assert g[name] == "Channels"
    assert g["output_mode"] == g["reference_channel"] == "Channels"


def test_sample_rate_choices():
    meta = {f.name: f.metadata for f in fields(MeasurementSettings)}
    assert meta["sample_rate"]["choices"] == COMMON_SAMPLE_RATES
    assert MeasurementSettings(sample_rate=96000).sample_rate == 96000
    with pytest.raises(ValueError, match="one of"):
        MeasurementSettings(sample_rate=22050)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_channels": (1, 2)}, "input_labels"),  # one label per channel
        ({"output_labels": ("s1", "s2")}, "output_labels"),
        ({"input_channels": (1, 2), "input_labels": ("m1", "m1")}, "duplicate"),
        ({"input_labels": ("",)}, "empty"),
        ({"input_channels": (1, 2), "input_labels": ("m1", "m2"), "reference_channel": 2},
         "reference"),
        ({"output_channels": (1, 2), "output_labels": ("s1", "s2"),
          "output_mode": "sequential", "reference_channel": 2}, "sequential"),
        ({"output_mode": "both"}, "one of"),
        ({"input_channels": (0,)}, "minimum"),
    ],
)  # fmt: skip
def test_channel_relations(changes, message):
    with pytest.raises(ValueError, match=message):
        MeasurementSettings(**changes)


def test_valid_multichannel_settings():
    s = MeasurementSettings(
        input_channels=(1, 2),
        input_labels=("m1L", "m1R"),
        output_channels=(1, 2),
        output_labels=("s1", "s2"),
        output_mode="sequential",
    )
    assert s.input_labels == ("m1L", "m1R")
    MeasurementSettings(input_channels=(1,), reference_channel=2)  # loopback on input 2
    MeasurementSettings(output_mode="sequential", reference_channel=2)  # one output: fine


def test_periodic_ir_length_limited_by_period():
    # Mueller & Massarani 2001, 2.1: the period must be at least as long as the IR
    MeasurementSettings(excitation="white_noise", duration=0.5, ir_length=0.5)
    with pytest.raises(ValueError, match="period"):
        MeasurementSettings(excitation="pink_noise", duration=0.5, ir_length=0.51)
    # MLS period 2**15 - 1 = 32767 samples (0.6826 s) for 0.5 s at 48 kHz
    MeasurementSettings(excitation="mls", duration=0.5, ir_length=0.68)
    with pytest.raises(ValueError, match="period"):
        MeasurementSettings(excitation="mls", duration=0.5, ir_length=0.69)
    # sweeps: no error, a warning instead
    MeasurementSettings(duration=1.0, tail=0.5, ir_length=2.0)


def test_measurement_warnings():
    assert measurement_warnings(MeasurementSettings(tail=1.0, ir_length=1.0)) == []
    w = measurement_warnings(MeasurementSettings(tail=0.5, ir_length=2.0))
    assert len(w) == 1 and "silence" in w[0]
    noise = MeasurementSettings(excitation="white_noise", tail=0.1, ir_length=1.0)
    assert measurement_warnings(noise) == []  # the tail is not used for periodic signals
    pair = MeasurementSettings(input_device="ALSA: a", output_device="ALSA: b")
    w = measurement_warnings(pair)
    assert len(w) == 1 and "clock" in w[0]
    same = MeasurementSettings(input_device="ALSA: a", output_device="ALSA: a")
    assert measurement_warnings(same) == []
    assert measurement_warnings(MeasurementSettings(input_device="ALSA: a")) == []


def test_toml_roundtrip_with_channels(tmp_path):
    s = MeasurementSettings(
        input_device="JACK Audio Connection Kit: UR22",
        input_channels=(1, 2),
        input_labels=("m1L", "m1R"),
        reference_channel=0,
        sample_rate=96000,
    )
    s.save(tmp_path / "m.toml")
    assert MeasurementSettings.load(tmp_path / "m.toml") == s


@pytest.fixture
def fake_sd(monkeypatch, recording_streams):

    calls = {}
    rows = [
        {"name": "UR22", "index": 0, "hostapi": 0, "max_input_channels": 2,
         "max_output_channels": 2, "default_samplerate": 48000.0},
        {"name": "Loop", "index": 1, "hostapi": 0, "max_input_channels": 4,
         "max_output_channels": 4, "default_samplerate": 48000.0},
    ]  # fmt: skip

    mod = types.SimpleNamespace(
        **recording_streams(calls),
        query_devices=lambda device=None: rows if device is None else rows[device],
        query_hostapis=lambda: ({"name": "ALSA"},),
        PortAudioError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", mod)
    return calls


def test_open_io_maps_devices_and_channels(fake_sd):
    s = MeasurementSettings(
        input_device="ALSA: Loop",
        input_channels=(2, 1),
        input_labels=("m1", "m2"),
        reference_channel=4,
        output_channels=(3,),
    )
    io = open_io(s)
    assert io.sample_rate == FS and io.n_inputs == 3 and io.n_outputs == 1
    io.playrec(np.ones(4))
    kw = fake_sd["playrec"]
    assert kw["device"] == (1, None)  # input device resolved by key, output default
    # measurement channels, then the reference; output 3 (streams open up to the
    # highest channel number and map inside the callback, Q-C10)
    assert kw["channels"] == (4, 3)
    assert "[2, 1, 4]" in io.describe() and "[3]" in io.describe()
    io = open_io(MeasurementSettings())
    io.playrec(np.ones(4))
    assert fake_sd["playrec"]["device"] is None and fake_sd["playrec"]["channels"] == (1, 1)
    with pytest.raises(ValueError, match="not found"):
        open_io(MeasurementSettings(output_device="ALSA: nothing"))
